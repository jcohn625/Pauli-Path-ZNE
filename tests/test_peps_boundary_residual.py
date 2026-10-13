"""Independent exact-contraction checks of retained boundary residuals."""
import math
import os
import numpy as np
import pytest
import peps_test_support as setup
from peps_boundary_residual import scalar,forward_residuals,correct_residuals,streaming_residuals


@pytest.fixture(autouse=True)
def safe_sparse_contractions():
    original=setup.get_lib_fn('numpy','tensordot')
    setup.register_function('numpy','tensordot',setup._safe_tensordot)
    try:yield
    finally:setup.register_function('numpy','tensordot',original)


@pytest.mark.parametrize('scale',[0.,1.])
def test_every_residual_telescopes_and_preserves_symmetry(scale):
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        tn,_=setup.fixture(4,4,2,4,scale)
        reference=scalar(tn)
        for cap in (1,4,16):
            forward=forward_residuals(tn,4,4,cap)
            assert len(forward['records'])==6
            for record in forward['records']:
                for t in record['before']:
                    assert type(t.data).__name__=='Z2Z2Array'
                    t.data.check()
            result=correct_residuals(tn,4,4,forward,None)
            np.testing.assert_allclose(result['corrected'],reference,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(math.fsum(r['correction'] for r in result['contributions']),
                                       reference-forward['approximate'],rtol=1e-9,atol=1e-12)
        # Snapshot construction and environment contraction do not mutate input.
        np.testing.assert_allclose(scalar(tn),reference,rtol=1e-13)


def test_compressed_environment_and_ratio_correction():
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        endpoints=[]
        for scale in (0.,1.):
            tn,log_scale=setup.fixture(4,4,2,4,scale)
            f=forward_residuals(tn,4,4,4)
            result=correct_residuals(tn,4,4,f,16)
            factor=np.exp(log_scale)
            endpoints.append({k:result[k]*factor for k in ('approximate','correction','corrected')})
            np.testing.assert_allclose(result['corrected'],scalar(tn),rtol=1e-10,atol=1e-12)
        a,b=endpoints
        ratio=b['approximate']/a['approximate']
        numerator=b['correction']-ratio*a['correction']
        np.testing.assert_allclose(b['corrected']/a['corrected']-ratio,
                                   numerator/a['corrected'],rtol=1e-11,atol=1e-13)


@pytest.mark.parametrize('environment_cap',[2,8,None])
def test_streamed_residuals_match_retained_snapshots(environment_cap):
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        tn,_=setup.fixture(4,4,2,4,1.)
        saved=correct_residuals(tn,4,4,forward_residuals(tn,4,4,4),environment_cap)
        streamed=streaming_residuals(tn,4,4,4,environment_cap)
        for key in ('approximate','corrected','correction'):
            np.testing.assert_allclose(streamed[key],saved[key],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose([r['correction'] for r in streamed['contributions']],
                                   [r['correction'] for r in saved['contributions']],rtol=1e-9,atol=1e-12)
        assert streamed['forward_snapshots_retained']==0


@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires explicit Metal access')
@pytest.mark.parametrize('environment_cap',[8,16])
def test_gpu_streamed_correction_matches_cpu(environment_cap):
    from peps_gpu_boundary import gpu_boundary,to_gpu
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        cpu_values=[];gpu_values=[]
        for scale in (0.,1.):
            tn,log_scale=setup.fixture(4,4,6,4,scale)
            cpu=streaming_residuals(tn,4,4,4,environment_cap)
            with gpu_boundary(32) as engine:
                gpu_tn=to_gpu(tn,engine.mx)
                def materialize(net):engine.mx.eval([b for t in net for b in t.data.blocks.values()])
                gpu=streaming_residuals(gpu_tn,4,4,4,environment_cap,materialize=materialize)
                assert engine.stats['svd_blocks']>0
                for tensor in gpu_tn:
                    assert type(tensor.data).__name__=='Z2Z2Array'
                    assert all(type(block).__module__.startswith('mlx') for block in tensor.data.blocks.values())
            for key in ('approximate','corrected','correction'):
                np.testing.assert_allclose(gpu[key],cpu[key],rtol=1e-4,atol=1e-9)
            cpu_values.append(cpu['corrected']*np.exp(log_scale))
            gpu_values.append(gpu['corrected']*np.exp(log_scale))
        np.testing.assert_allclose(gpu_values[1]/gpu_values[0],cpu_values[1]/cpu_values[0],rtol=1e-4)
