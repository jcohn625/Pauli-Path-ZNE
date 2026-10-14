import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires Metal')


@pytest.mark.parametrize('shape',[(180,144),(144,180)])
def test_hybrid_complete_factors_rank_deficiency_and_zero(shape):
    from peps_hybrid_decomp import HybridDecompositions
    engine=HybridDecompositions(cpu_qr_min=1,cpu_svd_min=1);mx=engine.mx
    m,n=shape;k=min(shape);rng=np.random.default_rng(94)
    a=np.stack([rng.normal(size=(m,3))@rng.normal(size=(3,n)),np.zeros(shape)]).astype('float32')
    dev=mx.array(a);q,r=engine.qr(dev);u,s,vh=engine.svd(dev);mx.eval(q,r,u,s,vh)
    q,r,u,s,vh=[np.asarray(x).astype(float) for x in (q,r,u,s,vh)]
    for left,right in ((q,r),(u*s[:,None,:],vh)):
        reconstructed=left@right
        assert np.linalg.norm(reconstructed[0]-a[0])/np.linalg.norm(a[0])<3e-6
        np.testing.assert_array_equal(reconstructed[1],a[1])
    for iso in (q,u):
        np.testing.assert_allclose(iso.transpose(0,2,1)@iso,np.broadcast_to(np.eye(k),(2,k,k)),atol=3e-6)
    np.testing.assert_allclose(vh@vh.transpose(0,2,1),np.broadcast_to(np.eye(k),(2,k,k)),atol=3e-6)
    np.testing.assert_array_equal(np.array(dev),a)
    assert engine.stats['cpu_svd_blocks']==2
    assert engine.stats['cpu_qr_blocks']==2


def test_hybrid_global_rank_selection_and_restore():
    import quimb.tensor as qtn
    from peps_gpu_boundary import gpu_boundary
    from pauli_peps_symmetry import symmray
    import mlx.core as mx
    from symmray.sparse.sparse_array_common import SparseArrayCommon
    original=SparseArrayCommon._split_abelian
    array=symmray().Z2Z2Array.from_blocks({
        ((0,0),(0,0)):mx.diag(mx.array([10.,1.])),
        ((1,1),(1,1)):mx.diag(mx.array([9.,.5])),
        ((0,1),(0,1)):mx.diag(mx.array([8.,.25])),
    },duals=(False,True),charge=(0,0))
    with gpu_boundary(16,decomposition_backend='hybrid',cpu_qr_min=1,cpu_svd_min=1) as engine:
        u,s,v=qtn.Tensor(array,inds=['a','b']).split(left_inds=['a'],get='tensors',max_bond=2,cutoff=0.,absorb=None)
        actual=sorted(float(x) for b in s.data.blocks.values() for x in np.array(b))
        np.testing.assert_allclose(actual,[9.,10.],rtol=2e-6)
        assert engine.stats['cpu_svd_blocks']==3
        u.data.check();v.data.check()
    assert SparseArrayCommon._split_abelian is original


@pytest.mark.parametrize('panel',[16,64,128])
def test_gpu_qr_panel_choices_preserve_gauge(panel):
    from peps_gpu_decomp import GPUDecompositions
    engine=GPUDecompositions(qr_panel_width=panel);mx=engine.mx
    a=np.random.default_rng(13).normal(size=(1,200,144)).astype('float32')
    a[:,:,1]=a[:,:,0]
    q,r=engine.qr(mx.array(a));mx.eval(q,r)
    q,r=np.asarray(q).astype(float),np.asarray(r).astype(float)
    assert np.linalg.norm(q@r-a)/np.linalg.norm(a)<5e-6
    np.testing.assert_allclose(q.transpose(0,2,1)@q,np.eye(144)[None],atol=4e-6)


def test_after_cnot_hybrid_ratio_and_q():
    from tensor_zne import evaluate_peps,curves
    kw=dict(nx=4,ny=4,steps=2,sites=(5,10),cap=8,boundary_cap=16,
            environment_cap=16,noise_placement='after_cnot',pair_rates=.001,
            gate_backend='batched-cpu')
    cpu=curves(evaluate_peps,[0.,1.],**kw)
    gpu=curves(evaluate_peps,[0.,1.],boundary_backend='metal-hybrid',**kw)
    for expected,actual in zip(cpu['rows'],gpu['rows']):
        for key in ('observable_corrected','unsigned_corrected','ratio_corrected',
                    'q_plus_corrected','q_minus_corrected'):
            if expected[key] is None:assert actual[key] is None
            else:assert actual[key]==pytest.approx(expected[key],rel=2e-4,abs=1e-9)
    assert any(v['signed']['details']['boundary_gpu_stats'].get('cpu_qr_blocks',0)>0 for v in gpu['raw'])


@pytest.mark.parametrize('residual',[False,True])
def test_hybrid_nonresidual_and_residual_public_api(residual):
    from tensor_zne import evaluate_peps
    kw=dict(nx=2,ny=3,steps=1,cap=4,boundary_cap=4,environment_cap=8,
            noise_placement='after_cnot',pair_rates=.001,residual=residual)
    cpu=evaluate_peps(**kw)
    gpu=evaluate_peps(**kw,boundary_backend='metal-hybrid')
    assert gpu['corrected']==pytest.approx(cpu['corrected'],rel=2e-4,abs=1e-7)


@pytest.mark.parametrize('shape',[(160,128),(200,144)])
def test_metal_qr_tiny_columns_keep_complete_orthogonal_gauge(shape):
    from peps_gpu_decomp import GPUDecompositions
    engine=GPUDecompositions();mx=engine.mx;m,n=shape
    rng=np.random.default_rng(27)
    a=(rng.normal(size=(2,m,n))*np.geomspace(1.,1e-30,n)[None,None,:]).astype('float32')
    a[1,:,:]=0.
    q,r=engine.qr(mx.array(a));mx.eval(q,r)
    q,r=np.asarray(q).astype(float),np.asarray(r).astype(float)
    assert np.isfinite(q).all() and np.isfinite(r).all()
    assert np.linalg.norm(q[0]@r[0]-a[0])/np.linalg.norm(a[0])<3e-6
    np.testing.assert_allclose(q.transpose(0,2,1)@q,np.broadcast_to(np.eye(n),(2,n,n)),atol=4e-6)
    np.testing.assert_array_equal(q[1]@r[1],a[1])
