"""Production row telescoping, including cancellation and exact suffix limits."""
import math
import os
import numpy as np
import pytest
import peps_test_support as setup
from peps_boundary_residual import streaming_residuals, scalar
from tensor_zne import peps_contractions, evaluate_peps, curves


@pytest.mark.parametrize('shape', [(4,2),(4,4)])
@pytest.mark.parametrize('environment_cap', [2,None])
@pytest.mark.parametrize('cutoff', [0.,1e-4])
def test_row_and_bond_trajectory(shape, environment_cap, cutoff):
    nx,ny=shape
    with peps_contractions():
        tn,_=setup.fixture(nx,ny,3,4,0.)
        exact=scalar(tn)
        events=[]
        bond=streaming_residuals(tn,nx,ny,3,environment_cap,cutoff=cutoff)
        row=streaming_residuals(tn,nx,ny,3,environment_cap,cutoff=cutoff,
                                granularity='row',progress=events.append)
        for key in ('approximate','corrected','correction'):
            np.testing.assert_allclose(row[key],bond[key],rtol=1e-10,atol=1e-13)
        for record in row['contributions']:
            expected=math.fsum(r['correction'] for r in bond['contributions'] if r['row']==record['row'])
            assert record['correction']==pytest.approx(expected,abs=1e-13)
            assert 'bond' not in record
        assert row['absolute_correction_sum']<=bond['absolute_correction_sum']+1e-13
        assert row['overlap_count']==2*(ny-2)
        assert bond['overlap_count']==nx*(ny-2)
        assert len(events)==ny-1
        if environment_cap is None:
            assert row['corrected']==pytest.approx(exact,rel=1e-10,abs=1e-13)
        assert scalar(tn)==pytest.approx(exact,rel=1e-13)


def test_after_cnot_ratio_and_q_unchanged():
    kw=dict(nx=4,ny=4,steps=2,sites=(5,10),cap=4,boundary_cap=3,
            environment_cap=4,noise_placement='after_cnot',pair_rates=.001,
            gate_backend='batched-cpu')
    row=curves(evaluate_peps,[0.,1.],**kw)
    bond=curves(evaluate_peps,[0.,1.],residual_granularity='bond',**kw)
    for a,b in zip(row['rows'],bond['rows']):
        for key in a:
            if isinstance(a[key],(int,float)):
                assert a[key]==pytest.approx(b[key],rel=1e-9,abs=1e-13)
            else:assert a[key]==b[key]


@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires Metal')
def test_gpu_row_matches_gpu_bond():
    from peps_gpu_boundary import gpu_boundary,to_gpu
    with peps_contractions():
        tn,_=setup.fixture(4,4,3,4,0.)
        with gpu_boundary(32) as engine:
            gpu=to_gpu(tn,engine.mx)
            def materialize(net):engine.mx.eval([b for t in net for b in t.data.blocks.values()])
            bond=streaming_residuals(gpu,4,4,4,8,materialize=materialize)
            row=streaming_residuals(gpu,4,4,4,8,materialize=materialize,granularity='row')
            for key in ('approximate','corrected','correction'):
                assert row[key]==pytest.approx(bond[key],rel=2e-6,abs=1e-9)
