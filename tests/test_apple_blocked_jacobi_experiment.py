"""Validate reconstruction separately from singular-vector quality."""
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires Metal')


@pytest.mark.parametrize('n',[64,128])
def test_blocked_cleanup_is_an_svd_for_well_conditioned_input(n):
    from apple_blocked_jacobi_experiment import BlockedGramJacobi
    rng=np.random.default_rng(57)
    q,_=np.linalg.qr(rng.normal(size=(n,n)));v,_=np.linalg.qr(rng.normal(size=(n,n)))
    a=q@np.diag(np.geomspace(1.,.01,n))@v.T
    engine=BlockedGramJacobi(block=16,sweeps=2,polish=True);mx=engine.mx
    out=engine.jacobi(mx.array(a[None],dtype=mx.float32));mx.eval(out)
    u,s,vh=[np.array(z[0]).astype(float) for z in out]
    assert np.linalg.norm((u*s)@vh-a)/np.linalg.norm(a)<1e-5
    assert np.linalg.norm(u.T@u-np.eye(n))<2e-4
    assert np.linalg.norm(vh@vh.T-np.eye(n))<2e-4
    np.testing.assert_allclose(s,np.geomspace(1.,.01,n),rtol=2e-5,atol=1e-7)


def test_blocked_zero_and_rank_deficient_reconstruction():
    from apple_blocked_jacobi_experiment import BlockedGramJacobi
    rng=np.random.default_rng(71)
    a=np.stack([np.zeros((64,64)),rng.normal(size=(64,3))@rng.normal(size=(3,64))])
    engine=BlockedGramJacobi(block=16,sweeps=2);mx=engine.mx
    out=engine.jacobi(mx.array(a,dtype=mx.float32));mx.eval(out)
    u,s,vh=[np.array(z).astype(float) for z in out]
    assert all(np.isfinite(z).all() for z in (u,s,vh))
    assert np.all(s[0]==0)
    assert np.linalg.norm((u[1]*s[1])@vh[1]-a[1])/np.linalg.norm(a[1])<1e-5
    # Deliberately no full-U isometry assertion: this research path does not
    # complete a nullspace or promise accurate weak singular vectors.
