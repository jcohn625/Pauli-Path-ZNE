"""Accuracy tests for the active-subspace research factorization."""
import os
import numpy as np
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires Metal')


@pytest.mark.parametrize('deflation_tol',[0.,1e-7])
@pytest.mark.parametrize('n',[7,16,32,64,128])
def test_shared_jacobi_reconstruction_and_active_subspace(n,deflation_tol):
    from apple_jacobi_experiment import AdaptiveJacobi
    rng=np.random.default_rng(48)
    q,_=np.linalg.qr(rng.normal(size=(n,n)));v,_=np.linalg.qr(rng.normal(size=(n,n)))
    full=q@np.diag(np.geomspace(1.,.01,n))@v.T
    deficient=q[:,:3]@np.diag([1.,.1,.001])@v[:,:3].T
    x=np.stack([full,deficient,np.zeros((n,n))])
    engine=AdaptiveJacobi(sweeps=24,deflation_tol=deflation_tol);mx=engine.mx
    out=engine.jacobi_info(mx.array(x,dtype=mx.float32));mx.eval(out)
    u,s,vh,info=[np.asarray(a) for a in out]
    assert np.isfinite(u).all() and np.isfinite(s).all()
    for j in range(3):
        np.testing.assert_allclose((u[j]*s[j])@vh[j],x[j],rtol=1e-4,atol=3e-6)
        np.testing.assert_allclose(s[j],np.linalg.svd(x[j],compute_uv=False),rtol=3e-4,atol=3e-6)
        keep=s[j]>1e-4
        np.testing.assert_allclose(u[j][:,keep].T@u[j][:,keep],np.eye(sum(keep)),rtol=1e-4,atol=5e-5)
    # Zero columns are explicitly zero, never huge nonorthogonal vectors.
    assert np.max(np.abs(u))<1.001
    assert info[2,1]==1
