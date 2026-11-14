"""Independent einsum references for rectangular, strided MPS contractions."""
import numpy as np
import pytest
from mps_contractions import (absorb_left, absorb_right, join_sites,
                              environment_left, environment_right)


@pytest.mark.parametrize('dims',[(1,4,1,1),(3,7,11,2),(7,3,2,11),(8,5,6,9)])
@pytest.mark.parametrize('complex_values',[False,True])
def test_bilinear_environments(dims,complex_values):
    rng=np.random.default_rng(412)
    def sample(shape):
        x=rng.normal(size=(*shape[:-1],shape[-1]*2))
        if complex_values:x=x+1j*rng.normal(size=x.shape)
        return x[...,::2]
    a,b,r,s=dims
    x,y=sample((a,4,r)),sample((b,4,s))
    left,right=sample((a,b)),sample((r,s))
    np.testing.assert_allclose(environment_left(left,x,y),
        np.einsum('ab,apr,bps->rs',left,x,y,optimize=False),rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(environment_right(right,x,y),
        np.einsum('apr,bps,rs->ab',x,y,right,optimize=False),rtol=2e-12,atol=2e-12)


def test_bond_absorption_and_join_with_strided_inputs():
    rng=np.random.default_rng(53)
    a=rng.normal(size=(5,4,12))[...,::2]
    b=rng.normal(size=(6,4,14))[...,::2]
    link_left=rng.normal(size=(3,5));link_right=rng.normal(size=(6,9))
    for actual,expected in (
        (absorb_left(link_left,a),np.einsum('ab,bpr->apr',link_left,a)),
        (absorb_right(a,link_right),np.einsum('lpa,ab->lpb',a,link_right)),
        (join_sites(a,b),np.einsum('lam,mbr->labr',a,b))):
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
