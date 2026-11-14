import numpy as np
import pytest
import peps_test_support as setup
from tensor_zne import peps_contractions
from peps_boundary_residual import scalar
from peps_environment_experiment import guided_residuals,compress_guided_bond


@pytest.mark.parametrize('cap',[1,4,16])
@pytest.mark.parametrize('scale',[0.,1.])
def test_guided_exact_suffix_telescopes_and_preserves_input(cap,scale):
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,scale)
        exact=scalar(tn)
        result=guided_residuals(tn,4,4,cap,None,check=True)
        np.testing.assert_allclose(result['corrected'],exact,rtol=1e-9,atol=1e-13)
        np.testing.assert_allclose(scalar(tn),exact,rtol=1e-13)


def test_untruncated_guided_sweep_exact():
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,0.)
        result=guided_residuals(tn,4,4,1024,4,check=True)
        assert result['correction']==0.
        np.testing.assert_allclose(result['approximate'],scalar(tn),rtol=1e-10,atol=1e-13)


def test_projector_matches_dense_nonhermitian_reference():
    # Use a real evolved row to exercise sector/global-rank bookkeeping.
    from peps_boundary_residual import backward_environments
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,1.)
        env=backward_environments(tn,4,4,4)[1]
        for x in range(4):tn.contract_((tn.site_tag(0,x),tn.site_tag(1,x)),which='any')
        tn.canonize_row(1,sweep='left',equalize_norms=False)
        opened=tn.select(tn.x_tag(1)) & env
        l=tn.site_tag(1,0);r=tn.site_tag(1,1);bond,=opened[l].bonds(opened[r])
        opened[l].reindex_({bond:'lc'});opened[r].reindex_({bond:'rc'})
        matrix=opened.contract(all,optimize='greedy',output_inds=['lc','rc']).data.to_dense()
        u,_,_=np.linalg.svd(matrix,full_matrices=False)
        expected=np.trace(u[:,:3].T @ matrix @ u[:,:3])
        compress_guided_bond(tn,env,1,0,3)
        actual=scalar(tn.select(tn.x_tag(1)) & env)
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-13)
        assert tn[l].ind_size(bond)==3
        for tensor in tn:tensor.data.check()


@pytest.mark.parametrize('cap',[1,4,16])
def test_rtm_exact_mps_guide_and_symmetry(cap):
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,0.)
        exact=scalar(tn)
        result=guided_residuals(tn,4,4,cap,64,method='rtm',check=True)
        np.testing.assert_allclose(result['corrected'],exact,rtol=1e-9,atol=1e-13)
        np.testing.assert_allclose(scalar(tn),exact,rtol=1e-13)


def test_rtm_last_site_matches_dense_weighted_right_subspace():
    from peps_boundary_residual import backward_environments
    from peps_environment_experiment import compress_rtm_row
    with peps_contractions():
        work,_=setup.fixture(4,4,2,4,1.)
        env=backward_environments(work,4,4,8)[1]
        for x in range(4):work.contract_((work.site_tag(0,x),work.site_tag(1,x)),which='any')
        work.canonize_row(1,sweep='right',equalize_norms=False)
        guide=env.copy();guide.canonize_row(2,sweep='right',equalize_norms=False)
        current=work[work.site_tag(1,3)]
        bond,=current.bonds(work[work.site_tag(1,2)])
        physical,=[i for i in current.inds if i!=bond]
        prefix=work.select([work.site_tag(1,x) for x in range(3)],which='any') & guide.select([guide.y_tag(x) for x in range(3)],which='any')
        weighted=(prefix & current).contract(all,optimize='greedy')
        left,=[i for i in weighted.inds if i!=physical]
        dense=weighted.transpose(left,physical).data.to_dense()
        _,_,vh=np.linalg.svd(dense,full_matrices=False)
        expected=vh[:3].T@vh[:3]
        compress_rtm_row(work,env,1,4,3)
        final=work[work.site_tag(1,3)]
        retained,=[i for i in final.inds if i!=physical]
        v=final.transpose(retained,physical).data.to_dense()
        np.testing.assert_allclose(v.T@v,expected,atol=1e-10)
