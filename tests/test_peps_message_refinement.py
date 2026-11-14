import numpy as np
import pytest
import peps_test_support as setup
from tensor_zne import peps_contractions
from peps_boundary_residual import scalar,streaming_residuals,backward_environments
from peps_message_refinement import refined_messages,refined_residuals


@pytest.mark.parametrize('shape',[(4,2),(4,4)])
def test_bidirectional_exact_limit_and_input_unchanged(shape):
    nx,ny=shape
    with peps_contractions():
        tn,_=setup.fixture(nx,ny,2,4,1.)
        exact=scalar(tn)
        result=refined_messages(tn,nx,ny,64,sweeps=2,check=True)
        assert len(result['history'])==3
        for record in result['history']:
            np.testing.assert_allclose(record['cut_values'],exact,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(scalar(tn),exact,rtol=1e-13)


def test_refined_exact_guides_correct_truncated_forward():
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,0.)
        result=refined_residuals(tn,4,4,2,64,sweeps=2,check=True)
        np.testing.assert_allclose(result['corrected'],scalar(tn),rtol=1e-9,atol=1e-12)
        assert result['guide_source']=='provided'


def test_provided_guides_match_ordinary_and_are_not_consumed():
    with peps_contractions():
        tn,_=setup.fixture(4,4,3,4,0.)
        guides=backward_environments(tn,4,4,4)
        spectra=[t.data.to_dense().copy() for env in guides.values() for t in env]
        result=streaming_residuals(tn,4,4,2,4,granularity='row',backward_guides=guides)
        reference=streaming_residuals(tn,4,4,2,4,granularity='row')
        np.testing.assert_allclose(result['corrected'],reference['corrected'],rtol=1e-13)
        assert set(guides)=={1,2}
        for a,t in zip(spectra,[t for env in guides.values() for t in env]):
            np.testing.assert_array_equal(a,t.data.to_dense())
        with pytest.raises(ValueError,match='every residual row'):
            streaming_residuals(tn,4,4,2,4,backward_guides={1:guides[1]})


def test_refinement_rebuilds_original_rows_each_iteration(monkeypatch):
    import peps_message_refinement as module
    with peps_contractions():
        tn,_=setup.fixture(4,4,2,4,0.)
        original=module._updated_message;calls=[]
        def spy(source,incoming,guide,y,previous,*args):
            assert source is tn
            calls.append((y,previous))
            return original(source,incoming,guide,y,previous,*args)
        monkeypatch.setattr(module,'_updated_message',spy)
        result=refined_messages(tn,4,4,4,sweeps=3,check=True)
        assert calls==[(1,0),(2,1),(2,3),(1,2)]*3
        assert all(np.isfinite(h['cut_values']).all() for h in result['history'])
