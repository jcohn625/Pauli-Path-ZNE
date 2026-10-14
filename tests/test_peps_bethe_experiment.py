"""Independent exact limits and message-gauge checks for research estimators."""
import numpy as np
import pytest
import peps_test_support as setup
import peps_bethe_experiment as experiment
from peps_boundary_residual import scalar, streaming_residuals


@pytest.mark.parametrize('shape',[(4,2),(4,4)])
def test_bethe_exact_limit_and_input_unchanged(shape):
    nx,ny=shape
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        tn,_=setup.fixture(nx,ny,2,4,0.)
        reference=scalar(tn)
        result=experiment.bethe_contract(tn,nx,ny,64)
        assert result['status']=='ok'
        np.testing.assert_allclose(result['value'],reference,rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(scalar(tn),reference,rtol=1e-13)
        for tensor in tn:tensor.data.check()


def test_message_rescaling_cancels_in_bethe_ratio(monkeypatch):
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        tn,_=setup.fixture(4,4,2,4,1.)
        reference=experiment.bethe_contract(tn,4,4,4)['value']
        original=experiment.boundary_messages
        def rescaled(*args):
            top,bottom=original(*args)
            for i,net in enumerate(list(top.values())+list(bottom.values())):
                tensor=next(iter(net));tensor.modify(data=tensor.data*((-2.)**(i+1)))
            return top,bottom
        monkeypatch.setattr(experiment,'boundary_messages',rescaled)
        result=experiment.bethe_contract(tn,4,4,4)
        np.testing.assert_allclose(result['value'],reference,rtol=2e-11,atol=1e-13)


@pytest.mark.parametrize('environment_cap',[2,8,None])
def test_row_residual_equals_sum_of_bond_residuals(environment_cap):
    with setup.default_implementation((setup._safe_einsum,setup._safe_tensordot)),setup.boundary_backend('blas'):
        tn,_=setup.fixture(4,4,3,4,0.)
        reference=streaming_residuals(tn,4,4,4,environment_cap)
        result=experiment.row_residual_contract(tn,4,4,4,environment_cap)
        for key in ('approximate','corrected','correction'):
            np.testing.assert_allclose(result[key],reference[key],rtol=1e-10,atol=1e-13)
        assert result['overlap_count']==4
        if environment_cap is None:
            np.testing.assert_allclose(result['corrected'],scalar(tn),rtol=1e-10,atol=1e-13)
