import json

import numpy as np
import pytest

from pauli_zne.workflow import (
    define_experiment,
    calculate_curves,
    fit_curves,
    save_results,
    analyze_results,
)


def test_staged_workflow_freezes_inputs_and_refits_without_mps(monkeypatch, tmp_path):
    experiment = define_experiment(N=4, L=2, O="IZZI", mps={"svd": "full"})
    calculation = calculate_curves(
        experiment, bond_dims=[16], show=False, progress=None
    )
    original_rates = calculation.experiment.single_rates.copy()
    experiment.single_rates[:] = 0.8
    experiment.config["circuit"]["L"] = 9
    np.testing.assert_array_equal(calculation.experiment.single_rates, original_rates)
    assert calculation.experiment.config["circuit"]["L"] == 2

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Fitting must not recalculate MPS")

    monkeypatch.setattr("pauli_zne.workflow.evaluate", unexpected_calculation)
    fits = fit_curves(calculation, shots={}, show=False, progress=None)
    assert abs(fits.noiseless["ratio_K16"]["absolute_error"]) < 1e-10
    assert abs(fits.noiseless["Q_K16"]["absolute_error"]) < 1e-10
    assert fits.shot_results == []
    noisy = fit_curves(
        calculation,
        trials=4,
        shots={"partial": [1000, None, None, None]},
        show=False,
        progress=None,
    )
    assert len(noisy.shot_results) == 4
    assert fits.shot_results == []  # independent result/config snapshot
    output = save_results(noisy, output_root=tmp_path)
    assert json.loads((output / "config.json").read_text())["circuit"]["L"] == 2
    assert (output / "curves.json").is_file()
    assert (output / "shot_noise.csv").is_file()


def test_centered_observable_and_observation_validation():
    experiment = define_experiment(N=4, L=1)
    assert experiment.config["circuit"]["O"] == "IZIZ"
    assert define_experiment(N=2).config["circuit"]["O"] == "ZZ"
    with pytest.raises(ValueError):
        define_experiment(N=3)
    calculation = calculate_curves(
        experiment, bond_dims=[16], show=False, progress=None
    )
    with pytest.raises(ValueError, match="one mean"):
        fit_curves(calculation, observations=[0.1], show=False)


def test_analysis_with_no_shots_and_zero_target(tmp_path):
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = define_experiment(N=2, L=0, O="ZI", mps={"svd": "full"})
    calculation = calculate_curves(experiment, bond_dims=[4], show=False, progress=None)
    fits = fit_curves(calculation, shots={}, show=False, progress=None)
    analysis = analyze_results(fits, save=True, output_root=tmp_path, show=False)
    assert len(analysis.figure.axes) == 6
    assert len(analysis.curve_table) == 5
    assert (analysis.output_dir / "analysis.png").is_file()
    plt.close(analysis.figure)
