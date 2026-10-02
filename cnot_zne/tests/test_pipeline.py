import json
import numpy as np
import pytest
from pauli_zne.config import load_config, schedules, shot_vectors
from pauli_zne.pipeline import run, refit
from pauli_zne.fitting import ExponentialFits, benchmark


@pytest.mark.parametrize(
    "update",
    [
        {"circuit": {"N": 3, "O": "ZZI"}},
        {"s_grid": [0, 1]},
        {"s_grid": [1, 1]},
        {"noise": {"single": [-0.1, 0, 0]}},
        {"noise": {"channels": {"typo": {}}}},
        {"fit": {"shot_scenarios": {"bad": [100, 100]}}},
        {"mps": {"bond_dims": [32, 16]}},
        {"noise": {"placement": "wrong"}},
        {"circuit": {"O": "bad"}},
    ],
)
def test_validation(update):
    with pytest.raises(ValueError):
        c, b = load_config(overrides=update)
        schedules(c, b)


def test_per_channel_rates_and_shots(tmp_path):
    np.savez(
        tmp_path / "rates.npz",
        single_rates=np.full((4, 3), 0.02),
        pair_rates=np.zeros((3, 3, 3)),
    )
    config = {
        "circuit": {"N": 4, "O": "IZZI"},
        "noise": {
            "channels": {
                "even_lr": {"single": [0.01, 0.02, 0.03]},
                "odd_rl": {"file": "rates.npz"},
            }
        },
        "fit": {
            "shot_scenarios": {"partial": {"1": 100, "2": 1000, "4": None, "8": None}}
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    c, b = load_config(path)
    s, p = schedules(c, b)
    np.testing.assert_allclose(s[:, 0, 0], [[0.01, 0.02, 0.03]] * 2)
    assert np.all(s[:, 3] == 0.02)
    assert shot_vectors(c)["partial"] == [100, 1000, None, None]


def test_exponential_recovery_and_insufficient_points():
    c, _ = load_config()
    s = np.array(c["s_grid"])
    f = ExponentialFits(s, c["fit"])
    one = f.single(0.16 * np.exp(-0.2 * s))
    assert one["estimate"] == pytest.approx(0.16, rel=1e-6)
    two = f.dual(0.1 * np.exp(-0.1 * s) + 0.06 * np.exp(-0.6 * s))
    assert two["estimate"] == pytest.approx(0.16, rel=1e-6)
    np.testing.assert_allclose(
        np.exp(-s[:, None] * two["rates"]) @ two["amplitudes"],
        0.1 * np.exp(-0.1 * s) + 0.06 * np.exp(-0.6 * s),
        atol=1e-9,
    )
    assert (
        ExponentialFits([1, 2], c["fit"]).dual(np.array([0.1, 0.05]))["status"]
        == "insufficient_points"
    )


def test_end_to_end_and_refit(tmp_path):
    overrides = {
        "circuit": {"N": 4, "L": 2, "O": "IZZI"},
        "mps": {"bond_dims": [4, 16], "svd": "full"},
        "fit": {"trials": 8, "shot_scenarios": {"partial": [10000, 10000, None, None]}},
    }
    out = run(None, tmp_path / "run", overrides, progress=lambda s: None)
    ref = json.loads((out / "reference.json").read_text())
    assert ref["converged"] and ref["curve_method"] == "density_matrix"
    curves = json.loads((out / "curves.json").read_text())
    fits = json.loads((out / "fits.json").read_text())
    assert abs(fits["ratio_K16"]["absolute_error"]) < 1e-10
    assert curves["16"]["ratio"][0] == pytest.approx(1)
    samples = json.loads((out / "exponential_trials.json").read_text())
    for sample in samples:
        np.testing.assert_array_equal(sample["observations"][2:], ref["values"][3:])
    refit(out, tmp_path / "refit", progress=lambda s: None)
    assert json.loads((tmp_path / "refit" / "fits.json").read_text()) == fits
    with pytest.raises(ValueError):
        run(None, out, overrides, progress=lambda s: None)


def test_mps_reference_budget_is_not_called_exact(tmp_path):
    overrides = {
        "circuit": {"N": 4, "L": 2, "O": "IZZI"},
        "mps": {"bond_dims": [4], "svd": "full"},
        "reference": {
            "statevector_max_n": 0,
            "density_max_n": 0,
            "bond_dims": [1, 2],
            "rtol": 0.0,
            "atol": 1e-15,
        },
        "fit": {"shot_scenarios": {}},
    }
    out = run(None, tmp_path / "run", overrides, progress=lambda s: None)
    ref = json.loads((out / "reference.json").read_text())
    assert (
        not ref["converged"]
        and ref["status"] == "budget_exhausted_not_converged"
        and ref["O0_method"] == "mps"
    )


def test_identity_does_not_crash_q_fit(tmp_path):
    out = run(
        None,
        tmp_path / "identity",
        {
            "circuit": {"N": 2, "L": 1, "O": "II"},
            "mps": {"bond_dims": [4], "svd": "full"},
            "fit": {"shot_scenarios": {}},
        },
        progress=lambda s: None,
    )
    fits = json.loads((out / "fits.json").read_text())
    assert fits["Q_K4"]["status"] == "undefined_or_degenerate_modes"
    assert fits["ratio_K4"]["estimate"] == pytest.approx(1, abs=1e-12)


def test_randomized_reference_convergence_is_checked(tmp_path):
    out = run(
        None,
        tmp_path / "reference",
        {
            "circuit": {"N": 4, "L": 2, "O": "IZZI"},
            "mps": {"bond_dims": [16]},
            "reference": {"density_max_n": 0, "bond_dims": [16, 32, 64]},
            "fit": {"shot_scenarios": {}},
        },
        progress=lambda s: None,
    )
    ref = json.loads((out / "reference.json").read_text())
    assert ref["curve_method"] == "mps"
    assert ref["O0_method"] == "statevector"
    assert ref["converged"] and ref["reproducibility"]["passed"]
    assert len(ref["checks"]) == 2


def test_near_zero_observable_does_not_require_a_ratio(tmp_path):
    out = run(
        None,
        tmp_path / "zero",
        {
            "circuit": {"N": 4, "L": 0, "O": "ZIII"},
            "mps": {"bond_dims": [4], "svd": "full"},
            "fit": {"shot_scenarios": {}},
        },
        progress=lambda s: None,
    )
    ref = json.loads((out / "reference.json").read_text())
    fits = json.loads((out / "fits.json").read_text())
    assert abs(ref["O0"]) < 1e-12
    assert fits["ratio_K4"]["status"] == "undefined_or_degenerate_modes"
    assert fits["single_exp"]["relative_error_pct"] is None


def test_linear_binomial_noise_matches_analytic_variance():
    c, _ = load_config()
    c["fit"].update(trials=4000, seed=8)
    curves = {4: {"ratio": np.array([1.0, 0.8]), "Q_avg": None, "Q_contrast": None}}
    baseline, rows, _ = benchmark(
        [1.0], [0.16], 0.2, curves, c["fit"], {"shots": [1000]}
    )
    result = next(r for r in rows if r["method"] == "ratio_K4")
    expected = np.sqrt((1 - 0.16**2) / 1000) / 0.8
    assert abs(result["rmse"] / expected - 1) < 0.06


def test_example_configs_are_valid():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ["small.json", "n32.json"]:
        c, base = load_config(root / "examples" / name)
        singles, pairs = schedules(c, base)
        assert singles.shape == (c["circuit"]["L"], 4, c["circuit"]["N"], 3)
        assert pairs.shape == (c["circuit"]["L"], 4, c["circuit"]["N"] - 1, 3, 3)
