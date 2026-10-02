"""Notebook-friendly workflow functions built on the tested simulator and fits.

The four high-level calls are define_experiment, calculate_curves, fit_curves,
and analyze_results. Returned dataclasses retain all arrays and diagnostics.
Pandas/Matplotlib/IPython are only needed for display and analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import copy
import warnings
import numpy as np

from .config import load_config, schedules, shot_vectors
from .mps import evaluate, modes
from .pipeline import reference, save as write_json, csv_save
from .fitting import benchmark, designs


@dataclass
class Experiment:
    config: dict
    single_rates: np.ndarray
    pair_rates: np.ndarray


@dataclass
class CurveCalculation:
    experiment: Experiment
    curves: dict
    reference: dict
    curve_rows: list
    contractions: list
    warnings: list


@dataclass
class FitResults:
    calculation: CurveCalculation
    config: dict
    observations: np.ndarray
    target: float
    noiseless: dict
    shot_results: list
    exponential_trials: list
    external_observations: bool = False


@dataclass
class AnalysisResults:
    figure: object
    fit_table: object
    shot_table: object
    curve_table: object
    error_table: object
    output_dir: Path | None


def _display_rows(rows):
    import pandas as pd
    from IPython.display import display

    display(pd.DataFrame(rows))


def define_experiment(
    N=6,
    L=2,
    phi=np.pi / 16,
    O=None,
    noise=None,
    mps=None,
    reference_options=None,
    rates_directory=None,
):
    """Define a circuit and validated noise arrays without running simulation.

    O is an N-character Pauli string; its default is a centered ZZ observable.
    Noise uses the same dictionary schema as the JSON config, including four
    named channel overrides and optional NPZ files. Other options are partial
    overrides of the standard mps/reference settings.
    """
    if O is None:
        if not isinstance(N, int) or isinstance(N, bool) or N < 2:
            raise ValueError("N must be an even integer >= 2")
        chars = ["I"] * N
        for q in ((0, 1) if N == 2 else (N // 2 - 1, N // 2 + 1)):
            chars[q] = "Z"
        O = "".join(chars)
    overrides = {"circuit": {"N": N, "L": L, "phi": phi, "O": O}}
    if noise is not None:
        overrides["noise"] = noise
    if mps is not None:
        overrides["mps"] = mps
    if reference_options is not None:
        overrides["reference"] = reference_options
    config, _ = load_config(overrides=overrides)
    single, pair = schedules(config, Path(rates_directory or ".").resolve())
    return Experiment(config, single, pair)


def calculate_curves(
    experiment,
    s_grid=None,
    bond_dims=None,
    reference_bonds=None,
    show=True,
    progress=print,
):
    """Compute O, unsigned path sums, R and Q modes, plus a reference curve.

    Inputs are copied so editing an Experiment afterward cannot silently change
    the provenance of existing calculations. A fresh cache is used per call.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("Pass an Experiment returned by define_experiment")
    config = copy.deepcopy(experiment.config)
    if s_grid is not None:
        config["s_grid"] = list(s_grid)
    if bond_dims is not None:
        config["mps"]["bond_dims"] = list(bond_dims)
    if reference_bonds is not None:
        config["reference"]["bond_dims"] = list(reference_bonds)
    config, _ = load_config(overrides=config)
    frozen = Experiment(
        config, experiment.single_rates.copy(), experiment.pair_rates.copy()
    )
    cache, contractions, warning_records, curves, rows = {}, [], [], {}, []
    scales = np.array([0.0] + config["s_grid"])
    emit = progress if progress is not None else lambda _: None

    def calculate(cap, scale, unsigned):
        key = (int(cap), float(scale), bool(unsigned))
        if key not in cache:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = evaluate(
                    config,
                    frozen.single_rates,
                    frozen.pair_rates,
                    float(scale),
                    int(cap),
                    unsigned=unsigned,
                )
            warning_records.extend(
                (w.category.__name__, str(w.message)) for w in caught
            )
            cache[key] = result
            contractions.append(
                {
                    "K": cap,
                    "s": scale,
                    "channel": "absolute" if unsigned else "signed",
                    **result,
                }
            )
        return cache[key]

    for cap in config["mps"]["bond_dims"]:
        signed = [calculate(cap, s, False)["value"] for s in scales]
        log_unsigned = [calculate(cap, s, True)["log_abs"] for s in scales]
        curve = modes(signed, log_unsigned)
        curves[cap] = {**curve, "O": np.array(signed), "log_absolute": log_unsigned}
        for j, scale in enumerate(scales):
            row = {
                "K": cap,
                "s": scale,
                "O_K(s)": signed[j],
                "log A_K(s)": log_unsigned[j],
                "mode_status": curve["status"],
            }
            for name in ("ratio", "Q_plus", "Q_minus", "Q_avg", "Q_diff", "Q_contrast"):
                row[name] = None if curve[name] is None else float(curve[name][j])
            rows.append(row)
        emit(f"Completed signed/unsigned MPS curves at K={cap}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ref = reference(config, frozen.single_rates, frozen.pair_rates, calculate, emit)
    warning_records.extend((w.category.__name__, str(w.message)) for w in caught)
    warning_summary = [
        {"category": category, "message": message, "count": count}
        for (category, message), count in Counter(warning_records).items()
    ]
    emit(
        f"O(0)={ref['O0']:.12g} ({ref['O0_method']}); noisy reference={ref['curve_method']}; converged={ref['converged']}"
    )
    if not ref["converged"]:
        emit("REFERENCE NOT CONVERGED: subsequent errors are provisional.")
    if show:
        if warning_summary:
            emit(
                "Captured numerical warnings: inspect alongside reference/convergence checks."
            )
            _display_rows(warning_summary)
        _display_rows(
            [{"s": s, "O_reference(s)": v} for s, v in zip(scales, ref["values"])]
        )
    return CurveCalculation(frozen, curves, ref, rows, contractions, warning_summary)


def fit_curves(
    calculation,
    shots=None,
    trials=None,
    seed=None,
    rate_bounds=None,
    amplitude_bound=None,
    observations=None,
    show=True,
    progress=print,
):
    """Fit all four methods without performing any tensor contractions.

    shots is a scenario-name -> count/vector/per-s-map dictionary. None keeps
    configured scenarios; {} disables shot simulation. A count of None inside
    a scenario leaves that point exact. observations optionally supplies means
    in sorted s order, e.g. hardware data; comparisons still use the simulation
    reference as target, and synthetic noise uses those supplied plug-in means.
    """
    if not isinstance(calculation, CurveCalculation):
        raise TypeError("Pass a CurveCalculation returned by calculate_curves")
    config = copy.deepcopy(calculation.experiment.config)
    for name, value in [
        ("shot_scenarios", shots),
        ("trials", trials),
        ("seed", seed),
        ("rate_bounds", rate_bounds),
        ("amplitude_bound", amplitude_bound),
    ]:
        if value is not None:
            config["fit"][name] = value
    config, _ = load_config(overrides=config)
    y = np.array(
        calculation.reference["values"][1:] if observations is None else observations,
        dtype=float,
    )
    if y.shape != (len(config["s_grid"]),):
        raise ValueError("observations must contain one mean for every positive scale")
    target = float(calculation.reference["O0"])
    baseline, noisy, samples = benchmark(
        config["s_grid"],
        y,
        target,
        calculation.curves,
        config["fit"],
        shot_vectors(config),
        progress=progress if progress is not None else lambda _: None,
    )
    result = FitResults(
        calculation,
        config,
        y.copy(),
        target,
        baseline,
        noisy,
        samples,
        external_observations=observations is not None,
    )
    if show:
        _display_rows(
            [
                {
                    "method": name,
                    "status": f["status"],
                    "estimate": f.get("estimate"),
                    "relative_error_pct": f.get("relative_error_pct"),
                    "sse": f.get("sse"),
                }
                for name, f in baseline.items()
            ]
        )
    return result


def save_results(fits, figure=None, output_root="runs"):
    """Save a unique result folder; does not overwrite previous exports."""
    calculation = fits.calculation
    output_dir = Path(output_root).resolve() / datetime.now(timezone.utc).strftime(
        "notebook_%Y%m%dT%H%M%S_%fZ"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", fits.config)
    np.savez_compressed(
        output_dir / "inputs.npz",
        single_rates=calculation.experiment.single_rates,
        pair_rates=calculation.experiment.pair_rates,
    )
    write_json(output_dir / "reference.json", calculation.reference)
    write_json(output_dir / "numerical_warnings.json", calculation.warnings)
    write_json(output_dir / "curves.json", calculation.curves)
    write_json(output_dir / "fits.json", fits.noiseless)
    write_json(output_dir / "exponential_trials.json", fits.exponential_trials)
    write_json(
        output_dir / "fit_observations.json",
        {
            "s": fits.config["s_grid"],
            "O": fits.observations,
            "target_O0": fits.target,
            "external_observations": fits.external_observations,
        },
    )
    csv_save(output_dir / "mps_contractions.csv", calculation.contractions)
    csv_save(output_dir / "curve_values.csv", calculation.curve_rows)
    csv_save(output_dir / "shot_noise.csv", fits.shot_results)
    if figure is not None:
        figure.savefig(output_dir / "analysis.png", dpi=170, bbox_inches="tight")
    return output_dir


def analyze_results(fits, save=True, output_root="runs", show=True):
    """Display diagnostics and six plots; return figure/tables and export path.

    Set show=False for scripts and save=False to avoid filesystem writes.
    Pandas and Matplotlib are required only when this function is called.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    if show:
        from IPython.display import display
    else:
        display = lambda _: None
    emit = print if show else lambda *args, **kwargs: None
    calculation = fits.calculation
    calculation_config = calculation.experiment.config
    curves, ref = calculation.curves, calculation.reference
    fit_config, target_O0 = fits.config, fits.target
    fit_scales = np.array(fit_config["s_grid"])
    scales = np.r_[0.0, fit_scales]
    observed_means = fits.observations
    noiseless_fits, shot_results = fits.noiseless, fits.shot_results
    fit_table = pd.DataFrame(
        [{"method": name, **f} for name, f in noiseless_fits.items()]
    )
    shot_table = pd.DataFrame(shot_results)
    curve_table = pd.DataFrame(calculation.curve_rows)
    if fits.external_observations:
        emit(
            "External observations: errors compare with a simulation target, not known experimental truth."
        )
    highest_K = max(curves)
    reference_values = np.array(ref["values"])
    relative_ok = abs(target_O0) > 1e-12
    error_factor = 100 / abs(target_O0) if relative_ok else 1.0
    error_label = "Error / |reference O(0)| (%)" if relative_ok else "Absolute error"

    # Diagnostic tables: reference changes, fit stability, and per-point mode error.
    if ref.get("checks"):
        convergence_rows = []
        for check in ref["checks"]:
            for j, scale in enumerate(scales):
                convergence_rows.append(
                    {
                        "K": check["bond"],
                        "s": scale,
                        "absolute_change": check["absolute_changes"][j],
                        "tolerance": check["tolerances"][j],
                        "whole_grid_passed": check["passed"],
                    }
                )
        display(pd.DataFrame(convergence_rows))
    if ref.get("reproducibility"):
        emit("Reference repeat passed:", ref["reproducibility"]["passed"])

    # Values normalized to |O_ref(0)| avoid misleading relative errors at O(s)≈0.
    error_rows = []
    reference_ratio = reference_values / target_O0 if relative_ok else None
    for cap, curve in curves.items():
        for j, scale in enumerate(scales):
            error_rows.append(
                {
                    "K": cap,
                    "s": scale,
                    "O error / |O_ref(0)| (%)" if relative_ok else "O absolute error": (
                        curve["O"][j] - reference_values[j]
                    )
                    * error_factor,
                    "R_K minus R_reference": (
                        None
                        if curve["ratio"] is None or reference_ratio is None
                        else curve["ratio"][j] - reference_ratio[j]
                    ),
                }
            )
    display(pd.DataFrame(error_rows))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax_o, ax_r, ax_q, ax_f, ax_bias, ax_noise = axes.flat

    for cap, curve in curves.items():
        ax_o.plot(scales, curve["O"], "o-", label=f"K={cap}")
        if curve["ratio"] is not None:
            ax_r.plot(scales, curve["ratio"], "o-", label=f"K={cap}")
        if curve["Q_plus"] is not None:
            ax_q.plot(scales, curve["Q_plus"], "o-", label=f"Q+, K={cap}")
            ax_q.plot(scales, curve["Q_minus"], "x--", label=f"Q−, K={cap}")
    ax_o.plot(scales, reference_values, "k*--", label="Reference")
    if reference_ratio is not None:
        ax_r.plot(scales, reference_ratio, "k*--", label="Reference")
    ax_o.set(title="Observable convergence", xlabel="s", ylabel="O(s)")
    ax_r.set(title="Ratio convergence", xlabel="s", ylabel="R(s)")
    ax_q.set(title="Q+ / Q− convergence (may overlap)", xlabel="s", ylabel="Q(s)")

    # Model reconstructions use only parameters fitted on the positive-s grid.
    dense_s = np.linspace(0, max(fit_scales), 300)
    for name in ("single_exp", "dual_exp"):
        fitted = noiseless_fits[name]
        if "estimate" in fitted:
            model_y = np.exp(-dense_s[:, None] * np.array(fitted["rates"])) @ np.array(
                fitted["amplitudes"]
            )
            ax_f.plot(dense_s, model_y, label=name)
    for name, design in designs(curves).items():
        if name not in (f"ratio_K{highest_K}", f"Q_K{highest_K}"):
            continue
        if design is not None and noiseless_fits[name]["status"] == "ok":
            coef = np.linalg.lstsq(design, observed_means, rcond=None)[0]
            ax_f.plot(
                np.r_[0, fit_scales], np.r_[coef[0], design @ coef], "o:", label=name
            )
    ax_f.scatter(
        fit_scales,
        observed_means,
        c="black",
        marker="x",
        label="Fit observations",
        zorder=5,
    )
    ax_f.scatter(
        [0], [target_O0], c="black", marker="*", s=110, label="Held-out O(0)", zorder=5
    )
    ax_f.set(title="Noiseless fits and held-out intercept", xlabel="s", ylabel="O(s)")

    caps = sorted(curves)
    for family in ("ratio", "Q"):
        errors = [
            (noiseless_fits[f"{family}_K{k}"].get("estimate", np.nan) - target_O0)
            * error_factor
            for k in caps
        ]
        ax_bias.plot(caps, errors, "o-", label=family)
    for name, color in (("single_exp", "tab:green"), ("dual_exp", "tab:red")):
        if "estimate" in noiseless_fits[name]:
            ax_bias.axhline(
                (noiseless_fits[name]["estimate"] - target_O0) * error_factor,
                linestyle="--",
                color=color,
                label=name,
            )
    ax_bias.axhline(0, color="gray", linewidth=0.8)
    ax_bias.set_xscale("log", base=2)
    ax_bias.set_xlim(min(caps) / 1.2, max(caps) * 1.2)
    ax_bias.set_yscale("symlog", linthresh=0.01 if relative_ok else 1e-6)
    ax_bias.set_xticks(caps, [str(k) for k in caps])
    ax_bias.set(title="Signed zero-noise bias", xlabel="MPS cap K", ylabel=error_label)

    if not shot_table.empty:
        shown = [f"ratio_K{highest_K}", f"Q_K{highest_K}", "single_exp", "dual_exp"]
        metric = "relative_rmse_pct" if relative_ok else "rmse"
        pivot = shot_table.pivot(index="scenario", columns="method", values=metric)
        pivot = pivot.reindex(
            index=list(fit_config["fit"]["shot_scenarios"]), columns=shown
        )
        # Scenario labels retain unequal/exact allocations; no misleading scalar shot axis.
        for name in shown:
            ax_noise.plot(range(len(pivot)), pivot[name], "o-", label=name)
        ax_noise.set_xticks(range(len(pivot)), pivot.index, rotation=20, ha="right")
        plotted = pivot.to_numpy().astype(float)
        finite = plotted[np.isfinite(plotted)]
        if finite.size and np.all(finite > 0):
            ax_noise.set_yscale("log")
            ax_noise.set_ylim(float(finite.min()) / 1.5, float(finite.max()) * 1.5)
        elif finite.size:
            ax_noise.set_yscale("symlog", linthresh=1e-6)
            ax_noise.set_ylim(0, max(1e-6, float(finite.max()) * 1.5))
        ax_noise.set(
            title=f"Shot-noise RMSE (classical K={highest_K})",
            xlabel="Shot scenario (see allocations below)",
            ylabel="Relative RMSE (%)" if relative_ok else "RMSE",
        )
        display(
            pd.DataFrame(
                [
                    {
                        "scenario": name,
                        **{f"shots @ s={s:g}": n for s, n in zip(fit_scales, counts)},
                    }
                    for name, counts in shot_vectors(fit_config).items()
                ]
            ).fillna("exact")
        )
        display(pivot)
    else:
        ax_noise.text(
            0.5,
            0.5,
            "Shot noise disabled",
            ha="center",
            va="center",
            transform=ax_noise.transAxes,
        )

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(
        f"N={calculation_config['circuit']['N']}, L={calculation_config['circuit']['L']}, "
        f"O={calculation_config['circuit']['O']} | "
        f"reference: {ref['curve_method']}, converged={ref['converged']}"
    )
    if show:
        plt.show()

    output_dir = save_results(fits, fig, output_root) if save else None
    if output_dir is not None:
        emit(f"Saved: {output_dir.parent.name}/{output_dir.name}")
    emit("Finite-bond reference checks are empirical, not rigorous error bounds.")
    return AnalysisResults(
        fig, fit_table, shot_table, curve_table, pd.DataFrame(error_rows), output_dir
    )
