"""Reproducible curve calculation, reference selection, fitting, and reports."""

import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from . import __version__
from .config import load_config, schedules, shot_vectors
from .mps import evaluate, modes
from .exact import statevector_value, density_value
from .fitting import benchmark


def native(x):
    if isinstance(x, np.ndarray):
        return [native(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        return native(x.item())
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [native(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save(path, data):
    path.write_text(json.dumps(native(data), indent=2, allow_nan=False) + "\n")


def csv_save(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: (
                        json.dumps(native(v))
                        if isinstance(v, (dict, list, tuple))
                        else native(v)
                    )
                    for k, v in row.items()
                }
            )


def reference(config, singles, pairs, calc, progress):
    n = config["circuit"]["N"]
    cfg = config["reference"]
    scales = [0.0] + config["s_grid"]
    budget = cfg["memory_mb"] * 1024**2
    use_state = n <= cfg["statevector_max_n"] and 6 * 16 * 2**n <= budget
    use_density = n <= cfg["density_max_n"] and 8 * 16 * 4**n <= budget
    o0 = statevector_value(config) if use_state else None
    result = dict(
        O0=o0, O0_method="statevector" if use_state else None, s_grid=scales, checks=[]
    )
    if use_density:
        values = []
        for s in scales:
            progress(f"Exact density reference s={s:g}")
            values.append(density_value(config, singles, pairs, s))
        if use_state:
            np.testing.assert_allclose(values[0], o0, atol=1e-10, rtol=1e-10)
        else:
            result.update(O0=values[0], O0_method="density_matrix")
        result.update(
            values=values,
            curve_method="density_matrix",
            converged=True,
            selected_bond=None,
        )
        return result
    previous = None
    streak = 0
    values = None
    for cap in cfg["bond_dims"]:
        values = np.array([calc(cap, s, False)["value"] for s in scales])
        if previous is not None:
            delta = abs(values - previous)
            tolerance = cfg["atol"] + cfg["rtol"] * abs(values)
            passed = bool(np.all(delta <= tolerance))
            streak = streak + 1 if passed else 0
            result["checks"].append(
                dict(
                    bond=cap,
                    absolute_changes=delta.tolist(),
                    tolerances=tolerance.tolist(),
                    passed=passed,
                )
            )
        previous = values
        if streak >= cfg["consecutive"]:
            break
    result.update(
        values=values.tolist(),
        curve_method="mps",
        selected_bond=cap,
        converged=streak >= cfg["consecutive"],
    )
    # A largest bond is never assumed exact. Independently repeat a randomized reference.
    if config["mps"]["svd"] == "randomized":
        checked = copy.deepcopy(config)
        checked["mps"]["seed"] += 1
        checked["mps"]["power_iterations"] += 1
        repeat = []
        for s in scales:
            progress(f"Reference reproducibility check K={cap}, s={s:g}")
            repeat.append(evaluate(checked, singles, pairs, s, cap)["value"])
        diff = abs(np.array(repeat) - values)
        passed = bool(np.all(diff <= cfg["atol"] + cfg["rtol"] * abs(values)))
        result["reproducibility"] = dict(
            values=repeat, absolute_changes=diff.tolist(), passed=passed
        )
        result["converged"] &= passed
    if o0 is None:
        result.update(O0=float(values[0]), O0_method="mps")
    else:
        result["MPS_O0_vs_statevector_absolute_error"] = float(values[0] - o0)
    result["status"] = (
        "converged_within_requested_tolerance"
        if result["converged"]
        else "budget_exhausted_not_converged"
    )
    return result


def report(out, config, ref, baseline, noise):
    lines = [
        "# CNOT-resolved ZNE benchmark",
        "",
        f"N={config['circuit']['N']}, L={config['circuit']['L']}, phi={config['circuit']['phi']}, O={config['circuit']['O']}.",
        f"Initial state: adjacent singlet product. Forward noise placement: {config['noise']['placement']}.",
        f"Fit scales: {config['s_grid']}. Equal-weight fits; zero is excluded.",
        f"O(0) reference: {ref['O0']:.12g} ({ref['O0_method']}). Noisy curve: {ref['curve_method']}.",
        f"Reference curve converged: {ref['converged']}. Status: {ref.get('status','exact_density_reference')}.",
        "",
        "|Method|Status|Estimated O(0)|Signed error / abs(reference) (%)|",
        "|:--|:--|--:|--:|",
    ]
    for name, f in baseline.items():
        lines.append(
            f"|{name}|{f['status']}|{f.get('estimate','—')}|{f.get('relative_error_pct','—')}|"
        )
    lines += [
        "",
        "## Shot noise",
        "",
        "Paired independent binomial Pauli shots; null means the point is kept exact. Classical modes stay fixed. The exponential fits use the saved rate/amplitude bounds; boundary hits and optimizer failures are reported, not discarded.",
        "",
        "|Scenario|Method|Relative RMSE (%)|Boundary fraction|Optimizer non-success|",
        "|:--|:--|--:|--:|--:|",
    ]
    for r in noise:
        lines.append(
            f"|{r['scenario']}|{r['method']}|{r['relative_rmse_pct']}|{r['boundary_fraction']}|{r['optimizer_non_success']}|"
        )
    lines += [
        "",
        "These are simulated benchmark observations. Finite-bond reference convergence is an empirical check, not a rigorous error bound. Relative errors are undefined for a near-zero target. Undefined Q sectors, zero ratio denominators, rank-deficient fits, and insufficient grids are marked explicitly.",
        "",
        "See config.json, inputs.npz, provenance.json, reference.json, curves.json, curve_values.csv, fits.json, shot_noise.csv, and exponential_trials.json for inputs and diagnostics.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(config_path, output, overrides=None, progress=print):
    config, base = load_config(config_path, overrides)
    single, pair = schedules(config, base)
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "config.json", config)
    np.savez_compressed(out / "inputs.npz", single_rates=single, pair_rates=pair)
    source = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    save(
        out / "provenance.json",
        dict(
            version=__version__,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            source_sha256=source,
            inputs_sha256=hashlib.sha256((out / "inputs.npz").read_bytes()).hexdigest(),
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    cache = {}
    raw = []
    start = time.perf_counter()

    def calc(cap, s, unsigned):
        key = (cap, s, unsigned)
        if key not in cache:
            progress(f'MPS K={cap}, s={s:g}, {"absolute" if unsigned else "signed"}')
            v = evaluate(config, single, pair, s, cap, unsigned)
            cache[key] = v
            raw.append(
                dict(bond=cap, s=s, channel="absolute" if unsigned else "signed", **v)
            )
            csv_save(out / "mps_contractions.csv", raw)
        return cache[key]

    try:
        ref = reference(config, single, pair, calc, progress)
        save(out / "reference.json", ref)
        if not ref["converged"]:
            progress(
                "Reference bond budget exhausted: results will be labeled unconverged."
            )
        curves = {}
        table = []
        scales = [0.0] + config["s_grid"]
        for cap in config["mps"]["bond_dims"]:
            signed = [calc(cap, s, False)["value"] for s in scales]
            log_a = [calc(cap, s, True)["log_abs"] for s in scales]
            m = modes(signed, log_a)
            curves[cap] = dict(**m, O=np.array(signed), log_absolute=log_a)
            for i, s in enumerate(scales):
                row = dict(
                    bond=cap,
                    s=s,
                    O=signed[i],
                    log_absolute=log_a[i],
                    mode_status=m["status"],
                )
                for key in (
                    "ratio",
                    "Q_plus",
                    "Q_minus",
                    "Q_avg",
                    "Q_diff",
                    "Q_contrast",
                ):
                    row[key] = None if m[key] is None else m[key][i]
                table.append(row)
            save(out / "curves.json", curves)
            csv_save(out / "curve_values.csv", table)
        baseline, noise, samples = benchmark(
            config["s_grid"],
            ref["values"][1:],
            ref["O0"],
            curves,
            config["fit"],
            shot_vectors(config),
            progress,
        )
        save(out / "fits.json", baseline)
        csv_save(
            out / "noiseless_fits.csv",
            [dict(method=k, **v) for k, v in baseline.items()],
        )
        csv_save(out / "shot_noise.csv", noise)
        save(out / "exponential_trials.json", samples)
        report(out, config, ref, baseline, noise)
        save(
            out / "completion.json",
            dict(status="complete", runtime_s=time.perf_counter() - start),
        )
    except Exception as e:
        save(out / "failure.json", dict(type=type(e).__name__, message=str(e)))
        raise
    return out


def refit(input_dir, output, fit_config=None, observations=None, progress=print):
    """Refit saved curves without rerunning tensor contractions."""
    inp = Path(input_dir)
    config = json.loads((inp / "config.json").read_text())
    ref = json.loads((inp / "reference.json").read_text())
    if fit_config is not None:
        from .config import merge

        config = merge(config, {"fit": json.loads(Path(fit_config).read_text())})
    config, _ = load_config(overrides=config)
    curves = json.loads((inp / "curves.json").read_text())
    for m in curves.values():
        for key, v in m.items():
            if isinstance(v, list):
                m[key] = np.array(v)
    y = ref["values"][1:]
    if observations is not None:
        rows = list(csv.DictReader(Path(observations).open()))
        mapping = {float(r["s"]): float(r["O"]) for r in rows}
        if len(mapping) != len(rows) or set(mapping) != set(config["s_grid"]):
            raise ValueError(
                "Observation CSV must have exactly one s,O row for every fit scale"
            )
        y = [mapping[s] for s in config["s_grid"]]
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    baseline, noise, samples = benchmark(
        config["s_grid"],
        y,
        ref["O0"],
        curves,
        config["fit"],
        shot_vectors(config),
        progress,
    )
    save(out / "config.json", config)
    save(out / "reference.json", ref)
    save(out / "fits.json", baseline)
    save(
        out / "refit_inputs.json",
        dict(
            source_run=str(inp.resolve()),
            observations=y,
            external_observations=observations is not None,
            comparison_target="saved_simulation_reference",
        ),
    )
    csv_save(out / "shot_noise.csv", noise)
    save(out / "exponential_trials.json", samples)
    report(out, config, ref, baseline, noise)
    if observations is not None:
        with (out / "report.md").open("a") as f:
            f.write(
                "\nExternal observations were fitted. Reported errors compare against the saved simulation reference, not experimentally known truth. Synthetic shots use supplied observations as plug-in means.\n"
            )
    return out
