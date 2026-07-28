import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
import matplotlib.pyplot as plt


DEFAULT_CASE = "case0"


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_noisy_s(text, s_fit):
    if text.lower() in ("", "all"):
        return set(float(s) for s in s_fit)
    if text.lower() in ("none", "exact"):
        return set()
    return set(parse_float_list(text))


def float_or_nan(value):
    if value is None or value == "":
        return np.nan
    return float(value)


def first_present(row, names):
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return None


def load_observables(path, default_case=DEFAULT_CASE):
    values = defaultdict(dict)
    sigmas = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row.get("case") or default_case
            s = float(row["s"])
            value = first_present(
                row,
                ("observable", "value", "mps_value", "O_mps", "mean", "obs", "y"),
            )
            if value is None:
                raise ValueError(f"{path} must contain an observable value column")
            values[case][s] = float(value)
            sigma = first_present(row, ("sigma", "se", "stderr", "std_error"))
            if sigma is not None:
                sigmas[case][s] = float(sigma)
    return values, sigmas


def load_q_groups(path, default_case=DEFAULT_CASE, q_total_samples=None):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "case": row.get("case") or default_case,
                "method": row["method"],
                "group": int(row.get("group", 1)),
                "s": float(row["s"]),
                "q_plus": float(row["q_plus"]),
                "q_minus": float(row["q_minus"]),
                "requested_total_samples": int(float(row.get("requested_total_samples", 0) or 0)),
                "plus_num": float_or_nan(row.get("plus_num")),
                "plus_den": float_or_nan(row.get("plus_den")),
                "minus_num": float_or_nan(row.get("minus_num")),
                "minus_den": float_or_nan(row.get("minus_den")),
            }
            rows.append(parsed)

    if q_total_samples is None:
        totals = [r["requested_total_samples"] for r in rows if r["requested_total_samples"] > 0]
        q_total_samples = max(totals) if totals else 0

    if q_total_samples > 0:
        rows = [r for r in rows if r["requested_total_samples"] == q_total_samples]

    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"q_plus": {}, "q_minus": {}, "plus_num": {}, "plus_den": {}, "minus_num": {}, "minus_den": {}})))
    for row in rows:
        item = groups[row["case"]][row["method"]][row["group"]]
        s = row["s"]
        item["q_plus"][s] = row["q_plus"]
        item["q_minus"][s] = row["q_minus"]
        item["plus_num"][s] = row["plus_num"]
        item["plus_den"][s] = row["plus_den"]
        item["minus_num"][s] = row["minus_num"]
        item["minus_den"][s] = row["minus_den"]
    return groups, q_total_samples


def grouped_q_arrays(groups_for_method, s_grid):
    group_ids = sorted(groups_for_method)
    if not group_ids:
        raise ValueError("empty Q group set")
    q_plus = np.array(
        [[groups_for_method[g]["q_plus"][float(s)] for s in s_grid] for g in group_ids],
        dtype=np.float64,
    )
    q_minus = np.array(
        [[groups_for_method[g]["q_minus"][float(s)] for s in s_grid] for g in group_ids],
        dtype=np.float64,
    )
    return group_ids, q_plus, q_minus


def q_curve_from_groups(groups_for_method, s_grid, summary_mode):
    _, q_plus, q_minus = grouped_q_arrays(groups_for_method, s_grid)
    if summary_mode == "pointwise_mom":
        return np.median(q_plus, axis=0), np.median(q_minus, axis=0)
    if summary_mode == "mean":
        return np.mean(q_plus, axis=0), np.mean(q_minus, axis=0)
    if summary_mode == "pooled":
        qp = []
        qm = []
        for s in s_grid:
            plus_num = []
            plus_den = []
            minus_num = []
            minus_den = []
            for g in sorted(groups_for_method):
                item = groups_for_method[g]
                plus_num.append(item["plus_num"].get(float(s), np.nan))
                plus_den.append(item["plus_den"].get(float(s), np.nan))
                minus_num.append(item["minus_num"].get(float(s), np.nan))
                minus_den.append(item["minus_den"].get(float(s), np.nan))
            plus_num = np.asarray(plus_num, dtype=np.float64)
            plus_den = np.asarray(plus_den, dtype=np.float64)
            minus_num = np.asarray(minus_num, dtype=np.float64)
            minus_den = np.asarray(minus_den, dtype=np.float64)
            if np.any(~np.isfinite(plus_num)) or np.any(~np.isfinite(plus_den)):
                raise ValueError("pooled Q summary requires plus_num/plus_den columns")
            if np.any(~np.isfinite(minus_num)) or np.any(~np.isfinite(minus_den)):
                raise ValueError("pooled Q summary requires minus_num/minus_den columns")
            qp.append(float(np.sum(plus_num) / np.sum(plus_den)))
            qm.append(float(np.sum(minus_num) / np.sum(minus_den)))
        return np.asarray(qp, dtype=np.float64), np.asarray(qm, dtype=np.float64)
    raise ValueError(f"unknown q summary mode {summary_mode}")


def qdict(curve, s_grid):
    return {float(s): float(curve[k]) for k, s in enumerate(s_grid)}


def fit_log_modes(q_plus_dict, q_minus_dict, s_fit_q):
    ss = np.array([s for s in s_fit_q if s > 0.0], dtype=np.float64)
    ell_p = np.array([np.log(max(q_plus_dict[float(s)], 1.0e-300)) for s in ss], dtype=np.float64)
    ell_m = np.array([np.log(max(q_minus_dict[float(s)], 1.0e-300)) for s in ss], dtype=np.float64)
    ell_avg = 0.5 * (ell_p + ell_m)
    delta_ell = 0.5 * (ell_p - ell_m)

    x_avg = np.stack([-ss, ss**2, -ss**3], axis=1)
    avg_coef = np.linalg.lstsq(x_avg, ell_avg, rcond=None)[0]

    x_delta = np.stack([ss, ss**2], axis=1)
    delta_coef = np.linalg.lstsq(x_delta, delta_ell, rcond=None)[0]

    avg_rms = float(np.sqrt(np.mean((x_avg @ avg_coef - ell_avg) ** 2)))
    delta_rms = float(np.sqrt(np.mean((x_delta @ delta_coef - delta_ell) ** 2)))
    return avg_coef, delta_coef, avg_rms, delta_rms


def eval_log_modes(avg_coef, delta_coef, s):
    s = np.asarray(s, dtype=np.float64)
    ell = -avg_coef[0] * s + avg_coef[1] * s * s - avg_coef[2] * s * s * s
    delta = delta_coef[0] * s + delta_coef[1] * s * s
    env = np.exp(np.clip(ell, -700.0, 80.0))
    delta_clip = np.clip(delta, -80.0, 80.0)
    return env * np.cosh(delta_clip), env * np.sinh(delta_clip)


def build_q_design(avg_coef, delta_coef, s_fit):
    q_avg, q_diff = eval_log_modes(avg_coef, delta_coef, np.asarray(s_fit, dtype=np.float64))
    return np.column_stack([q_avg, q_diff])


def fit_uv_from_design(x, y, alpha_v, sigma):
    y = np.asarray(y, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    w = 1.0 / sigma
    xw = x * w[:, None]
    yw = y * w
    beta = np.linalg.solve(xw.T @ xw + np.diag([0.0, alpha_v]), xw.T @ yw)
    pred = x @ beta
    chi2 = float(np.sum(((pred - y) / sigma) ** 2))
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(beta[0]), float(beta[1]), rms, chi2, pred


def noisy_pauli_means(true_values, shots, rng):
    true_values = np.asarray(true_values, dtype=np.float64)
    p_plus = np.clip((1.0 + true_values) * 0.5, 0.0, 1.0)
    counts = rng.binomial(shots, p_plus)
    return 2.0 * counts / shots - 1.0


def precompute_dual_exp_pairs(s_fit, sigma, b_grid, amp_bound):
    s_fit = np.asarray(s_fit, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    pairs = []
    inv_sigma = 1.0 / sigma
    for b1 in b_grid:
        e1 = np.exp(-b1 * s_fit)
        x1 = e1 * inv_sigma
        for b2 in b_grid:
            if abs(b1 - b2) < 1.0e-14:
                continue
            e2 = np.exp(-b2 * s_fit)
            x2 = e2 * inv_sigma
            x = np.column_stack([x1, x2])
            gram = x.T @ x
            if np.linalg.cond(gram) > 1.0e12:
                continue
            beta_map = np.linalg.solve(gram, x.T)
            basis = np.column_stack([e1, e2])
            pairs.append((float(b1), float(b2), beta_map, basis, float(amp_bound)))
    return pairs


def fit_dual_exp_grid(y_obs, sigma, pairs):
    y_obs = np.asarray(y_obs, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    y_weighted = y_obs / sigma
    best = None
    for b1, b2, beta_map, basis, amp_bound in pairs:
        amps = beta_map @ y_weighted
        if np.any(np.abs(amps) > amp_bound):
            continue
        pred = basis @ amps
        resid = (pred - y_obs) / sigma
        cost = 0.5 * float(resid @ resid)
        if best is None or cost < best["cost"]:
            best = {
                "a1": float(amps[0]),
                "b1": b1,
                "a2": float(amps[1]),
                "b2": b2,
                "u_O0": float(amps[0] + amps[1]),
                "fit_rms": float(np.sqrt(np.mean((pred - y_obs) ** 2))),
                "chi2": float(resid @ resid),
                "cost": cost,
                "pred": pred,
            }
    if best is None:
        return {
            "a1": np.nan,
            "b1": np.nan,
            "a2": np.nan,
            "b2": np.nan,
            "u_O0": np.nan,
            "fit_rms": np.nan,
            "chi2": np.inf,
            "cost": np.inf,
            "pred": np.full_like(y_obs, np.nan),
        }
    return best


def summarize_trials(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["case"], row["model"])].append(row)
    out = []
    for (case, model), selected in sorted(by_key.items()):
        estimates = np.array([r["u_O0"] for r in selected], dtype=np.float64)
        true0 = selected[0]["true_O0"]
        if np.isfinite(true0) and true0 != 0.0:
            rel = (estimates - true0) / true0
        else:
            rel = np.full_like(estimates, np.nan)
        out.append(
            {
                "case": case,
                "model": model,
                "n_trials": len(selected),
                "true_O0": true0,
                "mean_u_O0": float(np.nanmean(estimates)),
                "std_u_O0": float(np.nanstd(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                "bias": float(np.nanmean(estimates) - true0) if np.isfinite(true0) else np.nan,
                "mean_rel_error": float(np.nanmean(rel)),
                "std_rel_error": float(np.nanstd(rel, ddof=1)) if len(rel) > 1 else 0.0,
                "rmse_rel": float(np.sqrt(np.nanmean(rel**2))),
                "median_rel_error": float(np.nanmedian(rel)),
                "p05_rel_error": float(np.nanquantile(rel, 0.05)),
                "p95_rel_error": float(np.nanquantile(rel, 0.95)),
                "mean_fit_rms": float(np.nanmean([r["fit_rms"] for r in selected])),
                "mean_chi2": float(np.nanmean([r["chi2"] for r in selected])),
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_hist(summary_rows, trial_rows, out_path):
    cases = sorted({r["case"] for r in trial_rows})
    models = sorted({r["model"] for r in trial_rows})
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.5, max(3.0, 2.4 * len(cases))), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        for model in models:
            selected = [r for r in trial_rows if r["case"] == case and r["model"] == model]
            rel = np.array([r["rel_error"] for r in selected], dtype=np.float64)
            rel = rel[np.isfinite(rel)]
            if rel.size == 0:
                continue
            ax.hist(rel, bins=80, histtype="step", density=True, linewidth=1.4, label=model)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(case)
        ax.set_xlabel("relative error in extrapolated O(0)")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_curves(curve_rows, out_path):
    cases = sorted({r["case"] for r in curve_rows})
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.5, max(3.3, 3.0 * len(cases))), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        rows = [r for r in curve_rows if r["case"] == case]
        data = [r for r in rows if r["model"] == "data"]
        ax.errorbar(
            [r["s"] for r in data],
            [r["y_observed"] for r in data],
            yerr=[r["sigma"] for r in data],
            fmt="o",
            color="black",
            label="observed O(s)",
        )
        for model in sorted({r["model"] for r in rows if r["model"] != "data"}):
            selected = [r for r in rows if r["model"] == model]
            selected.sort(key=lambda r: r["s"])
            ax.plot([r["s"] for r in selected], [r["y_fit"] for r in selected], label=model)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_title(f"{case}, trial 1")
        ax.set_xlabel("s")
        ax.set_ylabel("O(s)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def model_curve_from_dual_exp(fit, s_values):
    s_values = np.asarray(s_values, dtype=np.float64)
    return fit["a1"] * np.exp(-fit["b1"] * s_values) + fit["a2"] * np.exp(-fit["b2"] * s_values)


def output_paths(out_prefix):
    out_prefix = Path(out_prefix)
    return {
        "q_fits_csv": out_prefix.with_name(f"{out_prefix.name}_q_fits.csv"),
        "trials_csv": out_prefix.with_name(f"{out_prefix.name}_trials.csv"),
        "summary_csv": out_prefix.with_name(f"{out_prefix.name}_summary.csv"),
        "curves_csv": out_prefix.with_name(f"{out_prefix.name}_curves.csv"),
        "hist_png": out_prefix.with_name(f"{out_prefix.name}_hist.png"),
        "curves_png": out_prefix.with_name(f"{out_prefix.name}_curves.png"),
    }


def comma_arg(value):
    if isinstance(value, np.ndarray):
        return ",".join(f"{float(x):g}" for x in value.tolist())
    if isinstance(value, (list, tuple)):
        return ",".join(f"{float(x):g}" for x in value)
    return str(value)


def run_zne_fit_compare(
    *,
    q_groups,
    observables,
    out_prefix="zne_fit_compare",
    cases="",
    default_case=DEFAULT_CASE,
    methods="gate,layer",
    q_total_samples=0,
    q_summary="pointwise_mom",
    s_grid="1,2,4,8",
    s_fit_q="1,2,4,8",
    s_fit_o="1,2,4,8",
    noisy_s="all",
    shots=100_000,
    use_observed=False,
    exact_sigma=1.0e-6,
    n_trials=500,
    seed=20260728,
    alpha_gate=3.0e-8,
    alpha_layer=1.0e-8,
    dual_exp_rate_min=1.0e-3,
    dual_exp_rate_max=5.0,
    dual_exp_grid_size=100,
    dual_exp_amp_bound=10.0,
    curve_s_max=8.0,
    curve_points=300,
):
    """Notebook-friendly wrapper around the command-line ZNE comparison.

    Returns a dictionary of generated output paths. The implementation calls the
    same ``main`` path as the CLI so notebook and command-line results stay in
    lockstep.
    """

    argv = [
        "fit_zne_compare.py",
        "--q-groups",
        str(q_groups),
        "--observables",
        str(observables),
        "--cases",
        str(cases),
        "--default-case",
        str(default_case),
        "--methods",
        str(methods),
        "--q-total-samples",
        str(q_total_samples),
        "--q-summary",
        str(q_summary),
        "--s-grid",
        comma_arg(s_grid),
        "--s-fit-q",
        comma_arg(s_fit_q),
        "--s-fit-o",
        comma_arg(s_fit_o),
        "--noisy-s",
        comma_arg(noisy_s),
        "--shots",
        str(shots),
        "--exact-sigma",
        str(exact_sigma),
        "--n-trials",
        str(n_trials),
        "--seed",
        str(seed),
        "--alpha-gate",
        str(alpha_gate),
        "--alpha-layer",
        str(alpha_layer),
        "--dual-exp-rate-min",
        str(dual_exp_rate_min),
        "--dual-exp-rate-max",
        str(dual_exp_rate_max),
        "--dual-exp-grid-size",
        str(dual_exp_grid_size),
        "--dual-exp-amp-bound",
        str(dual_exp_amp_bound),
        "--curve-s-max",
        str(curve_s_max),
        "--curve-points",
        str(curve_points),
        "--out-prefix",
        str(out_prefix),
    ]
    if use_observed:
        argv.append("--use-observed")

    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv
    return output_paths(out_prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-assisted ZNE fits against dual-exponential ZNE."
    )
    parser.add_argument("--q-groups", required=True, help="CSV with grouped q_plus/q_minus data")
    parser.add_argument("--observables", required=True, help="CSV with s and observable/mps_value/O_mps/value")
    parser.add_argument("--cases", default="", help="comma-separated case labels; default uses all observable cases")
    parser.add_argument("--default-case", default=DEFAULT_CASE)
    parser.add_argument("--methods", default="gate,layer")
    parser.add_argument("--q-total-samples", type=int, default=0, help="select requested_total_samples; 0 uses largest present")
    parser.add_argument("--q-summary", choices=["pointwise_mom", "mean", "pooled"], default="pointwise_mom")
    parser.add_argument("--s-grid", default="1,2,4,8", help="s points present in Q data")
    parser.add_argument("--s-fit-q", default="1,2,4,8", help="Q points used for log-mode fit")
    parser.add_argument("--s-fit-o", default="1,2,4,8", help="observable points used for final fit")
    parser.add_argument("--noisy-s", default="all", help="all, none, or comma-separated observable s values to shot-sample")
    parser.add_argument("--shots", type=int, default=100_000, help="shots for simulated noisy observable points; 0 means exact")
    parser.add_argument("--use-observed", action="store_true", help="fit loaded observable values directly instead of simulating noise")
    parser.add_argument("--exact-sigma", type=float, default=1.0e-6, help="sigma for exact/classical points")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--alpha-gate", type=float, default=3.0e-8)
    parser.add_argument("--alpha-layer", type=float, default=1.0e-8)
    parser.add_argument("--dual-exp-rate-min", type=float, default=1.0e-3)
    parser.add_argument("--dual-exp-rate-max", type=float, default=5.0)
    parser.add_argument("--dual-exp-grid-size", type=int, default=100)
    parser.add_argument("--dual-exp-amp-bound", type=float, default=10.0)
    parser.add_argument("--curve-s-max", type=float, default=8.0)
    parser.add_argument("--curve-points", type=int, default=300)
    parser.add_argument("--out-prefix", default="zne_fit_compare")
    args = parser.parse_args()

    s_grid = np.array(parse_float_list(args.s_grid), dtype=np.float64)
    s_fit_q = parse_float_list(args.s_fit_q)
    s_fit_o = np.array(parse_float_list(args.s_fit_o), dtype=np.float64)
    noisy_s = parse_noisy_s(args.noisy_s, s_fit_o)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    observables, observed_sigmas = load_observables(args.observables, args.default_case)
    q_groups, selected_q_total = load_q_groups(
        args.q_groups,
        args.default_case,
        None if args.q_total_samples == 0 else args.q_total_samples,
    )
    cases = [x.strip() for x in args.cases.split(",") if x.strip()] or sorted(observables)
    rng = np.random.default_rng(args.seed)

    print("ZNE fitting comparison")
    print("----------------------")
    print(f"cases={cases}")
    print(f"methods={methods}")
    print(f"q_summary={args.q_summary} q_total_samples={selected_q_total or 'all'}")
    print(f"s_fit_q={s_fit_q}")
    print(f"s_fit_o={list(s_fit_o)}")
    print(f"shots={args.shots} noisy_s={sorted(noisy_s)} use_observed={args.use_observed}")

    q_fit_rows = []
    q_designs = {}
    for case in cases:
        if case not in q_groups:
            raise ValueError(f"case {case!r} not found in Q groups")
        for method in methods:
            if method not in q_groups[case]:
                raise ValueError(f"method {method!r} not found for case {case!r}")
            qp_curve, qm_curve = q_curve_from_groups(q_groups[case][method], s_grid, args.q_summary)
            avg_coef, delta_coef, avg_rms, delta_rms = fit_log_modes(
                qdict(qp_curve, s_grid),
                qdict(qm_curve, s_grid),
                s_fit_q,
            )
            q_designs[(case, method)] = {
                "x": build_q_design(avg_coef, delta_coef, s_fit_o),
                "avg_coef": avg_coef,
                "delta_coef": delta_coef,
            }
            q_fit_rows.append(
                {
                    "case": case,
                    "method": method,
                    "q_summary": args.q_summary,
                    "q_total_samples": selected_q_total,
                    "a1": float(avg_coef[0]),
                    "a2": float(avg_coef[1]),
                    "a3": float(avg_coef[2]),
                    "d1": float(delta_coef[0]),
                    "d2": float(delta_coef[1]),
                    "ell_avg_rms": avg_rms,
                    "delta_ell_rms": delta_rms,
                }
            )

    rate_grid = np.geomspace(args.dual_exp_rate_min, args.dual_exp_rate_max, args.dual_exp_grid_size)
    trial_rows = []
    curve_rows = []
    s_dense = np.linspace(0.0, args.curve_s_max, args.curve_points)

    for case in cases:
        obs = observables[case]
        true0 = obs.get(0.0, np.nan)
        missing = [float(s) for s in s_fit_o if float(s) not in obs]
        if missing:
            raise ValueError(f"case {case!r} missing observable s values {missing}")
        y_ref = np.array([obs[float(s)] for s in s_fit_o], dtype=np.float64)
        noisy_mask = np.array([float(s) in noisy_s for s in s_fit_o], dtype=bool)
        base_sigma = np.full_like(y_ref, args.exact_sigma, dtype=np.float64)
        if args.shots > 0:
            shot_sigma = np.sqrt(np.maximum(1.0 - y_ref * y_ref, 0.0) / args.shots)
            base_sigma = np.where(noisy_mask, shot_sigma, base_sigma)
        for k, s in enumerate(s_fit_o):
            if s in observed_sigmas[case]:
                base_sigma[k] = observed_sigmas[case][float(s)]
        base_sigma = np.maximum(base_sigma, 1.0e-15)
        dual_pairs = precompute_dual_exp_pairs(
            s_fit_o,
            base_sigma,
            rate_grid,
            args.dual_exp_amp_bound,
        )

        n_trials = 1 if args.use_observed else args.n_trials
        for trial in range(1, n_trials + 1):
            if args.use_observed or args.shots <= 0:
                y_obs = y_ref.copy()
            else:
                y_obs = y_ref.copy()
                if np.any(noisy_mask):
                    y_obs[noisy_mask] = noisy_pauli_means(y_ref[noisy_mask], args.shots, rng)

            dual = fit_dual_exp_grid(y_obs, base_sigma, dual_pairs)
            rel = (dual["u_O0"] - true0) / true0 if np.isfinite(true0) and true0 != 0.0 else np.nan
            trial_rows.append(
                {
                    "case": case,
                    "model": "dual_exp",
                    "trial": trial,
                    "u_O0": dual["u_O0"],
                    "v": np.nan,
                    "true_O0": true0,
                    "rel_error": rel,
                    "fit_rms": dual["fit_rms"],
                    "chi2": dual["chi2"],
                    "success": bool(np.isfinite(dual["u_O0"])),
                    "a1": dual["a1"],
                    "b1": dual["b1"],
                    "a2": dual["a2"],
                    "b2": dual["b2"],
                }
            )

            if trial == 1:
                for s, y, sigma in zip(s_fit_o, y_obs, base_sigma):
                    curve_rows.append(
                        {
                            "case": case,
                            "model": "data",
                            "trial": trial,
                            "s": float(s),
                            "y_observed": float(y),
                            "sigma": float(sigma),
                            "y_fit": np.nan,
                        }
                    )
                for s, y in zip(s_dense, model_curve_from_dual_exp(dual, s_dense)):
                    curve_rows.append(
                        {
                            "case": case,
                            "model": "dual_exp",
                            "trial": trial,
                            "s": float(s),
                            "y_observed": np.nan,
                            "sigma": np.nan,
                            "y_fit": float(y),
                        }
                    )

            for method in methods:
                alpha_v = args.alpha_gate if method == "gate" else args.alpha_layer
                design = q_designs[(case, method)]
                u, v, fit_rms, chi2, pred = fit_uv_from_design(design["x"], y_obs, alpha_v, base_sigma)
                model = f"{method}_qmom"
                rel = (u - true0) / true0 if np.isfinite(true0) and true0 != 0.0 else np.nan
                trial_rows.append(
                    {
                        "case": case,
                        "model": model,
                        "trial": trial,
                        "u_O0": u,
                        "v": v,
                        "true_O0": true0,
                        "rel_error": rel,
                        "fit_rms": fit_rms,
                        "chi2": chi2,
                        "success": True,
                        "a1": np.nan,
                        "b1": np.nan,
                        "a2": np.nan,
                        "b2": np.nan,
                    }
                )
                if trial == 1:
                    q_avg_dense, q_diff_dense = eval_log_modes(
                        design["avg_coef"],
                        design["delta_coef"],
                        s_dense,
                    )
                    y_fit_dense = u * q_avg_dense + v * q_diff_dense
                    for s, y in zip(s_dense, y_fit_dense):
                        curve_rows.append(
                            {
                                "case": case,
                                "model": model,
                                "trial": trial,
                                "s": float(s),
                                "y_observed": np.nan,
                                "sigma": np.nan,
                                "y_fit": float(y),
                            }
                        )

    summary_rows = summarize_trials(trial_rows)
    out_prefix = Path(args.out_prefix)
    write_csv(out_prefix.with_name(f"{out_prefix.name}_q_fits.csv"), q_fit_rows)
    write_csv(out_prefix.with_name(f"{out_prefix.name}_trials.csv"), trial_rows)
    write_csv(out_prefix.with_name(f"{out_prefix.name}_summary.csv"), summary_rows)
    write_csv(out_prefix.with_name(f"{out_prefix.name}_curves.csv"), curve_rows)
    plot_hist(summary_rows, trial_rows, out_prefix.with_name(f"{out_prefix.name}_hist.png"))
    plot_curves(curve_rows, out_prefix.with_name(f"{out_prefix.name}_curves.png"))

    for row in summary_rows:
        print(
            f"{row['case']:14s} {row['model']:11s} "
            f"mean_rel={row['mean_rel_error']:+.3%} "
            f"std_rel={row['std_rel_error']:.3%} "
            f"rmse_rel={row['rmse_rel']:.3%} "
            f"p05/p95=({row['p05_rel_error']:+.3%},{row['p95_rel_error']:+.3%})"
        )
    print(f"wrote {out_prefix.name}_q_fits.csv")
    print(f"wrote {out_prefix.name}_trials.csv")
    print(f"wrote {out_prefix.name}_summary.csv")
    print(f"wrote {out_prefix.name}_curves.csv")
    print(f"wrote {out_prefix.name}_hist.png")
    print(f"wrote {out_prefix.name}_curves.png")


if __name__ == "__main__":
    main()
