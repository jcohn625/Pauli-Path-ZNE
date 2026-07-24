import argparse
import csv
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Pauli_path_Heis_full_layer_sampling_restricted as layer
import Pauli_path_Heis_mixture_trigger as gate
from benchmark_q_curve_convergence import (
    dp_q_curve_aggregate,
    find_s1_index,
    gate_q_curve_aggregate,
    parse_s_grid,
    product_eta_from_lambda,
    row_from_aggregate,
)
from qpm_numba_utils import qpm_grid_from_outputs, qpm_grid_from_outputs_no_iw


def parse_int_list(text):
    return [int(float(x)) for x in text.split(",") if x.strip()]


def existing_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["method"], row["estimator"], int(row["n_samples"]), int(row["batch"])))
    return done


def append_rows(path, rows):
    fields = [
        "method",
        "estimator",
        "n_samples",
        "batch",
        "s",
        "q_plus",
        "q_minus",
        "runtime_s",
        "nonzero_frac",
        "triggered_frac",
    ]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def single_rows(method, n_samples, batch, qpm, runtime, triggered_frac, s_grid):
    counts = qpm[-1]
    total = np.sum(counts)
    nonzero_frac = float((counts[0] + counts[1]) / total) if total else np.nan
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "estimator": "single_final_pauli",
                "n_samples": n_samples,
                "batch": batch,
                "s": float(s),
                "q_plus": float(qpm[0][k]),
                "q_minus": float(qpm[1][k]),
                "runtime_s": runtime,
                "nonzero_frac": nonzero_frac,
                "triggered_frac": triggered_frac,
            }
        )
    return rows


def final_dp_rows(method, n_samples, batch, result, s_grid):
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "estimator": "final_layer_dp",
                "n_samples": n_samples,
                "batch": batch,
                "s": float(s),
                "q_plus": float(result["q_plus"][k]),
                "q_minus": float(result["q_minus"][k]),
                "runtime_s": float(result["runtime"]),
                "nonzero_frac": 1.0,
                "triggered_frac": float(result["triggered_frac"]),
            }
        )
    return rows


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "method": row["method"],
                    "estimator": row["estimator"],
                    "n_samples": int(row["n_samples"]),
                    "batch": int(row["batch"]),
                    "s": float(row["s"]),
                    "q_plus": float(row["q_plus"]),
                    "q_minus": float(row["q_minus"]),
                    "runtime_s": float(row["runtime_s"]),
                    "nonzero_frac": float(row["nonzero_frac"]),
                    "triggered_frac": float(row["triggered_frac"]),
                }
            )
    return rows


def nanstd(x):
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)
    if np.sum(good) <= 1:
        return np.nan
    return float(np.std(x[good], ddof=1))


def summarize(rows):
    out = []
    keys = sorted({(r["method"], r["estimator"], r["n_samples"], r["s"]) for r in rows})
    for method, estimator, n_samples, s in keys:
        selected = [
            r
            for r in rows
            if r["method"] == method
            and r["estimator"] == estimator
            and r["n_samples"] == n_samples
            and r["s"] == s
        ]
        qp = np.array([r["q_plus"] for r in selected], dtype=np.float64)
        qm = np.array([r["q_minus"] for r in selected], dtype=np.float64)
        out.append(
            {
                "method": method,
                "estimator": estimator,
                "n_samples": n_samples,
                "s": s,
                "n_batches": len(selected),
                "q_plus_mean": float(np.nanmean(qp)),
                "q_plus_std": nanstd(qp),
                "q_minus_mean": float(np.nanmean(qm)),
                "q_minus_std": nanstd(qm),
                "mean_runtime_s": float(np.mean([r["runtime_s"] for r in selected])),
                "mean_nonzero_frac": float(np.nanmean([r["nonzero_frac"] for r in selected])),
                "mean_triggered_frac": float(np.nanmean([r["triggered_frac"] for r in selected])),
            }
        )
    return out


def write_summary(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, out_path):
    methods = sorted({r["method"] for r in summary})
    s_values = sorted({r["s"] for r in summary})
    sectors = ("plus", "minus")
    fig, axes = plt.subplots(
        len(methods) * len(sectors),
        len(s_values),
        figsize=(3.5 * len(s_values), 2.7 * len(methods) * len(sectors)),
        squeeze=False,
        sharex=True,
    )
    colors = {"single_final_pauli": "C0", "final_layer_dp": "C3"}
    labels = {"single_final_pauli": "single final Pauli", "final_layer_dp": "final DP"}
    for mi, method in enumerate(methods):
        for si, sector in enumerate(sectors):
            row_idx = mi * len(sectors) + si
            for col_idx, s in enumerate(s_values):
                ax = axes[row_idx, col_idx]
                for estimator in ("single_final_pauli", "final_layer_dp"):
                    selected = [
                        r
                        for r in summary
                        if r["method"] == method
                        and r["estimator"] == estimator
                        and r["s"] == s
                    ]
                    selected.sort(key=lambda r: r["n_samples"])
                    if not selected:
                        continue
                    x = np.array([r["n_samples"] for r in selected], dtype=np.float64)
                    y = np.array([r[f"q_{sector}_mean"] for r in selected], dtype=np.float64)
                    e = np.array([r[f"q_{sector}_std"] for r in selected], dtype=np.float64)
                    ax.errorbar(
                        x,
                        y,
                        yerr=e,
                        marker="o",
                        linewidth=1.2,
                        capsize=2.5,
                        color=colors[estimator],
                        label=labels[estimator],
                    )
                ax.set_xscale("log")
                ax.set_title(f"s={s:g}")
                ax.grid(True, alpha=0.25)
                if col_idx == 0:
                    ax.set_ylabel(f"{method} Q{sector[0]}")
                if row_idx == axes.shape[0] - 1:
                    ax.set_xlabel("samples per batch")
                if row_idx == 0 and col_idx == len(s_values) - 1:
                    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Convergence with sample count: single final Pauli vs final-layer DP", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_compact(summary):
    print("method estimator samples s Q+mean Q+std Q-mean Q-std runtime")
    for r in summary:
        print(
            f"{r['method']:5s} {r['estimator']:18s} {r['n_samples']:>8g} "
            f"{r['s']:>4g} {r['q_plus_mean']:.6g} {r['q_plus_std']:.2g} "
            f"{r['q_minus_mean']:.6g} {r['q_minus_std']:.2g} "
            f"{r['mean_runtime_s']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", default="1,2,4,8")
    parser.add_argument("--sample-sizes", default="1e4,1e5,1e6")
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument("--methods", default="gate,layer")
    parser.add_argument("--out-prefix", default="q_finaldp_vs_single_convergence_N20_L6_Z6Z10_phi02")
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    sample_sizes = parse_int_list(args.sample_sizes)
    s_grid = parse_s_grid(args.s_grid)
    s1_idx = find_s1_index(s_grid)

    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[args.q1] = 3
    init_pauli[args.q2] = 3
    lam_xyz = np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)
    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    alpha_schedule = np.ones(args.n_steps, dtype=np.float64)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    out_prefix = Path(args.out_prefix)
    rows_path = out_prefix.with_name(f"{out_prefix.name}_batches.csv")
    summary_path = out_prefix.with_name(f"{out_prefix.name}_summary.csv")
    plot_path = out_prefix.with_name(f"{out_prefix.name}.png")

    print("Final-layer convergence sweep", flush=True)
    print("-----------------------------", flush=True)
    print(f"N={args.n_qubits} layers={args.n_steps} observable=Z_{args.q1}Z_{args.q2}", flush=True)
    print(f"phi={args.phi} s_grid={s_grid}", flush=True)
    print(f"sample_sizes={sample_sizes} n_batches={args.n_batches}", flush=True)
    print(f"methods={methods}", flush=True)

    print("\nWarming kernels...", flush=True)
    gate.evolve_many_triggered_mixture_layers(
        init_pauli,
        even_gates,
        odd_gates,
        trans_g,
        probs_g,
        is_comm_g,
        sign_g,
        amp_factor_g,
        eta_xyz,
        alpha_schedule,
        args.n_steps,
        8,
    )
    gate_q_curve_aggregate(
        init_pauli,
        even_gates,
        odd_gates,
        trans_g,
        probs_g,
        is_comm_g,
        sign_g,
        amp_factor_g,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        args.n_steps,
        8,
        s_grid,
        s1_idx,
    )
    layer.evolve_many_full_layer_sampling_mixture_restricted(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        alpha_schedule,
        8,
        use_restricted=True,
        use_trigger=True,
    )
    dp_q_curve_aggregate(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        args.n_steps,
        8,
        s_grid,
        s1_idx,
    )
    print("Warmup complete.", flush=True)

    done = existing_done(rows_path)
    for n_samples in sample_sizes:
        for batch in range(1, args.n_batches + 1):
            if "gate" in methods and ("gate", "single_final_pauli", n_samples, batch) not in done:
                t0 = time.perf_counter()
                p, so, a, wi, d = gate.evolve_many_triggered_mixture_layers(
                    init_pauli,
                    even_gates,
                    odd_gates,
                    trans_g,
                    probs_g,
                    is_comm_g,
                    sign_g,
                    amp_factor_g,
                    eta_xyz,
                    alpha_schedule,
                    args.n_steps,
                    n_samples,
                )
                runtime = time.perf_counter() - t0
                qpm = qpm_grid_from_outputs(p, so, a, wi, d, s_grid)
                append_rows(rows_path, single_rows("gate", n_samples, batch, qpm, runtime, np.nan, s_grid))
                done.add(("gate", "single_final_pauli", n_samples, batch))
                del p, so, a, wi, d, qpm
                gc.collect()
                print(f"gate single n={n_samples:g} batch {batch}/{args.n_batches} {runtime:.3f}s", flush=True)

            if "gate" in methods and ("gate", "final_layer_dp", n_samples, batch) not in done:
                t0 = time.perf_counter()
                agg = gate_q_curve_aggregate(
                    init_pauli,
                    even_gates,
                    odd_gates,
                    trans_g,
                    probs_g,
                    is_comm_g,
                    sign_g,
                    amp_factor_g,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    args.n_steps,
                    n_samples,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, n_samples, time.perf_counter() - t0)
                append_rows(rows_path, final_dp_rows("gate", n_samples, batch, result, s_grid))
                done.add(("gate", "final_layer_dp", n_samples, batch))
                print(f"gate finalDP n={n_samples:g} batch {batch}/{args.n_batches} {result['runtime']:.3f}s", flush=True)

            if "layer" in methods and ("layer", "single_final_pauli", n_samples, batch) not in done:
                t0 = time.perf_counter()
                p, pe, so, a, d, tr = layer.evolve_many_full_layer_sampling_mixture_restricted(
                    init_pauli,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    alpha_schedule,
                    n_samples,
                    use_restricted=True,
                    use_trigger=True,
                )
                runtime = time.perf_counter() - t0
                qpm = qpm_grid_from_outputs_no_iw(p, so, a, d, s_grid)
                append_rows(rows_path, single_rows("layer", n_samples, batch, qpm, runtime, float(np.mean(tr)), s_grid))
                done.add(("layer", "single_final_pauli", n_samples, batch))
                del p, pe, so, a, d, tr, qpm
                gc.collect()
                print(f"layer single n={n_samples:g} batch {batch}/{args.n_batches} {runtime:.3f}s", flush=True)

            if "layer" in methods and ("layer", "final_layer_dp", n_samples, batch) not in done:
                t0 = time.perf_counter()
                agg = dp_q_curve_aggregate(
                    init_pauli,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    args.n_steps,
                    n_samples,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, n_samples, time.perf_counter() - t0)
                append_rows(rows_path, final_dp_rows("layer", n_samples, batch, result, s_grid))
                done.add(("layer", "final_layer_dp", n_samples, batch))
                print(f"layer finalDP n={n_samples:g} batch {batch}/{args.n_batches} {result['runtime']:.3f}s", flush=True)

    rows = load_rows(rows_path)
    summary = summarize(rows)
    write_summary(summary_path, summary)
    plot_summary(summary, plot_path)
    print_compact(summary)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
