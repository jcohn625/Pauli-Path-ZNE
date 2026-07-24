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


def append_rows(path, rows):
    fieldnames = [
        "method",
        "estimator",
        "batch",
        "s",
        "q_plus",
        "q_minus",
        "q_plus_se_internal",
        "q_minus_se_internal",
        "nonzero_frac",
        "target_mean_s1",
        "target_se_s1",
        "runtime_s",
        "triggered_frac",
        "n_samples",
        "n_qubits",
        "n_steps",
        "phi",
        "lambda_base",
        "q1",
        "q2",
    ]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def existing_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["method"], row["estimator"], int(row["batch"])))
    return done


def final_dp_rows(method, batch, result, s_grid, args):
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "estimator": "final_layer_dp",
                "batch": batch,
                "s": float(s),
                "q_plus": float(result["q_plus"][k]),
                "q_minus": float(result["q_minus"][k]),
                "q_plus_se_internal": np.nan,
                "q_minus_se_internal": np.nan,
                "nonzero_frac": 1.0,
                "target_mean_s1": float(result["mean"]),
                "target_se_s1": float(result["se"]),
                "runtime_s": float(result["runtime"]),
                "triggered_frac": float(result["triggered_frac"]),
                "n_samples": int(args.n_samples),
                "n_qubits": int(args.n_qubits),
                "n_steps": int(args.n_steps),
                "phi": float(args.phi),
                "lambda_base": float(args.lambda_base),
                "q1": int(args.q1),
                "q2": int(args.q2),
            }
        )
    return rows


def single_rows(method, batch, qpm, runtime, triggered_frac, s_grid, args):
    counts = qpm[-1]
    denom = np.sum(counts)
    nonzero_frac = float((counts[0] + counts[1]) / denom) if denom else 0.0
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "estimator": "single_final_pauli",
                "batch": batch,
                "s": float(s),
                "q_plus": float(qpm[0][k]),
                "q_minus": float(qpm[1][k]),
                "q_plus_se_internal": float(qpm[4][k]),
                "q_minus_se_internal": float(qpm[5][k]),
                "nonzero_frac": nonzero_frac,
                "target_mean_s1": np.nan,
                "target_se_s1": np.nan,
                "runtime_s": float(runtime),
                "triggered_frac": float(triggered_frac),
                "n_samples": int(args.n_samples),
                "n_qubits": int(args.n_qubits),
                "n_steps": int(args.n_steps),
                "phi": float(args.phi),
                "lambda_base": float(args.lambda_base),
                "q1": int(args.q1),
                "q2": int(args.q2),
            }
        )
    return rows


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out = dict(row)
            for key in (
                "s",
                "q_plus",
                "q_minus",
                "q_plus_se_internal",
                "q_minus_se_internal",
                "nonzero_frac",
                "target_mean_s1",
                "target_se_s1",
                "runtime_s",
                "triggered_frac",
                "phi",
                "lambda_base",
            ):
                out[key] = float(out[key])
            for key in ("batch", "n_samples", "n_qubits", "n_steps", "q1", "q2"):
                out[key] = int(out[key])
            rows.append(out)
    return rows


def summarize(rows):
    out = []
    keys = sorted({(r["method"], r["estimator"], r["s"]) for r in rows})
    for method, estimator, s in keys:
        selected = [r for r in rows if r["method"] == method and r["estimator"] == estimator and r["s"] == s]
        qp = np.array([r["q_plus"] for r in selected], dtype=np.float64)
        qm = np.array([r["q_minus"] for r in selected], dtype=np.float64)
        rt_by_batch = {}
        nz_by_batch = {}
        tr_by_batch = {}
        for r in selected:
            rt_by_batch[r["batch"]] = r["runtime_s"]
            nz_by_batch[r["batch"]] = r["nonzero_frac"]
            tr_by_batch[r["batch"]] = r["triggered_frac"]
        out.append(
            {
                "method": method,
                "estimator": estimator,
                "s": s,
                "n_batches": len(selected),
                "q_plus_mean": float(np.mean(qp)),
                "q_plus_std": float(np.std(qp, ddof=1)) if len(qp) > 1 else 0.0,
                "q_plus_median": float(np.median(qp)),
                "q_minus_mean": float(np.mean(qm)),
                "q_minus_std": float(np.std(qm, ddof=1)) if len(qm) > 1 else 0.0,
                "q_minus_median": float(np.median(qm)),
                "mean_runtime_s": float(np.mean(list(rt_by_batch.values()))),
                "mean_nonzero_frac": float(np.mean(list(nz_by_batch.values()))),
                "mean_triggered_frac": float(np.mean(list(tr_by_batch.values()))),
            }
        )
    return out


def add_differences(summary):
    rows = list(summary)
    by_key = {(r["method"], r["s"], r["estimator"]): r for r in summary}
    for method in sorted({r["method"] for r in summary}):
        for s in sorted({r["s"] for r in summary if r["method"] == method}):
            a = by_key.get((method, s, "single_final_pauli"))
            b = by_key.get((method, s, "final_layer_dp"))
            if a is None or b is None:
                continue
            rows.append(
                {
                    "method": method,
                    "estimator": "final_dp_minus_single",
                    "s": s,
                    "n_batches": min(a["n_batches"], b["n_batches"]),
                    "q_plus_mean": b["q_plus_mean"] - a["q_plus_mean"],
                    "q_plus_std": np.nan,
                    "q_plus_median": b["q_plus_median"] - a["q_plus_median"],
                    "q_minus_mean": b["q_minus_mean"] - a["q_minus_mean"],
                    "q_minus_std": np.nan,
                    "q_minus_median": b["q_minus_median"] - a["q_minus_median"],
                    "mean_runtime_s": b["mean_runtime_s"] - a["mean_runtime_s"],
                    "mean_nonzero_frac": b["mean_nonzero_frac"] - a["mean_nonzero_frac"],
                    "mean_triggered_frac": b["mean_triggered_frac"] - a["mean_triggered_frac"],
                }
            )
    return rows


def write_summary(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, out_path):
    base = [r for r in summary if r["estimator"] != "final_dp_minus_single"]
    methods = sorted({r["method"] for r in base})
    s_values = sorted({r["s"] for r in base})
    fig, axes = plt.subplots(len(methods), len(s_values), figsize=(3.4 * len(s_values), 2.8 * len(methods)), squeeze=False)
    colors = {"single_final_pauli": "C0", "final_layer_dp": "C3"}
    labels = {"single_final_pauli": "single final Pauli", "final_layer_dp": "final DP"}
    for mi, method in enumerate(methods):
        for si, s in enumerate(s_values):
            ax = axes[mi, si]
            for estimator in ("single_final_pauli", "final_layer_dp"):
                row = next((r for r in base if r["method"] == method and r["estimator"] == estimator and r["s"] == s), None)
                if row is None:
                    continue
                x = np.array([0, 1], dtype=np.float64)
                y = np.array([row["q_plus_mean"], row["q_minus_mean"]], dtype=np.float64)
                e = np.array([row["q_plus_std"], row["q_minus_std"]], dtype=np.float64)
                ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=labels[estimator], color=colors[estimator])
            ax.set_xticks([0, 1], ["Q+", "Q-"])
            ax.set_title(f"{method}, s={s:g}")
            ax.grid(True, alpha=0.25)
            if si == 0:
                ax.set_ylabel("batch mean +/- batch std")
            if mi == 0 and si == len(s_values) - 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Single final Pauli vs final-layer DP Q estimates", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_summary(summary):
    print("method estimator s Q+mean Q+std Q-mean Q-std runtime nonzero triggered")
    for r in summary:
        if r["estimator"] == "final_dp_minus_single":
            continue
        print(
            f"{r['method']:5s} {r['estimator']:18s} {r['s']:>4g} "
            f"{r['q_plus_mean']:.6g} {r['q_plus_std']:.2g} "
            f"{r['q_minus_mean']:.6g} {r['q_minus_std']:.2g} "
            f"{r['mean_runtime_s']:.2f} {r['mean_nonzero_frac']:.3g} "
            f"{r['mean_triggered_frac']:.3g}"
        )
    print("\nfinal DP minus single-final-Pauli means")
    for r in summary:
        if r["estimator"] != "final_dp_minus_single":
            continue
        print(
            f"{r['method']:5s} s={r['s']:>4g} "
            f"dQ+={r['q_plus_mean']:+.6g} dQ-={r['q_minus_mean']:+.6g} "
            f"dRuntime={r['mean_runtime_s']:+.2f}s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=36)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", default="1,2,4,8")
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--methods", default="gate,layer")
    parser.add_argument("--estimators", default="single_final_pauli,final_layer_dp")
    parser.add_argument("--out-prefix", default="q_finaldp_vs_single_N36_L16_Z6Z10_phi02_8x1e6")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    estimators = [m.strip() for m in args.estimators.split(",") if m.strip()]
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

    print("Final-layer DP vs single-final-Pauli Q benchmark", flush=True)
    print("------------------------------------------------", flush=True)
    print(f"N={args.n_qubits} layers={args.n_steps} phi={args.phi}", flush=True)
    print(f"observable=Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"s_grid={s_grid}", flush=True)
    print(f"n_batches={args.n_batches} n_samples={args.n_samples}", flush=True)
    print(f"methods={methods} estimators={estimators}", flush=True)
    print(f"out={rows_path}", flush=True)

    print("\nWarming Numba kernels...", flush=True)
    if "gate" in methods:
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
    if "layer" in methods:
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
    for batch in range(1, args.n_batches + 1):
        if "gate" in methods and "single_final_pauli" in estimators and ("gate", "single_final_pauli", batch) not in done:
            t0 = time.perf_counter()
            p_g, s_g, a_g, wi_g, d_g = gate.evolve_many_triggered_mixture_layers(
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
                args.n_samples,
            )
            runtime = time.perf_counter() - t0
            qpm = qpm_grid_from_outputs(p_g, s_g, a_g, wi_g, d_g, s_grid)
            append_rows(rows_path, single_rows("gate", batch, qpm, runtime, np.nan, s_grid, args))
            done.add(("gate", "single_final_pauli", batch))
            del p_g, s_g, a_g, wi_g, d_g, qpm
            gc.collect()
            print(f"completed gate single batch {batch}/{args.n_batches} runtime={runtime:.3f}s", flush=True)

        if "gate" in methods and "final_layer_dp" in estimators and ("gate", "final_layer_dp", batch) not in done:
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
                args.n_samples,
                s_grid,
                s1_idx,
            )
            result = row_from_aggregate(agg, args.n_samples, time.perf_counter() - t0)
            append_rows(rows_path, final_dp_rows("gate", batch, result, s_grid, args))
            done.add(("gate", "final_layer_dp", batch))
            print(f"completed gate final-DP batch {batch}/{args.n_batches} runtime={result['runtime']:.3f}s", flush=True)

        if "layer" in methods and "single_final_pauli" in estimators and ("layer", "single_final_pauli", batch) not in done:
            t0 = time.perf_counter()
            p_l, pe_l, s_l, a_l, d_l, trig_l = layer.evolve_many_full_layer_sampling_mixture_restricted(
                init_pauli,
                trans_l,
                signed_l,
                abs_l,
                n_branches_l,
                eta_xyz,
                alpha_schedule,
                args.n_samples,
                use_restricted=True,
                use_trigger=True,
            )
            runtime = time.perf_counter() - t0
            qpm = qpm_grid_from_outputs_no_iw(p_l, s_l, a_l, d_l, s_grid)
            append_rows(
                rows_path,
                single_rows("layer", batch, qpm, runtime, float(np.mean(trig_l)), s_grid, args),
            )
            done.add(("layer", "single_final_pauli", batch))
            del p_l, pe_l, s_l, a_l, d_l, trig_l, qpm
            gc.collect()
            print(f"completed layer single batch {batch}/{args.n_batches} runtime={runtime:.3f}s", flush=True)

        if "layer" in methods and "final_layer_dp" in estimators and ("layer", "final_layer_dp", batch) not in done:
            t0 = time.perf_counter()
            agg = dp_q_curve_aggregate(
                init_pauli,
                trans_l,
                signed_l,
                abs_l,
                n_branches_l,
                eta_xyz,
                args.n_steps,
                args.n_samples,
                s_grid,
                s1_idx,
            )
            result = row_from_aggregate(agg, args.n_samples, time.perf_counter() - t0)
            append_rows(rows_path, final_dp_rows("layer", batch, result, s_grid, args))
            done.add(("layer", "final_layer_dp", batch))
            print(f"completed layer final-DP batch {batch}/{args.n_batches} runtime={result['runtime']:.3f}s", flush=True)

    rows = load_rows(rows_path)
    summary = add_differences(summarize(rows))
    write_summary(summary_path, summary)
    plot_summary(summary, plot_path)
    print_summary(summary)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
