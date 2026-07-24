import argparse
import csv
import math
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


def parse_int_list(text):
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        out.append(int(float(raw)))
    return out


def existing_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["method"], int(row["samples_per_batch"]), int(row["batch"])))
    return done


def append_batch_rows(path, rows):
    fieldnames = [
        "method",
        "samples_per_batch",
        "batch",
        "s",
        "target_mean_s1",
        "target_se_s1",
        "q_plus",
        "q_minus",
        "plus_num",
        "plus_den",
        "minus_num",
        "minus_den",
        "runtime_s",
        "triggered_frac",
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


def rows_from_run(method, samples_per_batch, batch, result, s_grid, args):
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "samples_per_batch": int(samples_per_batch),
                "batch": int(batch),
                "s": float(s),
                "target_mean_s1": result["mean"],
                "target_se_s1": result["se"],
                "q_plus": float(result["q_plus"][k]),
                "q_minus": float(result["q_minus"][k]),
                "plus_num": float(result["plus_num"][k]),
                "plus_den": float(result["plus_den"]),
                "minus_num": float(result["minus_num"][k]),
                "minus_den": float(result["minus_den"]),
                "runtime_s": float(result["runtime"]),
                "triggered_frac": float(result["triggered_frac"]),
                "n_qubits": int(args.n_qubits),
                "n_steps": int(args.n_steps),
                "phi": float(args.phi),
                "lambda_base": float(args.lambda_base),
                "q1": int(args.q1),
                "q2": int(args.q2),
            }
        )
    return rows


def load_batch_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            r = dict(row)
            for key in (
                "s",
                "target_mean_s1",
                "target_se_s1",
                "q_plus",
                "q_minus",
                "plus_num",
                "plus_den",
                "minus_num",
                "minus_den",
                "runtime_s",
                "triggered_frac",
                "phi",
                "lambda_base",
            ):
                r[key] = float(r[key])
            for key in ("samples_per_batch", "batch", "n_qubits", "n_steps", "q1", "q2"):
                r[key] = int(r[key])
            rows.append(r)
    return rows


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], row["samples_per_batch"], row["s"], "plus")
        grouped.setdefault(key, []).append(row)
        key = (row["method"], row["samples_per_batch"], row["s"], "minus")
        grouped.setdefault(key, []).append(row)

    summary = []
    for (method, samples_per_batch, s, sector), group_rows in sorted(grouped.items()):
        q_key = "q_plus" if sector == "plus" else "q_minus"
        num_key = "plus_num" if sector == "plus" else "minus_num"
        den_key = "plus_den" if sector == "plus" else "minus_den"
        values = np.array([r[q_key] for r in group_rows], dtype=np.float64)
        nums = np.array([r[num_key] for r in group_rows], dtype=np.float64)
        dens = np.array([r[den_key] for r in group_rows], dtype=np.float64)
        runtime_by_batch = {}
        triggered_by_batch = {}
        for r in group_rows:
            runtime_by_batch[r["batch"]] = r["runtime_s"]
            triggered_by_batch[r["batch"]] = r["triggered_frac"]

        pooled = float(np.sum(nums) / np.sum(dens))
        mean = float(np.mean(values))
        median = float(np.median(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mad = float(np.median(np.abs(values - median)))
        denom = abs(pooled) if pooled != 0.0 else math.nan
        summary.append(
            {
                "method": method,
                "samples_per_batch": samples_per_batch,
                "s": s,
                "sector": sector,
                "n_batches": len(values),
                "mean_of_batch_ratios": mean,
                "median_of_batch_ratios": median,
                "pooled_ratio_of_sums": pooled,
                "batch_std": std,
                "batch_mad": mad,
                "mean_minus_pooled_rel": (mean - pooled) / denom,
                "median_minus_pooled_rel": (median - pooled) / denom,
                "mean_runtime_s": float(np.mean(list(runtime_by_batch.values()))),
                "total_runtime_s": float(np.sum(list(runtime_by_batch.values()))),
                "mean_triggered_frac": float(np.mean(list(triggered_by_batch.values()))),
            }
        )
    return summary


def write_summary(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows, out_path):
    methods = sorted({r["method"] for r in summary_rows})
    s_values = sorted({r["s"] for r in summary_rows})
    sectors = ["plus", "minus"]
    fig, axes = plt.subplots(
        len(methods) * len(sectors),
        len(s_values),
        figsize=(3.4 * len(s_values), 2.7 * len(methods) * len(sectors)),
        squeeze=False,
        sharex=True,
    )

    for mi, method in enumerate(methods):
        for si, sector in enumerate(sectors):
            row_idx = mi * len(sectors) + si
            for col_idx, s in enumerate(s_values):
                ax = axes[row_idx, col_idx]
                selected = [
                    r
                    for r in summary_rows
                    if r["method"] == method and r["sector"] == sector and r["s"] == s
                ]
                selected.sort(key=lambda r: r["samples_per_batch"])
                x = np.array([r["samples_per_batch"] for r in selected], dtype=np.float64)
                for key, label, marker in (
                    ("pooled_ratio_of_sums", "pooled", "o"),
                    ("mean_of_batch_ratios", "mean", "s"),
                    ("median_of_batch_ratios", "median", "^"),
                ):
                    y = np.array([r[key] for r in selected], dtype=np.float64)
                    ax.plot(x, y, marker=marker, linewidth=1.4, label=label)
                if selected:
                    y = np.array([r["mean_of_batch_ratios"] for r in selected], dtype=np.float64)
                    e = np.array([r["batch_std"] for r in selected], dtype=np.float64)
                    ax.fill_between(x, y - e, y + e, alpha=0.12, linewidth=0)
                ax.set_xscale("log")
                ax.set_title(f"s={s:g}")
                if col_idx == 0:
                    ax.set_ylabel(f"{method} Q{sector[0]}")
                ax.grid(True, alpha=0.25)
                if row_idx == axes.shape[0] - 1:
                    ax.set_xlabel("samples/batch")
                if row_idx == 0 and col_idx == len(s_values) - 1:
                    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Q convergence: batch mean vs batch median vs pooled ratio", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_compact_summary(summary_rows):
    print("\nConvergence summary", flush=True)
    print("-------------------", flush=True)
    for method in sorted({r["method"] for r in summary_rows}):
        print(f"\n{method}", flush=True)
        selected = [r for r in summary_rows if r["method"] == method]
        for samples_per_batch in sorted({r["samples_per_batch"] for r in selected}):
            sub = [r for r in selected if r["samples_per_batch"] == samples_per_batch]
            rt = np.mean([r["mean_runtime_s"] for r in sub])
            print(f"  samples/batch={samples_per_batch:g} mean runtime={rt:.3f}s", flush=True)
            for sector in ("plus", "minus"):
                line = [r for r in sub if r["sector"] == sector]
                line.sort(key=lambda r: r["s"])
                pieces = []
                for r in line:
                    pieces.append(
                        f"s={r['s']:g}: mean {r['mean_of_batch_ratios']:.5g}, "
                        f"med {r['median_of_batch_ratios']:.5g}, "
                        f"pooled {r['pooled_ratio_of_sums']:.5g}, "
                        f"std {r['batch_std']:.2g}"
                    )
                print(f"    Q{sector[0]}  " + " | ".join(pieces), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=36)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", default="1,2,4,8")
    parser.add_argument("--samples-per-batch", default="1e5,1e6,1e7,1e8")
    parser.add_argument("--n-batches", type=int, default=16)
    parser.add_argument("--methods", default="gate,layer")
    parser.add_argument("--out-prefix", default="q_batchsize_N36_L16_Z6Z10_phi02")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    samples_list = parse_int_list(args.samples_per_batch)
    s_grid = parse_s_grid(args.s_grid)
    s1_idx = find_s1_index(s_grid)

    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[args.q1] = 3
    init_pauli[args.q2] = 3

    lam_xyz = np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)

    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    out_prefix = Path(args.out_prefix)
    batch_path = out_prefix.with_name(f"{out_prefix.name}_batches.csv")
    summary_path = out_prefix.with_name(f"{out_prefix.name}_summary.csv")
    plot_path = out_prefix.with_name(f"{out_prefix.name}_convergence.png")

    print("Q batch-size convergence", flush=True)
    print("------------------------", flush=True)
    print(f"N={args.n_qubits} layers={args.n_steps} phi={args.phi}", flush=True)
    print(f"observable=Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"s_grid={s_grid}", flush=True)
    print(f"samples_per_batch={samples_list}", flush=True)
    print(f"n_batches={args.n_batches}", flush=True)
    print(f"methods={methods}", flush=True)
    print(f"output={batch_path}", flush=True)

    print("\nWarming Numba kernels...", flush=True)
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

    done = existing_done(batch_path)
    for samples_per_batch in samples_list:
        for batch in range(1, args.n_batches + 1):
            if "gate" in methods and ("gate", samples_per_batch, batch) not in done:
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
                    samples_per_batch,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, samples_per_batch, time.perf_counter() - t0)
                append_batch_rows(
                    batch_path,
                    rows_from_run("gate", samples_per_batch, batch, result, s_grid, args),
                )
                done.add(("gate", samples_per_batch, batch))
                print(
                    f"completed gate samples/batch={samples_per_batch:g} "
                    f"batch {batch}/{args.n_batches} runtime={result['runtime']:.3f}s",
                    flush=True,
                )

            if "layer" in methods and ("layer", samples_per_batch, batch) not in done:
                t0 = time.perf_counter()
                agg = dp_q_curve_aggregate(
                    init_pauli,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    args.n_steps,
                    samples_per_batch,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, samples_per_batch, time.perf_counter() - t0)
                append_batch_rows(
                    batch_path,
                    rows_from_run("layer", samples_per_batch, batch, result, s_grid, args),
                )
                done.add(("layer", samples_per_batch, batch))
                print(
                    f"completed layer samples/batch={samples_per_batch:g} "
                    f"batch {batch}/{args.n_batches} runtime={result['runtime']:.3f}s",
                    flush=True,
                )

    all_rows = load_batch_rows(batch_path)
    summary_rows = summarize_rows(all_rows)
    write_summary(summary_path, summary_rows)
    plot_summary(summary_rows, plot_path)
    print_compact_summary(summary_rows)
    print(f"\nwrote {summary_path}", flush=True)
    print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
