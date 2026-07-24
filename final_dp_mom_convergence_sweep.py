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
    return [int(float(x)) for x in text.split(",") if x.strip()]


def existing_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            done.add((row["method"], int(row["requested_total_samples"]), int(row["group"])))
    return done


def append_rows(path, rows):
    fields = [
        "method",
        "requested_total_samples",
        "actual_total_samples",
        "n_groups",
        "samples_per_group",
        "group",
        "s",
        "q_plus",
        "q_minus",
        "plus_num",
        "plus_den",
        "minus_num",
        "minus_den",
        "runtime_s",
        "triggered_frac",
    ]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def rows_from_result(method, requested_total, actual_total, n_groups, group_size, group, result, s_grid):
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "requested_total_samples": requested_total,
                "actual_total_samples": actual_total,
                "n_groups": n_groups,
                "samples_per_group": group_size,
                "group": group,
                "s": float(s),
                "q_plus": float(result["q_plus"][k]),
                "q_minus": float(result["q_minus"][k]),
                "plus_num": float(result["plus_num"][k]),
                "plus_den": float(result["plus_den"]),
                "minus_num": float(result["minus_num"][k]),
                "minus_den": float(result["minus_den"]),
                "runtime_s": float(result["runtime"]),
                "triggered_frac": float(result["triggered_frac"]),
            }
        )
    return rows


def load_rows(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in (
                "s",
                "q_plus",
                "q_minus",
                "plus_num",
                "plus_den",
                "minus_num",
                "minus_den",
                "runtime_s",
                "triggered_frac",
            ):
                parsed[key] = float(parsed[key])
            for key in (
                "requested_total_samples",
                "actual_total_samples",
                "n_groups",
                "samples_per_group",
                "group",
            ):
                parsed[key] = int(parsed[key])
            out.append(parsed)
    return out


def summarize(rows):
    summary = []
    keys = sorted({(r["method"], r["requested_total_samples"], r["s"], "plus") for r in rows})
    keys += sorted({(r["method"], r["requested_total_samples"], r["s"], "minus") for r in rows})
    for method, requested_total, s, sector in keys:
        selected = [
            r
            for r in rows
            if r["method"] == method
            and r["requested_total_samples"] == requested_total
            and abs(r["s"] - s) < 1.0e-12
        ]
        if not selected:
            continue
        q_key = "q_plus" if sector == "plus" else "q_minus"
        num_key = "plus_num" if sector == "plus" else "minus_num"
        den_key = "plus_den" if sector == "plus" else "minus_den"
        q = np.array([r[q_key] for r in selected], dtype=np.float64)
        num = np.array([r[num_key] for r in selected], dtype=np.float64)
        den = np.array([r[den_key] for r in selected], dtype=np.float64)
        weights = den / np.sum(den)
        pooled = float(np.sum(num) / np.sum(den))
        mean = float(np.mean(q))
        median = float(np.median(q))
        mad = float(np.median(np.abs(q - median)))
        std = float(np.std(q, ddof=1)) if q.size > 1 else 0.0
        ess = float(1.0 / np.sum(weights * weights))
        order = np.argsort(weights)[::-1]
        denom = abs(pooled) if pooled != 0.0 else math.nan
        summary.append(
            {
                "method": method,
                "requested_total_samples": requested_total,
                "actual_total_samples": selected[0]["actual_total_samples"],
                "samples_per_group": selected[0]["samples_per_group"],
                "n_groups": len(selected),
                "s": s,
                "sector": sector,
                "pooled_ratio": pooled,
                "mean_group_ratio": mean,
                "mom_median_group_ratio": median,
                "group_std": std,
                "group_mad": mad,
                "mean_minus_pooled_rel": float((mean - pooled) / denom),
                "mom_minus_pooled_rel": float((median - pooled) / denom),
                "den_ess_groups": ess,
                "top1_den_share": float(weights[order[:1]].sum()),
                "top4_den_share": float(weights[order[:4]].sum()),
                "mean_runtime_s": float(np.mean([r["runtime_s"] for r in selected])),
                "total_runtime_s": float(np.sum([r["runtime_s"] for r in selected])),
                "mean_triggered_frac": float(np.mean([r["triggered_frac"] for r in selected])),
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
    styles = (
        ("pooled_ratio", "pooled", "o", "C0"),
        ("mean_group_ratio", "group mean", "s", "C1"),
        ("mom_median_group_ratio", "MoM median", "^", "C3"),
    )
    for mi, method in enumerate(methods):
        for si, sector in enumerate(sectors):
            row_idx = mi * len(sectors) + si
            for col_idx, s in enumerate(s_values):
                ax = axes[row_idx, col_idx]
                selected = [
                    r
                    for r in summary
                    if r["method"] == method and r["sector"] == sector and r["s"] == s
                ]
                selected.sort(key=lambda r: r["actual_total_samples"])
                if not selected:
                    continue
                x = np.array([r["actual_total_samples"] for r in selected], dtype=np.float64)
                for key, label, marker, color in styles:
                    y = np.array([r[key] for r in selected], dtype=np.float64)
                    ax.plot(x, y, marker=marker, linewidth=1.2, color=color, label=label)
                y = np.array([r["mean_group_ratio"] for r in selected], dtype=np.float64)
                e = np.array([r["group_std"] for r in selected], dtype=np.float64)
                ax.fill_between(x, y - e, y + e, color="C1", alpha=0.10, linewidth=0)
                ax.set_xscale("log")
                ax.grid(True, alpha=0.25)
                ax.set_title(f"s={s:g}")
                if col_idx == 0:
                    ax.set_ylabel(f"{method} Q{sector[0]}")
                if row_idx == axes.shape[0] - 1:
                    ax.set_xlabel("total samples")
                if row_idx == 0 and col_idx == len(s_values) - 1:
                    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Final-DP Q convergence: pooled mean vs group mean vs MoM", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_compact(summary):
    print("method total group sector s pooled group_mean mom group_std ESS top4")
    for r in summary:
        if r["s"] not in (1.0, 8.0):
            continue
        print(
            f"{r['method']:5s} {r['actual_total_samples']:>9g} "
            f"{r['samples_per_group']:>7g} {r['sector']:5s} {r['s']:>4g} "
            f"{r['pooled_ratio']:.6g} {r['mean_group_ratio']:.6g} "
            f"{r['mom_median_group_ratio']:.6g} {r['group_std']:.2g} "
            f"{r['den_ess_groups']:.2f} {r['top4_den_share']:.1%}"
        )


def group_size_for(total_samples, n_groups):
    return max(1, int(math.ceil(total_samples / n_groups)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.2)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", default="1,2,4,8")
    parser.add_argument("--total-samples", default="1e3,3e3,1e4,3e4,1e5,3e5,1e6,3e6,1e7")
    parser.add_argument("--n-groups", type=int, default=16)
    parser.add_argument("--methods", default="gate,layer")
    parser.add_argument("--out-prefix", default="final_dp_mom_convergence_N20_L6_Z6Z10_phi02")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    total_samples = parse_int_list(args.total_samples)
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
    rows_path = out_prefix.with_name(f"{out_prefix.name}_groups.csv")
    summary_path = out_prefix.with_name(f"{out_prefix.name}_summary.csv")
    plot_path = out_prefix.with_name(f"{out_prefix.name}.png")

    print("Final-DP MoM convergence sweep", flush=True)
    print("------------------------------", flush=True)
    print(f"N={args.n_qubits} layers={args.n_steps} target=Z_{args.q1}Z_{args.q2}", flush=True)
    print(f"phi={args.phi} s_grid={s_grid}", flush=True)
    print(f"total_samples={total_samples}", flush=True)
    print(f"n_groups={args.n_groups} methods={methods}", flush=True)

    print("\nWarming kernels...", flush=True)
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

    done = existing_done(rows_path)
    for requested_total in total_samples:
        group_size = group_size_for(requested_total, args.n_groups)
        actual_total = group_size * args.n_groups
        for group in range(1, args.n_groups + 1):
            if "gate" in methods and ("gate", requested_total, group) not in done:
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
                    group_size,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, group_size, time.perf_counter() - t0)
                append_rows(
                    rows_path,
                    rows_from_result(
                        "gate",
                        requested_total,
                        actual_total,
                        args.n_groups,
                        group_size,
                        group,
                        result,
                        s_grid,
                    ),
                )
                done.add(("gate", requested_total, group))
                print(
                    f"gate total={requested_total:g} group {group}/{args.n_groups} "
                    f"size={group_size:g} runtime={result['runtime']:.3f}s",
                    flush=True,
                )

            if "layer" in methods and ("layer", requested_total, group) not in done:
                t0 = time.perf_counter()
                agg = dp_q_curve_aggregate(
                    init_pauli,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    args.n_steps,
                    group_size,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, group_size, time.perf_counter() - t0)
                append_rows(
                    rows_path,
                    rows_from_result(
                        "layer",
                        requested_total,
                        actual_total,
                        args.n_groups,
                        group_size,
                        group,
                        result,
                        s_grid,
                    ),
                )
                done.add(("layer", requested_total, group))
                print(
                    f"layer total={requested_total:g} group {group}/{args.n_groups} "
                    f"size={group_size:g} runtime={result['runtime']:.3f}s",
                    flush=True,
                )

    rows = load_rows(rows_path)
    summary = summarize(rows)
    write_summary(summary_path, summary)
    plot_summary(summary, plot_path)
    print_compact(summary)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
