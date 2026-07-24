import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out = dict(row)
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
                out[key] = float(out[key])
            for key in ("samples_per_batch", "batch"):
                out[key] = int(out[key])
            rows.append(out)
    return rows


def safe_corr(x, y):
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def sector_arrays(rows, method, samples_per_batch, s, sector):
    selected = [
        r
        for r in rows
        if r["method"] == method
        and r["samples_per_batch"] == samples_per_batch
        and abs(r["s"] - s) < 1.0e-12
    ]
    selected.sort(key=lambda r: r["batch"])
    if sector == "plus":
        q = np.array([r["q_plus"] for r in selected], dtype=np.float64)
        num = np.array([r["plus_num"] for r in selected], dtype=np.float64)
        den = np.array([r["plus_den"] for r in selected], dtype=np.float64)
    else:
        q = np.array([r["q_minus"] for r in selected], dtype=np.float64)
        num = np.array([r["minus_num"] for r in selected], dtype=np.float64)
        den = np.array([r["minus_den"] for r in selected], dtype=np.float64)
    batch = np.array([r["batch"] for r in selected], dtype=np.int64)
    return batch, q, num, den


def summarize_group(rows, method, samples_per_batch, s, sector):
    batch, q, num, den = sector_arrays(rows, method, samples_per_batch, s, sector)
    if len(q) == 0:
        return None

    weights = den / np.sum(den)
    pooled = np.sum(num) / np.sum(den)
    mean = np.mean(q)
    median = np.median(q)
    ess = 1.0 / np.sum(weights * weights)
    order = np.argsort(weights)[::-1]
    top1 = weights[order[:1]].sum()
    top2 = weights[order[:2]].sum()
    top4 = weights[order[:4]].sum()

    return {
        "method": method,
        "samples_per_batch": samples_per_batch,
        "s": s,
        "sector": sector,
        "n_batches": len(q),
        "q_mean": float(mean),
        "q_median": float(median),
        "q_pooled": float(pooled),
        "mean_minus_pooled_rel": float((mean - pooled) / abs(pooled)),
        "median_minus_pooled_rel": float((median - pooled) / abs(pooled)),
        "den_cv": float(np.std(den, ddof=1) / np.mean(den)),
        "den_ess_batches": float(ess),
        "top1_den_share": float(top1),
        "top2_den_share": float(top2),
        "top4_den_share": float(top4),
        "corr_logden_q": safe_corr(np.log(den), q),
        "corr_logden_logq": safe_corr(np.log(den), np.log(q)),
        "max_den_batch": int(batch[order[0]]),
        "max_den_q": float(q[order[0]]),
        "max_den_weight": float(weights[order[0]]),
    }


def write_summary(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_group(rows, samples_per_batch, s_values, out_path):
    methods = sorted({r["method"] for r in rows})
    sectors = ("plus", "minus")
    fig, axes = plt.subplots(
        len(methods) * len(sectors),
        len(s_values),
        figsize=(3.4 * len(s_values), 2.7 * len(methods) * len(sectors)),
        squeeze=False,
    )

    for mi, method in enumerate(methods):
        for si, sector in enumerate(sectors):
            row_idx = mi * len(sectors) + si
            for col_idx, s in enumerate(s_values):
                ax = axes[row_idx, col_idx]
                batch, q, _, den = sector_arrays(rows, method, samples_per_batch, s, sector)
                if len(q) == 0:
                    continue
                weights = den / np.sum(den)
                sizes = 35.0 + 850.0 * weights
                ax.scatter(den, q, s=sizes, alpha=0.75, edgecolor="black", linewidth=0.35)
                for b, x, y in zip(batch, den, q):
                    ax.annotate(str(int(b)), (x, y), fontsize=6, alpha=0.7)
                ax.axhline(np.mean(q), color="C1", linestyle="--", linewidth=1.0, label="mean")
                ax.axhline(np.median(q), color="C2", linestyle=":", linewidth=1.2, label="median")
                ax.axhline(np.sum(q * den) / np.sum(den), color="C3", linewidth=1.0, label="pooled")
                ax.set_xscale("log")
                ax.set_title(f"s={s:g}")
                if col_idx == 0:
                    ax.set_ylabel(f"{method} Q{sector[0]}")
                if row_idx == axes.shape[0] - 1:
                    ax.set_xlabel("batch denominator mass")
                ax.grid(True, alpha=0.25)
                if row_idx == 0 and col_idx == len(s_values) - 1:
                    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(f"Batch denominator mass vs Q, samples/batch={samples_per_batch:g}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_summary(rows):
    print("method samples sector s pooled mean median den_cv ESS top1 top4 corr(logD,logQ)")
    for r in rows:
        print(
            f"{r['method']:5s} {r['samples_per_batch']:>9g} {r['sector']:5s} "
            f"{r['s']:>4g} {r['q_pooled']:.6g} {r['q_mean']:.6g} "
            f"{r['q_median']:.6g} {r['den_cv']:.3g} {r['den_ess_batches']:.2f} "
            f"{r['top1_den_share']:.2%} {r['top4_den_share']:.2%} "
            f"{r['corr_logden_logq']:+.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="q_batchsize_N36_L16_Z6Z10_phi02_16b_partial_batches.csv")
    parser.add_argument("--samples-per-batch", type=int, default=10_000_000)
    parser.add_argument("--s-values", default="1,2,4,8")
    parser.add_argument("--out-prefix", default="q_batchsize_N36_L16_Z6Z10_phi02_16b_taildiag")
    args = parser.parse_args()

    rows = load_rows(args.batches)
    s_values = [float(x) for x in args.s_values.split(",") if x.strip()]
    out_prefix = Path(args.out_prefix)
    summary_path = out_prefix.with_name(f"{out_prefix.name}_summary.csv")
    plot_path = out_prefix.with_name(f"{out_prefix.name}_scatter.png")

    summary = []
    for method in sorted({r["method"] for r in rows}):
        for sector in ("plus", "minus"):
            for s in s_values:
                item = summarize_group(rows, method, args.samples_per_batch, s, sector)
                if item is not None:
                    summary.append(item)

    write_summary(summary_path, summary)
    plot_group(rows, args.samples_per_batch, s_values, plot_path)
    print_summary(summary)
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
