# Pauli-Path ZNE

Research code for Pauli-path sampling and zero-noise extrapolation experiments
on 1D Heisenberg brickwall circuits.

The current workflow estimates the sector damping functions:

```math
Q^\pm(s)
=
\frac{\sum_{\gamma \in \pm} |A_\gamma| D_\gamma^s}
{\sum_{\gamma \in \pm} |A_\gamma|}.
```

Here `+` and `-` denote the signed Pauli-path sectors, `A_gamma` is the
noiseless path amplitude, and `D_gamma^s` is the noise damping factor at noise
scale `s`.

Deterministic MPS simulations are included for comparison against the noisy
observable curve:

```math
O(s).
```

## What Is New

The newest code adds final-layer dynamic programming (DP) to both sampling
paths.

The gate-wise prefix sampler samples gates one by one, with trigger-based
restricted sampling once the support can still reach the target.

The layer-wise DP prefix sampler samples full Trotter layers using blocked
light-cone/full-layer transition tables.

The final-layer DP marginal enumerates all nonzero valid final-layer
continuations conditioned on the sampled prefix. This removes final-layer branch
noise. Remaining rare-event behavior comes from the sampled prefix distribution,
not from the final-layer enumeration.

## Repository Guide

Core final-layer DP files:

```text
Pauli_path_Heis_full_layer_sampling_restricted.py
Pauli_path_Heis_mixture_trigger.py
benchmark_q_curve_convergence.py
final_dp_mom_convergence_sweep.py
qpm_numba_utils.py
```

MPS comparison:

```text
pauli_mps_solver.py
pauli_mps_solver_user_guide.md
compute_mps_zizj_vs_s.py
sweep_mps_zizj_chi_vs_s.py
```

Diagnostics and convergence checks:

```text
diagnose_q_batch_heavytails.py
compare_final_dp_vs_single_pauli_q.py
sweep_final_dp_vs_single_convergence.py
benchmark_q_batchsize_convergence.py
```

Fitting and ZNE comparison:

```text
fit_zne_compare.py
```

Notebook tutorial:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

## Basic Setup

The scripts are plain Python and expect NumPy, Numba, and Matplotlib. The MPS
solver uses the scientific Python stack already used elsewhere in this repo.

From the repository root:

```bash
python -m py_compile \
  Pauli_path_Heis_full_layer_sampling_restricted.py \
  Pauli_path_Heis_mixture_trigger.py \
  benchmark_q_curve_convergence.py \
  final_dp_mom_convergence_sweep.py \
  pauli_mps_solver.py
```

Generated CSV, PNG, cache, and notebook checkpoint files are ignored by default.

## Notebook Usage

The recommended entry point is the notebook:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

The examples below mirror the notebook style: import the repo modules, set a
small parameter block, compute an MPS reference curve, then run the gate-wise and
layer-wise final-DP samplers.

### 1. Imports

If the notebook is opened from `notebooks/`, add the repo root to `sys.path`
before importing local modules.

```python
from pathlib import Path
import os
import sys
import time

import numpy as np

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pauli_mps_solver.py").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

import Pauli_path_Heis_mixture_trigger as gate
import Pauli_path_Heis_full_layer_sampling_restricted as layer

from benchmark_q_curve_convergence import (
    dp_q_curve_aggregate,
    find_s1_index,
    gate_q_curve_aggregate,
    product_eta_from_lambda,
    row_from_aggregate,
)
from final_dp_mom_convergence_sweep import summarize as summarize_q_groups
from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz
```

### 2. Parameters

This is intentionally small enough for notebook testing. Increase
`TOTAL_SAMPLES`, `N_STEPS`, and `CHI_MAX` for production runs.

```python
N_QUBITS = 20
N_STEPS = 6
Q1, Q2 = 6, 10
PHI = 0.2
LAMBDA_BASE = 1.0e-3
S_GRID = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

TOTAL_SAMPLES = [10_000, 100_000, 1_000_000]
N_GROUPS = 16

CHI_MAX = 128
CUTOFF = 1.0e-10
SVD_METHOD = "auto"
```

### 3. MPS Reference for the Observable

The MPS solver computes the noisy observable curve `O(s)`. It does not compute
`Q_plus(s)` or `Q_minus(s)` directly.

```python
def run_mps_curve(
    n_qubits=N_QUBITS,
    n_steps=N_STEPS,
    q1=Q1,
    q2=Q2,
    phi=PHI,
    lambda_base=LAMBDA_BASE,
    s_values=(0.0, 1.0, 2.0, 4.0, 8.0),
    chi_max=CHI_MAX,
):
    target = pauli_zz(n_qubits, q1, q2)
    rows = []
    for s in s_values:
        lam = np.full((n_qubits, 3), lambda_base * s, dtype=np.float64)
        t0 = time.perf_counter()
        value, _, info = evolve_observable_backward_mps(
            target,
            n_qubits=n_qubits,
            phi=phi,
            lam_xyz=lam,
            n_steps=n_steps,
            chi_max=chi_max,
            cutoff=CUTOFF,
            svd_method=SVD_METHOD,
            noise_model="independent",
            noise_placement="layer",
            use_lightcone=True,
            return_mps=True,
        )
        rows.append(
            {
                "s": float(s),
                "O_mps": float(value),
                "runtime_s": time.perf_counter() - t0,
                "max_bond": int(np.max(info["bond_dims"])),
                "max_discarded_weight": float(
                    np.max(info["discarded_by_backward_step"])
                ),
            }
        )
    return rows


mps_rows = run_mps_curve()
mps_rows
```

Plot the MPS curve:

```python
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot([r["s"] for r in mps_rows], [r["O_mps"] for r in mps_rows], marker="o")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xlabel("s")
ax.set_ylabel("MPS O(s)")
ax.grid(True, alpha=0.25)
```

### 4. Final-Layer DP Sampling for Q Curves

This cell runs both samplers in groups. Each group produces numerator and
denominator sums for `Q_plus(s)` and `Q_minus(s)` over the full `S_GRID`.

```python
def group_size_for(total_samples, n_groups):
    return max(1, int(np.ceil(total_samples / n_groups)))


def q_result_rows(
    method,
    requested_total,
    actual_total,
    n_groups,
    group_size,
    group_index,
    result,
    s_grid,
):
    rows = []
    for k, s in enumerate(s_grid):
        rows.append(
            {
                "method": method,
                "requested_total_samples": int(requested_total),
                "actual_total_samples": int(actual_total),
                "n_groups": int(n_groups),
                "samples_per_group": int(group_size),
                "group": int(group_index),
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
```

```python
def run_final_dp_q_groups(
    total_samples_list=TOTAL_SAMPLES,
    n_groups=N_GROUPS,
    methods=("gate", "layer"),
    n_qubits=N_QUBITS,
    n_steps=N_STEPS,
    q1=Q1,
    q2=Q2,
    phi=PHI,
    lambda_base=LAMBDA_BASE,
    s_grid=S_GRID,
):
    s1_idx = find_s1_index(s_grid)

    init_pauli = np.zeros(n_qubits, dtype=np.int8)
    init_pauli[q1] = 3
    init_pauli[q2] = 3

    lam_xyz = np.full((n_qubits, 3), lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)

    even_gates, odd_gates = gate.make_even_odd_layers(n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(phi)

    group_rows = []
    for requested_total in total_samples_list:
        group_size = group_size_for(requested_total, n_groups)
        actual_total = group_size * n_groups
        for group_index in range(1, n_groups + 1):
            if "gate" in methods:
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
                    n_steps,
                    group_size,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, group_size, time.perf_counter() - t0)
                group_rows.extend(
                    q_result_rows(
                        "gate",
                        requested_total,
                        actual_total,
                        n_groups,
                        group_size,
                        group_index,
                        result,
                        s_grid,
                    )
                )

            if "layer" in methods:
                t0 = time.perf_counter()
                agg = dp_q_curve_aggregate(
                    init_pauli,
                    trans_l,
                    signed_l,
                    abs_l,
                    n_branches_l,
                    eta_xyz,
                    n_steps,
                    group_size,
                    s_grid,
                    s1_idx,
                )
                result = row_from_aggregate(agg, group_size, time.perf_counter() - t0)
                group_rows.extend(
                    q_result_rows(
                        "layer",
                        requested_total,
                        actual_total,
                        n_groups,
                        group_size,
                        group_index,
                        result,
                        s_grid,
                    )
                )
    return group_rows


q_group_rows = run_final_dp_q_groups()
q_summary = summarize_q_groups(q_group_rows)
q_summary[:4]
```

### 5. Plot Pooled vs Median-of-Means

`summarize_q_groups` reports the direct ratio-of-sums estimator, the mean of
group ratios, and the median of group ratios.

```python
def plot_q_summary(q_summary, s_to_show=(1.0, 8.0)):
    methods = sorted({r["method"] for r in q_summary})
    sectors = ("plus", "minus")
    fig, axes = plt.subplots(len(methods), len(sectors), figsize=(10, 5.8), sharex=True)
    if len(methods) == 1:
        axes = np.array([axes])

    for i, method in enumerate(methods):
        for j, sector in enumerate(sectors):
            ax = axes[i, j]
            for s in s_to_show:
                rows = [
                    r
                    for r in q_summary
                    if r["method"] == method and r["sector"] == sector and r["s"] == s
                ]
                rows.sort(key=lambda r: r["actual_total_samples"])
                x = np.array([r["actual_total_samples"] for r in rows], dtype=float)
                ax.plot(
                    x,
                    [r["pooled_ratio"] for r in rows],
                    marker="o",
                    label=f"pooled, s={s:g}",
                )
                ax.plot(
                    x,
                    [r["mom_median_group_ratio"] for r in rows],
                    marker="^",
                    linestyle="--",
                    label=f"MoM, s={s:g}",
                )
            ax.set_xscale("log")
            ax.set_title(f"{method} Q{sector[0]}")
            ax.set_xlabel("total samples")
            ax.grid(True, alpha=0.25)
            if j == 0:
                ax.set_ylabel("Q estimate")
            ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


plot_q_summary(q_summary)
```

## Estimator Notes

For group `g`, define:

```math
Q_g(s) = \frac{N_g(s)}{D_g}.
```

The pooled estimator is:

```math
Q_{\mathrm{pooled}}(s)
=
\frac{\sum_g N_g(s)}{\sum_g D_g}.
```

The median-of-means style summary used here is:

```math
Q_{\mathrm{MoM}}(s)
=
\mathrm{median}_g Q_g(s).
```

`Q_pooled(s)` is the direct ratio-of-sums estimator. `Q_MoM(s)` is a
finite-sample robustness diagnostic and can be useful for extrapolation fits
when denominator-heavy rare events make the pooled estimates unstable. If groups
are too small, MoM can suppress real rare-tail contributions, so group-size
convergence should always be checked.

Useful diagnostic columns:

```text
pooled_ratio
mean_group_ratio
mom_median_group_ratio
den_ess_groups
top4_den_share
```

If pooled, group mean, and MoM agree, the group size is likely large enough for
that system. If pooled sits far below mean/MoM, a few high-denominator prefix
groups may be dominating the ratio.

## Command-Line Runs

The notebook import workflow is best for exploration. For long production
runs, use the resumable command-line scripts.

Final-layer DP Q convergence:

```bash
MPLCONFIGDIR=.mplconfig python final_dp_mom_convergence_sweep.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-grid 1,2,4,8 \
  --total-samples 1e3,3e3,1e4,3e4,1e5,3e5,1e6 \
  --n-groups 16 \
  --methods gate,layer \
  --out-prefix final_dp_mom_convergence_N20_L10_Z6Z10_phi02
```

MPS observable reference:

```bash
python sweep_mps_zizj_chi_vs_s.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-values 0,1,2,4,8 \
  --chi-values 128,256,350 \
  --noise-model independent \
  --noise-placement layer \
  --out-csv mps_N20_L10_Z6Z10_phi02_chi_sweep.csv \
  --resume
```

Q-assisted ZNE versus dual-exponential ZNE:

The fitting driver combines:

```text
Q data:   final_dp_mom_convergence_*_groups.csv
O data:   MPS or hardware observable CSV with s and mps_value/O_mps/observable/value
models:   dual_exp, gate_qmom, layer_qmom
```

Generate grouped Q data first:

```bash
MPLCONFIGDIR=.mplconfig python final_dp_mom_convergence_sweep.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-grid 1,2,4,8 \
  --total-samples 1e6,1e7 \
  --n-groups 16 \
  --methods gate,layer \
  --out-prefix final_dp_mom_convergence_N20_L10_Z6Z10_phi02
```

Generate the matching MPS observable curve:

```bash
python sweep_mps_zizj_chi_vs_s.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-values 0,1,2,4,8 \
  --chi-values 350 \
  --noise-model independent \
  --noise-placement layer \
  --out-csv mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --resume
```

Then compare Q-assisted fits against dual-exponential ZNE:

```bash
MPLCONFIGDIR=.mplconfig python fit_zne_compare.py \
  --q-groups final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --observables mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --default-case case0 \
  --s-grid 1,2,4,8 \
  --s-fit-q 1,2,4,8 \
  --s-fit-o 1,2,4,8 \
  --shots 100000 \
  --n-trials 500 \
  --q-summary pointwise_mom \
  --out-prefix zne_fit_N20_L10_Z6Z10_phi02_100k
```

`--shots` simulates noisy hardware-style observable points from exact/MPS
values. For real measured observable data, put the measured values in the
observable CSV and use:

```bash
MPLCONFIGDIR=.mplconfig python fit_zne_compare.py \
  --q-groups final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --observables hardware_observables_N20_L10_Z6Z10_phi02.csv \
  --s-grid 1,2,4,8 \
  --s-fit-q 1,2,4,8 \
  --s-fit-o 1,2,4,8 \
  --use-observed \
  --q-summary pointwise_mom \
  --out-prefix zne_fit_hardware_N20_L10_Z6Z10_phi02
```

For hybrid data, where lower noise points are hardware/noisy and larger `s`
points are exact classical values, use `--noisy-s`:

```bash
MPLCONFIGDIR=.mplconfig python fit_zne_compare.py \
  --q-groups final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --observables mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --s-grid 1,2,4,8 \
  --s-fit-q 1,2,4,8 \
  --s-fit-o 1,2,4,8 \
  --noisy-s 1,2 \
  --shots 100000 \
  --exact-sigma 1e-6 \
  --q-summary pointwise_mom \
  --out-prefix zne_fit_hybrid_s12_100k_N20_L10_Z6Z10_phi02
```

This writes:

```text
*_q_fits.csv
*_trials.csv
*_summary.csv
*_curves.csv
*_hist.png
*_curves.png
```

The compared models are `dual_exp`, `gate_qmom`, and `layer_qmom`. Use
`--q-total-samples` to choose a particular sample-count block from a grouped Q
CSV; by default the fitter uses the largest `requested_total_samples` present.

For current sampling comparisons, use:

```text
noise_model     independent
noise_placement layer
```

Avoid `legacy_sum` unless intentionally reproducing old results.

## Development Notes

Keep generated data and plots out of commits unless a benchmark artifact is
intentionally being published.

Use `MPLCONFIGDIR=.mplconfig` when running plotting scripts in restricted
environments.

Prefer the resumable command-line scripts for long runs.

Check MPS bond dimensions and discarded weights before treating MPS data as an
exact reference.
