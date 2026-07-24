# Pauli-Path ZNE

Research code for Pauli-path sampling and zero-noise extrapolation experiments
on 1D Heisenberg brickwall circuits.

The current workflow focuses on estimating the sector damping functions

$$
Q^\pm(s)
=
\frac{\sum_{\gamma\in\pm}|A_\gamma|D_\gamma^s}
{\sum_{\gamma\in\pm}|A_\gamma|}
$$

using Monte Carlo prefix sampling plus an exact dynamic-programming marginal over
the final circuit layer. Deterministic MPS simulations are included for
comparison against noisy observable curves \(O(s)\).

## What Is New

The newest code adds **final-layer DP** for both sampling paths:

- **Gate-wise prefix sampler**: samples gates one by one, with trigger-based
  restricted sampling once the support can still reach the target.
- **Layer-wise DP prefix sampler**: samples full Trotter layers using blocked
  light-cone/full-layer transition tables.
- **Final-layer DP marginal**: instead of sampling one terminal Pauli in the
  final layer, it enumerates all nonzero valid final-layer continuations
  conditionally on the sampled prefix.

This removes final-layer branch noise. Any remaining rare-event behavior comes
from the sampled prefix distribution, not from final-layer enumeration.

## Repository Guide

Core final-layer DP files:

```text
Pauli_path_Heis_full_layer_sampling_restricted.py
Pauli_path_Heis_mixture_trigger.py
benchmark_q_curve_convergence.py
final_dp_mom_convergence_sweep.py
```

MPS comparison:

```text
pauli_mps_solver.py
pauli_mps_solver_user_guide.md
compute_mps_zizj_vs_s.py
sweep_mps_zizj_chi_vs_s.py
```

Diagnostics:

```text
diagnose_q_batch_heavytails.py
compare_final_dp_vs_single_pauli_q.py
sweep_final_dp_vs_single_convergence.py
benchmark_q_batchsize_convergence.py
qpm_numba_utils.py
```

Tutorial notebook:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

## Basic Setup

The scripts are plain Python and expect NumPy, Numba, and Matplotlib. The MPS
solver also uses SciPy-style linear algebra dependencies available in a standard
scientific Python environment.

From the repo root:

```bash
python -m py_compile \
  Pauli_path_Heis_full_layer_sampling_restricted.py \
  Pauli_path_Heis_mixture_trigger.py \
  benchmark_q_curve_convergence.py \
  final_dp_mom_convergence_sweep.py \
  pauli_mps_solver.py
```

Generated CSV/PNG/cache files are ignored by default.

## Quick Tutorial

### 1. Estimate \(Q^\pm(s)\) With Final-Layer DP

This command runs both gate-wise and layer-wise samplers on a small example,
splits each total sample count into 16 groups, and compares pooled, group-mean,
and median-of-means summaries.

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

The important output columns are:

```text
pooled_ratio
mean_group_ratio
mom_median_group_ratio
den_ess_groups
top4_den_share
```

Interpretation:

- `pooled_ratio` is the direct ratio-of-sums estimator.
- `mean_group_ratio` averages the group-level ratios.
- `mom_median_group_ratio` is the median group-level ratio.
- `den_ess_groups` and `top4_den_share` diagnose rare denominator domination.

If pooled, group mean, and MoM agree, the group size is likely large enough for
that system. If pooled sits far below mean/MoM, a few high-denominator prefix
groups are dominating the ratio.

### 2. Compare Against MPS \(O(s)\)

The MPS solver computes the noisy observable curve, not \(Q^\pm(s)\) directly.
Use it as a reference for the observable:

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

For current sampling comparisons, use:

```text
noise_model     independent
noise_placement layer
```

Avoid `legacy_sum` unless you are intentionally reproducing old results.

### 3. Diagnose Heavy-Tailed Prefix Weights

After a grouped Q sweep, inspect whether high-denominator groups are pulling the
pooled estimator:

```bash
MPLCONFIGDIR=.mplconfig python diagnose_q_batch_heavytails.py \
  --batches final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --samples-per-batch 62500 \
  --out-prefix taildiag_N20_L10_Z6Z10_phi02_1e6
```

For 16 groups, `samples-per-batch = total_samples / 16`.

The key diagnostic is whether large-denominator groups have systematically lower
group ratios \(Q_g^\pm(s)\). If so, pooled approaches from below while group
mean/MoM approach from above.

## Notebook Walkthrough

For an executable, annotated walkthrough, open:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

The notebook includes:

- MPS reference calculation for \(O(s)\)
- gate-wise and layer-wise final-DP Q sampling
- pooled versus MoM summaries
- production command examples
- interpretation checklist

## Notes On Estimators

For group \(g\),

$$
Q_g(s)=\frac{N_g(s)}{D_g}.
$$

The pooled estimator is

$$
Q_{\rm pooled}(s)
=
\frac{\sum_g N_g(s)}{\sum_g D_g},
$$

while MoM uses

$$
Q_{\rm MoM}(s)
=
\operatorname{median}_g Q_g(s).
$$

Pooled is the direct ratio-of-sums estimator. MoM is a finite-sample robustness
diagnostic and can be useful for extrapolation fits when denominator-heavy rare
events make pooled estimates unstable. If groups are too small, MoM can suppress
real rare-tail contributions, so group-size convergence should always be checked.

## Development Notes

- Keep generated data and plots out of commits unless a benchmark artifact is
  intentionally being published.
- Use `MPLCONFIGDIR=.mplconfig` when running plotting scripts in restricted
  environments.
- Prefer the resumable command-line scripts for long runs.
- Check MPS bond dimensions and discarded weights before treating MPS data as an
  exact reference.
