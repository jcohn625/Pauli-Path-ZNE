# Pauli-Path ZNE

Research code for Pauli-path sampling and zero-noise extrapolation (ZNE) on
1D Heisenberg brickwall circuits.

The practical goal of this repo is to estimate a zero-noise target observable,
such as `Z_6 Z_10`, by combining:

```text
Q data       classical Pauli-path sampling of Q+(s) and Q-(s)
O data       noisy hardware, simulated shot-noisy, or MPS observable data O(s)
baseline     ordinary dual-exponential ZNE fit for comparison
```

The recommended entry point is the notebook:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

The main final fitting function is:

```python
from fit_zne_compare import run_zne_fit_compare
```

## Motivation

Standard ZNE fits the measured observable curve `O(s)` directly as a function of
noise scale `s`. For the sign-changing and rare-event regimes we are studying,
direct fits such as a dual exponential can be unstable because the same few
noisy points determine both the decay rates and the zero-noise intercept.

The Q-assisted strategy separates the problem into two pieces.

First, use Pauli-path sampling to learn the sector damping curves:

```math
Q^\pm(s)
=
\frac{\sum_{\gamma \in \pm} |A_\gamma|D_\gamma^s}
{\sum_{\gamma \in \pm} |A_\gamma|}.
```

Second, use the measured or simulated observable data only to fit two linear
coefficients. This makes the final noisy-data fit much better conditioned.

## High-Level Workflow

1. Choose a target problem.

```text
N          number of qubits
L          number of Heisenberg layers
target     for example Z_6 Z_10
phi        Heisenberg circuit angle
lambda     base noise strength
s_grid     noise scale points, often 1,2,4,8
```

2. Generate or load an observable curve `O(s)`.

For benchmarks, use the MPS solver with the same independent/product noise
model used by the sampler. For real experiments, load hardware estimates of
`O(s)` and their error bars if available.

3. Estimate `Q+(s)` and `Q-(s)`.

Use final-layer DP sampling with both samplers:

```text
gate_qmom     gate-wise triggered prefix sampler plus final-layer DP
layer_qmom    layer-wise DP prefix sampler plus final-layer DP
```

The final-layer DP exactly sums all nonzero final-layer continuations
conditioned on the sampled prefix. This removes final-layer branch noise. The
remaining hard part is prefix rare-event variance.

4. Check Q convergence.

Compare pooled, group mean, and median-of-means (MoM). If they disagree, the Q
data is still prefix-tail dominated.

5. Fit Q modes.

The fitter converts `Q+` and `Q-` into average and difference modes, then fits a
smooth log ansatz.

6. Fit `O(s)`.

The final Q-assisted ZNE model is linear in two coefficients:

```math
O_{\mathrm{fit}}(s)
=
u Q_{\mathrm{ave}}(s)
+
v Q_{\mathrm{diff}}(s).
```

The zero-noise estimate is:

```math
O_{\mathrm{fit}}(0) = u.
```

In output files, read this as `u_O0`.

## Repository Map

Core samplers:

```text
Pauli_path_Heis_mixture_trigger.py
Pauli_path_Heis_full_layer_sampling_restricted.py
benchmark_q_curve_convergence.py
final_dp_mom_convergence_sweep.py
qpm_numba_utils.py
```

MPS comparison:

```text
pauli_mps_solver.py
compute_mps_zizj_vs_s.py
sweep_mps_zizj_chi_vs_s.py
pauli_mps_solver_user_guide.md
```

Final fitting:

```text
fit_zne_compare.py
```

Diagnostics:

```text
diagnose_q_batch_heavytails.py
compare_final_dp_vs_single_pauli_q.py
sweep_final_dp_vs_single_convergence.py
benchmark_q_batchsize_convergence.py
```

Notebook tutorial:

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

## Setup

From the repository root:

```bash
python -m py_compile \
  Pauli_path_Heis_full_layer_sampling_restricted.py \
  Pauli_path_Heis_mixture_trigger.py \
  benchmark_q_curve_convergence.py \
  final_dp_mom_convergence_sweep.py \
  fit_zne_compare.py \
  pauli_mps_solver.py
```

Generated CSV, PNG, cache, and notebook checkpoint files are ignored by default.

For plotting in restricted environments, set:

```bash
export MPLCONFIGDIR=.mplconfig
```

## Notebook User Manual

Use `notebooks/Final_Layer_DP_Usage.ipynb` as the canonical notebook workflow.
The README below mirrors the notebook sections and explains what each step is
for.

### 1. Import The Repo

If the notebook is opened from `notebooks/`, add the repo root to `sys.path`
before importing local modules.

```python
from pathlib import Path
import os
import sys

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "pauli_mps_solver.py").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
```

The key notebook imports are:

```python
from fit_zne_compare import run_zne_fit_compare
from final_dp_mom_convergence_sweep import summarize as summarize_q_groups
from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz
```

The notebook also imports the gate-wise and layer-wise sampler modules directly.

### 2. Choose The Target Problem

The main target parameters are:

```python
N_QUBITS = 20
N_STEPS = 6
Q1, Q2 = 6, 10
PHI = 0.2
LAMBDA_BASE = 1.0e-3
S_GRID = [1.0, 2.0, 4.0, 8.0]
```

For production runs, increase `N_STEPS`, `TOTAL_SAMPLES`, and the MPS bond
dimension `CHI_MAX`. Always keep the MPS noise convention aligned with the
sampler:

```text
noise_model      independent
noise_placement  layer
```

Avoid `legacy_sum` unless intentionally reproducing older results.

### 3. Compute Or Load O(s)

For benchmarks, compute an MPS reference curve at:

```text
s = 0,1,2,4,8
```

The `s=0` value is the true zero-noise benchmark when MPS is converged. The
positive `s` values are the noisy curve used by the ZNE fit.

For real hardware data, create a CSV with:

```text
s, observable
```

or:

```text
s, observable, sigma
```

Accepted observable value columns are:

```text
mps_value
O_mps
observable
value
mean
obs
y
```

Accepted error columns are:

```text
sigma
se
stderr
std_error
```

### 4. Estimate Q+(s) And Q-(s)

The notebook runs grouped Q sampling. Each group produces:

```text
q_plus(s)
q_minus(s)
plus_num(s), plus_den
minus_num(s), minus_den
```

The grouped data lets us compare:

```text
pooled ratio       sum numerator / sum denominator
group mean         average of group ratios
pointwise MoM      median of group ratios
```

The default final fit uses `q_summary="pointwise_mom"` because the finite-sample
prefix distribution can be heavy-tailed.

### 5. Diagnose Q Convergence

Use the notebook plots and summary columns:

```text
pooled_ratio
mean_group_ratio
mom_median_group_ratio
den_ess_groups
top4_den_share
```

Interpretation:

```text
pooled ~= mean ~= MoM       Q estimate is behaving well
pooled far from MoM         rare prefix batches still dominate
low den_ess_groups          denominator weight is concentrated in few groups
large top4_den_share        a few groups control the ratio
```

The final-layer DP itself is exact conditional on the sampled prefix. These
diagnostics are about prefix sampling variance, not final-layer bias.

### 6. Run The Final ZNE Fit

In notebooks, use the Python API:

```python
fit_outputs = run_zne_fit_compare(
    q_groups="final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv",
    observables="mps_N20_L10_Z6Z10_phi02_chi350.csv",
    s_grid=[1, 2, 4, 8],
    s_fit_q=[1, 2, 4, 8],
    s_fit_o=[1, 2, 4, 8],
    shots=100_000,
    n_trials=500,
    q_summary="pointwise_mom",
    out_prefix="zne_fit_N20_L10_Z6Z10_phi02_100k",
)
```

`fit_outputs` is a dictionary of paths:

```python
fit_outputs["summary_csv"]
fit_outputs["trials_csv"]
fit_outputs["q_fits_csv"]
fit_outputs["curves_png"]
fit_outputs["hist_png"]
```

For real measured data, use:

```python
fit_outputs = run_zne_fit_compare(
    q_groups="final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv",
    observables="hardware_observables_N20_L10_Z6Z10_phi02.csv",
    s_grid=[1, 2, 4, 8],
    s_fit_q=[1, 2, 4, 8],
    s_fit_o=[1, 2, 4, 8],
    use_observed=True,
    q_summary="pointwise_mom",
    out_prefix="zne_fit_hardware_N20_L10_Z6Z10_phi02",
)
```

For hybrid fits where only some low-`s` points are noisy and higher-`s` points
are exact classical/MPS values:

```python
fit_outputs = run_zne_fit_compare(
    q_groups="final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv",
    observables="mps_N20_L10_Z6Z10_phi02_chi350.csv",
    s_grid=[1, 2, 4, 8],
    s_fit_q=[1, 2, 4, 8],
    s_fit_o=[1, 2, 4, 8],
    noisy_s=[1, 2],
    shots=100_000,
    exact_sigma=1.0e-6,
    q_summary="pointwise_mom",
    out_prefix="zne_fit_hybrid_s12_100k_N20_L10_Z6Z10_phi02",
)
```

### 7. Read The Results

The final zero-noise estimate is `u_O0`.

Look first at:

```text
*_summary.csv
```

Important columns:

```text
model              dual_exp, gate_qmom, or layer_qmom
mean_u_O0          mean zero-noise estimate over trials
std_u_O0           trial-to-trial spread
mean_rel_error     relative error if true O(0) is known
rmse_rel           RMS relative error if true O(0) is known
p05_rel_error      lower 5 percent quantile
p95_rel_error      upper 95 percent quantile
mean_chi2          average weighted fit residual
```

For real hardware data with no known `O(0)`, compare:

```text
gate_qmom vs layer_qmom agreement
dual_exp vs Q-assisted disagreement
fit residuals
stability under q_summary and alpha changes
```

### 8. Interpret The Plots

`*_curves.png` overlays:

```text
observed O(s)
dual_exp fit
gate_qmom fit
layer_qmom fit
```

`*_hist.png` shows the distribution of extrapolated errors over simulated noisy
trials when a known MPS `O(0)` is available.

## Fitting Model

The Q-assisted fit does not fit `Q+` and `Q-` independently. It first forms:

```math
\ell(s)
=
\frac{1}{2}
\left[
\log Q^+(s)
+
\log Q^-(s)
\right],
```

```math
\delta(s)
=
\frac{1}{2}
\left[
\log Q^+(s)
-
\log Q^-(s)
\right].
```

Then it fits:

```math
\ell(s)
=
-a_1s + a_2s^2 - a_3s^3,
```

```math
\delta(s)
=
d_1s + d_2s^2.
```

These define:

```math
Q_{\mathrm{ave}}(s)
=
e^{\ell(s)}\cosh(\delta(s)),
```

```math
Q_{\mathrm{diff}}(s)
=
e^{\ell(s)}\sinh(\delta(s)).
```

The observable model is:

```math
O_{\mathrm{fit}}(s)
=
u Q_{\mathrm{ave}}(s)
+
v Q_{\mathrm{diff}}(s).
```

At `s=0`, `Q_ave(0)=1` and `Q_diff(0)=0`, so:

```math
O_{\mathrm{fit}}(0) = u.
```

With noisy observable data, the final fit minimizes:

```math
\sum_s
\frac{
\left[
uQ_{\mathrm{ave}}(s)
+
vQ_{\mathrm{diff}}(s)
-
O_{\mathrm{obs}}(s)
\right]^2
}{\sigma_O(s)^2}
+
\alpha_v v^2.
```

Only `v` is regularized. The reported zero-noise estimate `u` is not
regularized.

If `use_observed=False`, the script simulates shot noise from exact/MPS
observable values using:

```math
\sigma_O(s)
\approx
\sqrt{\frac{1-O(s)^2}{N_{\mathrm{shots}}}}.
```

The baseline model ignores Q data and fits:

```math
O_{\mathrm{dual}}(s)
=
c_1e^{-b_1s}
+
c_2e^{-b_2s}.
```

Use this as a baseline, not as the preferred estimator in difficult
sign-changing cases.

## `run_zne_fit_compare` Parameters

Core inputs:

| parameter | meaning |
|---|---|
| `q_groups` | Grouped Q CSV from `final_dp_mom_convergence_sweep.py`. |
| `observables` | MPS or hardware observable CSV. |
| `s_grid` | `s` values present in the Q data. |
| `s_fit_q` | Q points used for the log-mode fit. |
| `s_fit_o` | Observable points used in the final fit. |

Noise and estimator controls:

| parameter | meaning |
|---|---|
| `shots` | Shots per noisy point when simulating noisy observations. |
| `use_observed` | Fit loaded observable values directly instead of simulating shot noise. |
| `noisy_s` | Which `s_fit_o` values get shot sampled. Use `"all"`, `"none"`, or a list such as `[1, 2]`. |
| `exact_sigma` | Error assigned to exact/classical points not shot sampled. |
| `q_summary` | `"pointwise_mom"`, `"mean"`, or `"pooled"`. |
| `q_total_samples` | Select a specific `requested_total_samples` block. `0` uses the largest. |
| `alpha_gate`, `alpha_layer` | Ridge penalty on `v` for gate-wise and layer-wise Q fits. |
| `n_trials`, `seed` | Number of noisy resamples and RNG seed. |

Dual-exponential baseline controls:

| parameter | meaning |
|---|---|
| `dual_exp_rate_min`, `dual_exp_rate_max` | Range of decay rates for the baseline grid. |
| `dual_exp_grid_size` | Number of rate points in the grid. |
| `dual_exp_amp_bound` | Maximum absolute fitted amplitude. |
| `curve_s_max`, `curve_points` | Range and resolution of saved curve plots. |
| `out_prefix` | Prefix for generated outputs. |

## Output Files

`run_zne_fit_compare` writes:

```text
*_summary.csv
*_trials.csv
*_q_fits.csv
*_curves.csv
*_curves.png
*_hist.png
```

Meaning:

| file | purpose |
|---|---|
| `*_summary.csv` | Aggregate statistics over noisy trials. |
| `*_trials.csv` | One row per model and trial. `u_O0` is the extrapolated `O(0)`. |
| `*_q_fits.csv` | Fitted Q-mode coefficients `a1,a2,a3,d1,d2`. |
| `*_curves.csv` | Curve values used to make the first-trial plot. |
| `*_curves.png` | Fit curves plotted against the observed `O(s)` points. |
| `*_hist.png` | Distribution of `O(0)` errors over simulated noisy trials. |

## Command-Line Appendix

The notebook API is the recommended interface for exploration. The scripts can
also be run directly for long resumable jobs.

Grouped Q data:

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

MPS observable data:

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

Final ZNE comparison:

```bash
MPLCONFIGDIR=.mplconfig python fit_zne_compare.py \
  --q-groups final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --observables mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --s-grid 1,2,4,8 \
  --s-fit-q 1,2,4,8 \
  --s-fit-o 1,2,4,8 \
  --shots 100000 \
  --n-trials 500 \
  --q-summary pointwise_mom \
  --out-prefix zne_fit_N20_L10_Z6Z10_phi02_100k
```

Heavy-tail diagnostic:

```bash
MPLCONFIGDIR=.mplconfig python diagnose_q_batch_heavytails.py \
  --batches final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --samples-per-batch 625000 \
  --out-prefix taildiag_N20_L10_Z6Z10_phi02
```

## Development Notes

Keep generated data and plots out of commits unless a benchmark artifact is
intentionally being published.

Check MPS bond dimensions and discarded weights before treating MPS data as an
exact reference.
