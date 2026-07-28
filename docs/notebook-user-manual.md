# Notebook User Manual

[Back to README](../README.md)

Use this notebook as the canonical workflow:

[../notebooks/Final_Layer_DP_Usage.ipynb](../notebooks/Final_Layer_DP_Usage.ipynb)

## 1. Import The Repo

If the notebook is opened from `notebooks/`, add the repo root to `sys.path`.

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

Key notebook imports:

```python
from fit_zne_compare import run_zne_fit_compare
from final_dp_mom_convergence_sweep import summarize as summarize_q_groups
from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz
```

The notebook also imports the gate-wise and layer-wise sampler modules directly.

## 2. Choose The Target Problem

Example notebook parameters:

```python
N_QUBITS = 20
N_STEPS = 6
Q1, Q2 = 6, 10
PHI = 0.2
LAMBDA_BASE = 1.0e-3
S_GRID = [1.0, 2.0, 4.0, 8.0]
```

For production runs, increase `N_STEPS`, `TOTAL_SAMPLES`, and MPS `CHI_MAX`.

Keep the MPS noise convention aligned with the sampler:

```text
noise_model       independent
noise_placement   layer
```

Avoid `legacy_sum` unless intentionally reproducing older results.

## 3. Compute Or Load O(s)

For benchmarks, compute an MPS reference curve at:

```text
s = 0,1,2,4,8
```

The `s=0` point is the true zero-noise benchmark when MPS is converged. The
positive `s` values are the noisy curve used by the ZNE fit.

For hardware data, create a CSV with:

```text
s, observable
```

or:

```text
s, observable, sigma
```

Accepted observable value columns:

```text
mps_value
O_mps
observable
value
mean
obs
y
```

Accepted error columns:

```text
sigma
se
stderr
std_error
```

## 4. Estimate Q+(s) And Q-(s)

The grouped Q sampler produces:

```text
q_plus(s)
q_minus(s)
plus_num(s), plus_den
minus_num(s), minus_den
```

The grouped data supports:

```text
pooled ratio       sum numerator / sum denominator
group mean         average of group ratios
pointwise MoM      median of group ratios
```

The default final fit uses:

```text
q_summary = "pointwise_mom"
```

because the finite-sample prefix distribution can be heavy-tailed.

## 5. Diagnose Q Convergence

Use these summary columns:

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

The final-layer DP is exact conditional on the sampled prefix. These diagnostics
are about prefix sampling variance, not final-layer bias.

## 6. Run The Final ZNE Fit

For MPS benchmark data with simulated shot noise:

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

For real measured data:

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

For hybrid fits where low-`s` points are noisy and high-`s` points are exact
classical/MPS values:

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

## 7. Read The Results

`fit_outputs` is a dictionary of paths:

```python
fit_outputs["summary_csv"]
fit_outputs["trials_csv"]
fit_outputs["q_fits_csv"]
fit_outputs["curves_png"]
fit_outputs["hist_png"]
```

The final zero-noise estimate is `u_O0`.

Important summary columns:

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

## 8. Interpret The Plots

`*_curves.png` overlays:

```text
observed O(s)
dual_exp fit
gate_qmom fit
layer_qmom fit
```

`*_hist.png` shows the distribution of extrapolated errors over simulated noisy
trials when a known MPS `O(0)` is available.
