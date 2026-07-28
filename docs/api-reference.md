# API Reference

[Back to README](../README.md)

## Main Function

```python
from fit_zne_compare import run_zne_fit_compare
```

Example:

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

## Parameters

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

## Input CSVs

Grouped Q CSV columns:

```text
method
group
s
q_plus
q_minus
```

For `q_summary="pooled"`, the Q CSV must also contain:

```text
plus_num
plus_den
minus_num
minus_den
```

Observable CSV columns:

```text
s
```

plus one value column:

```text
mps_value
O_mps
observable
value
mean
obs
y
```

Optional error columns:

```text
sigma
se
stderr
std_error
```

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
