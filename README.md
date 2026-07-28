# Pauli-Path ZNE

Research code for Pauli-path sampling and zero-noise extrapolation (ZNE) on
1D Heisenberg brickwall circuits.

The practical target is to estimate a zero-noise observable, such as `Z_6 Z_10`,
by combining:

```text
Q data       classical Pauli-path sampling of Q+(s) and Q-(s)
O data       noisy hardware, simulated shot-noisy, or MPS observable data O(s)
baseline     ordinary dual-exponential ZNE fit for comparison
```

The recommended interface is the notebook:

[notebooks/Final_Layer_DP_Usage.ipynb](notebooks/Final_Layer_DP_Usage.ipynb)

The main final fitting function is:

```python
from fit_zne_compare import run_zne_fit_compare
```

## Table Of Contents

1. [Motivation And Workflow](docs/workflow.md)
2. [Notebook User Manual](docs/notebook-user-manual.md)
3. [Fitting Procedure](docs/fitting-procedure.md)
4. [API Reference](docs/api-reference.md)
5. [Command-Line Appendix](docs/command-line.md)
6. [Repository Map](docs/repository-map.md)

## Why This Exists

Standard ZNE fits the measured observable curve `O(s)` directly as a function of
noise scale `s`. In the sign-changing and rare-event regimes we are studying,
direct fits such as a dual exponential can be unstable because the same few
noisy points determine both the decay rates and the zero-noise intercept.

The Q-assisted strategy separates the problem into two pieces:

1. Learn the Pauli-path sector damping curves `Q+(s)` and `Q-(s)` classically.
2. Use measured or simulated observable data only to fit two final coefficients.

This makes the noisy-data part of the final extrapolation much better
conditioned.

## Quick Start

From a notebook:

```python
from fit_zne_compare import run_zne_fit_compare

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

fit_outputs["summary_csv"]
```

The final zero-noise estimate is the `u_O0` column in the generated summary and
trial CSVs.

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

For plotting in restricted environments:

```bash
export MPLCONFIGDIR=.mplconfig
```

Generated CSV, PNG, cache, and notebook checkpoint files are ignored by default.

## Current Preferred Defaults

```text
noise_model       independent
noise_placement   layer
q_summary         pointwise_mom
s_grid            1,2,4,8
fit models        dual_exp, gate_qmom, layer_qmom
```

Avoid `legacy_sum` unless intentionally reproducing older results.

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

See [Output Files](docs/api-reference.md#output-files) for how to interpret
them.

## Development Notes

Keep generated data and plots out of commits unless a benchmark artifact is
intentionally being published.

Check MPS bond dimensions and discarded weights before treating MPS data as an
exact reference.
