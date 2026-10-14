# Pauli-Path ZNE

Research code for Pauli-path sampling and zero-noise extrapolation (ZNE) on
1D Heisenberg brickwall circuits, with symmetry-resolved MPS and rectangular
PEPS extensions.

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
7. [MPS/PEPS Installation, Residual Corrections, and Ratio/Q Usage](docs/tensor-network-methods.md)
8. [Per-CNOT-Layer Noise with Parity Symmetry](docs/cnot-parity-methods.md)

## Tensor-network workflows: CPU/Numba and Apple Metal

The `tensor_zne.py` CLI generates symmetry-resolved MPS or PEPS observable,
ratio, and Q curves, with residual correction enabled by default. Run commands
from the repository root; site indices are zero based. The sampling notebooks
and fitting interface above remain available.

### Installation

Use Python **3.11 or 3.12** (validated with 3.12). The pinned NumPy/Numba stack
is not compatible with Python 3.13.

```bash
git clone https://github.com/jcohn625/Pauli-Path-ZNE.git
cd Pauli-Path-ZNE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-peps.txt
```

`requirements-peps.txt` installs both methods, including Numba, quimb and
symmray. For MPS only, use `requirements-mps.txt` instead. On Windows, activate
with `.venv\Scripts\activate`. CPU use does not require MLX or a GPU.

For reproducible single-thread CPU comparisons:

```bash
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/.numba_cache"
```

### CPU runs with Numba

The measured Numba CPU policy is **enabled automatically**. For after-CNOT MPS,
it accelerates charge-sector gathering, global singular-value selection and
factor reconstruction. The CPU PEPS layer engine uses fused gauge scaling and
packing for small FP64 blocks; larger or unsupported blocks retain NumPy.
QR, SVD and large matrix products continue to use NumPy/SciPy BLAS/LAPACK.
The first call can include compilation or cache-loading overhead.

```bash
# MPS: correlated Pauli noise after each CNOT layer; residual ratio and Q curves.
python tensor_zne.py mps --noise-placement after_cnot \
  --n 6 --steps 2 --sites 1 4 --cap 4 --guide-cap 32 \
  --lam .001 .002 .003 --pair-rate .001 --mps-backend cpu \
  --scales 0 1 2 4 --mode both --out results/mps_cpu

# PEPS: CPU gate updates and CPU boundary contraction.
python tensor_zne.py peps --noise-placement after_cnot \
  --nx 4 --ny 4 --steps 2 --sites 5 10 --cap 8 \
  --boundary-cap 16 --environment-cap 16 --pair-rate .001 \
  --gate-backend batched-cpu --boundary-backend blas \
  --scales 0 1 2 4 --mode both --out results/peps_cpu
```

`--noise-placement after_cnot` inserts a graph-wide Pauli-Lindblad channel after
each nonempty CNOT layer. `--pair-rate .001` is the rate of **each of the nine**
Pauli-pair generators per noise edge. The standard Z2 × Z2 labels are restored
after each full Heisenberg matching using transported intermediate frames.
See the [CNOT noise guide](docs/cnot-parity-methods.md) for scheduled rates,
long-range correlations and the noise-file format. Omitting the flag retains
the historical full-Heisenberg-noise workflows.

To compare against the reference CPU helper routines, set
`export PAULI_CPU_NUMBA=0` before starting Python, then rerun the same command.
Use `unset PAULI_CPU_NUMBA` to restore the optimized policy. This switch controls
the new tensor-network helpers, not every use of Numba elsewhere in the project.

### Apple Silicon Metal and hybrid runs

Install the optional MLX dependency on a supported Apple Silicon macOS system:

```bash
python -m pip install -r requirements-metal.txt
```

Select gate evolution and boundary contraction independently:

| Option | Execution |
|---|---|
| `--mps-backend cpu` (default) | CPU evolution, Numba helpers and CPU decompositions |
| `--mps-backend metal` | Randomized-SVD matrix products on Metal; QR/reduced SVD and other MPS work on CPU |
| `--gate-backend quimb` (PEPS default) | quimb/symmray CPU gate updates, with specialized CPU noise updates |
| `--gate-backend batched-cpu` | Custom CPU gate engine, including eligible Numba scaling/packing |
| `--gate-backend metal` | Metal packing/products with CPU gate decompositions |
| `--boundary-backend blas` (default) | CPU FP64 boundary contractions and decompositions |
| `--boundary-backend metal-hybrid` | Metal FP32 boundary contractions; selected QR/SVD batches on CPU |
| `--boundary-backend metal-jacobi` | Metal FP32 boundary contractions and custom QR/Jacobi SVD |

A hybrid PEPS run keeps gate evolution on CPU and moves boundary contractions
to Metal, using CPU LAPACK where the tested routing policy favors it:

```bash
python tensor_zne.py peps --noise-placement after_cnot \
  --nx 6 --ny 6 --steps 2 --sites 7 14 --cap 16 \
  --boundary-cap 32 --environment-cap 32 --pair-rate .001 \
  --gate-backend batched-cpu --boundary-backend metal-hybrid \
  --scales 0 1 --mode both --out results/peps_hybrid
```

For Metal gate products, change `--gate-backend batched-cpu` to `metal`.
For custom GPU decompositions, change the boundary backend to `metal-jacobi`;
`--jacobi-sweeps` defaults to 32. To try MPS hybrid execution, change
`--mps-backend cpu` to `metal` in the MPS example. Small sector matrices can still
use full CPU SVD, so selecting Metal may perform little or no GPU work.

CPU is the default and often wins for small problems. Metal arithmetic is FP32;
compare results and bond convergence against CPU FP64. There is no universal
bond-dimension threshold for GPU speedup. On our M4 Pro, the new CPU kernels
reduced tested full-solve times by about 5–10% for MPS and 4–8% for PEPS. An
earlier 6×6 hybrid test reduced full-solve time by 9%; these are separate
experiments and the gains should not be added together. See the
[CPU benchmarks](docs/cpu-numba-results.md) and
[Metal/hybrid benchmarks](docs/apple-hybrid-refinement-results.md).

### Residuals, outputs and further usage

Use `--mode ratio`, `--mode q`, or `--mode both`. Residual correction is on by
default; `--no-residual` disables it. PEPS defaults to row-level residual
contractions; `--residual-granularity bond` retains detailed bond-level
diagnostics. MPS guide-memory options are in the [CNOT guide](docs/cnot-parity-methods.md).

Each output prefix produces `.json`, `_curves.csv`, `_observables.csv`, and,
when Q normalization is valid, `_q_groups.csv`. The zero-noise normalization
point is added automatically. PEPS residuals correct contraction of the evolved
PEPS, not its spatial gate-truncation error; converge spatial and boundary caps
separately. See the [installation, Python API and ratio/Q fitting guide](docs/tensor-network-methods.md)
for definitions and examples. Experimental research modules and rejected
optimizations are documented in the reports and are not enabled automatically.

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
