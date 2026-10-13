# Symmetry-resolved MPS and PEPS, residual corrections, and ratio/Q workflows

The portable entry point is `tensor_zne.py`. It adds deterministic tensor-network
curves alongside the existing Pauli-path sampling notebooks and
`fit_zne_compare.py`. Both engines use a local Pauli basis and Z2 × Z2 symmetry.
Run commands from the repository root. Site indices are zero based.

For noise after each CNOT layer, including cross-pair and longer-range Pauli
generators with parity-resolved signed/Q evolution, see the
[CNOT parity extension](cnot-parity-methods.md). The defaults described below
remain the original full-Heisenberg-noise modes.

## Installation

Use **Python 3.11 or 3.12** for the pinned CPU environment (validation uses 3.12).
Python 3.13 is not supported by these particular NumPy/Numba pins.

```bash
git clone https://github.com/jcohn625/Pauli-Path-ZNE.git
cd Pauli-Path-ZNE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# MPS only:
python -m pip install -r requirements-mps.txt

# MPS plus PEPS (includes quimb and symmray):
python -m pip install -r requirements-peps.txt

# Also install plotting for the existing ZNE fitter/notebooks:
python -m pip install matplotlib jupyter
```

On Windows activate with `.venv\Scripts\activate`. The source modules are run
from the checkout; no editable package installation is necessary. PEPS directly
uses quimb and symmray, so they must be installed. MPS does not import either.
The dependencies are ordinary PyPI packages; no author-specific folders are
required. First calls can include Numba compilation time.

The pinned NumPy 1.26.4 / SciPy 1.17.1 / Numba 0.60.0 stack uses NumPy
OpenBLAS wheels while meeting quimb's SciPy >=1.16 requirement. The host's NumPy 2.3.0/Accelerate stack emitted
matrix-product warnings during development; it is not the reference installation.
Do not suppress numerical warnings as a substitute for a convergence check.
For reproducible, modest CPU resource use:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/.numba_cache"
```

On Apple Silicon macOS, optionally install Metal support into the same environment
(validated on macOS 26; the MLX wheel must support your macOS version):

```bash
python -m pip install -r requirements-metal.txt
```

MLX/Metal is optional; Linux/Windows CPU users should omit it. PEPS uses pinned
quimb 1.14.0, symmray 0.4.0, cotengra 0.8.1 and autoray 0.8.11 because the custom
boundary implementation uses their internal interfaces.

## Quick runs

A small MPS run with both ratio and Q curves, including residual correction:

```bash
python tensor_zne.py mps --n 6 --steps 3 --sites 2 4 \
  --cap 4 --guide-cap 64 --terminal 1 \
  --scales 0 1 2 4 8 --mode both --out results/mps
```

A small rectangular PEPS run:

```bash
python tensor_zne.py peps --nx 2 --ny 3 --steps 2 --sites 0 1 \
  --cap 8 --boundary-cap 4 --environment-cap 16 \
  --scales 0 1 2 4 8 --mode both --out results/peps
```

Use `--mode ratio` to avoid the extra unsigned contractions, or `--mode q` for Q
work. `--no-residual` runs the underlying finite-bond calculation. The default
angle is pi/16; `--phi` changes it, and `--lam x y z` sets independent Pauli-noise
rates. The noise scale multiplies all rates. The zero-noise point is added
automatically because it normalizes both methods.

Outputs for each prefix:

| File | Contents |
|---|---|
| `_curves.csv` | Base/corrected O, unsigned U, R, Q± and sign-sector masses |
| `_observables.csv` | Corrected O(s), in the existing fitter's observable schema |
| `_q_groups.csv` | Deterministic Q± in the existing fitter's grouped schema |
| `.json` | Configuration, complete per-channel results, and residual diagnostics |

Undefined ratios are blank in CSV and null in JSON. The Q fit export is omitted
when any sign sector is absent, nonpositive, or too small to normalize reliably.
No artificial clipping is used to make an invalid Q curve fit-ready. An existing
Q export at the same prefix is removed in that case. Outputs overwrite the chosen
prefix. A single deterministic group is not a Monte Carlo uncertainty estimate.

## Definitions and fitting

For each noise scale, compute the signed observable O(s) and the unsigned
Pauli-path mass U(s). Absolute values are taken at each two-qubit Heisenberg
transfer and at the terminal singlet functional, **before summing paths**:

```text
Z+(s) = [U(s) + O(s)] / 2
Z-(s) = [U(s) - O(s)] / 2
Q+(s) = Z+(s) / Z+(0)
Q-(s) = Z-(s) / Z-(0)
R(s)  = O(s) / O(0)
```

Each channel is corrected separately before forming the corrected ratios.
The sign sectors depend on the gate decomposition: these Heisenberg-gate Q
curves are not interchangeable with the elementary-CNOT Q curves in `cnot_zne`.
Large U compared with O can make the sector recovery/normalization sensitive to
floating-point precision and truncation. Check convergence of the derived curve.

### Ratio method

Combine the computed `ratio_corrected` at a positive scale with a matching noisy
observable measurement y(s): `O0_estimate = y(s) / R(s)`. Multiple independent
measurements with standard errors sigma can use a one-parameter weighted fit:

```python
import numpy as np

# All arrays must refer to the same positive scales and the same circuit.
r = np.array([0.8, 0.6])       # replace with ratio_corrected from your run
y = np.array([0.16, 0.12])     # replace with measured noisy observables
sigma = np.array([0.01, 0.01])
w = 1 / sigma**2
O0_estimate = np.sum(w * r * y) / np.sum(w * r**2)
measurement_se = 1 / np.sqrt(np.sum(w * r**2))
```

The reported measurement error in this formula excludes error in the computed
ratio. Near O(0)=0 the ratio normalization is ill-conditioned; use raw observables
or assess a Q-based fit instead. The CSV includes both base and corrected ratios
for comparison; a correction is not automatically an improvement.

### Existing Q-assisted fitter

The generated files can be used directly with the existing interface:

```python
from fit_zne_compare import run_zne_fit_compare

outputs = run_zne_fit_compare(
    q_groups="results/mps_q_groups.csv",
    observables="results/mps_observables.csv",
    methods="layer",
    s_grid=[1, 2, 4, 8],
    s_fit_q=[1, 2, 4, 8],
    s_fit_o=[1, 2, 4, 8],
    q_summary="pointwise_mom",
    shots=100_000,
    n_trials=100,
    out_prefix="results/mps_fit",
)
```

This example generates a simulated shot-noise comparison using the computed
observable curve. For measured observables, replace the observable file and use
`use_observed=True`; supply the existing fitter's `sigma` column for measurement
errors. Match `case=tensor_zne`, noise scales and the circuit convention. Use
`methods="gate"` for the default PEPS export, or `"layer"` if PEPS was run with
`--peps-noise-placement layer`. Do not pass the fitter's default `gate,layer` when
only one method is in the file. See [fitting details](fitting-procedure.md).
Finite-bond bias and finite-guide bias are not included in the fitter's statistical
uncertainty. A 2D PEPS dataset must be paired with measurements of that same 2D
circuit, not existing 1D observable data.

## Circuit conventions

Both engines start from a product of singlets and compute a two-Z observable.
The wrapper uses uniform independent Pauli noise; lower-level modules expose
more general observables/noise schedules where supported.

| | MPS | PEPS |
|---|---|---|
| Geometry | Open even-length chain | Open rectangle, even site count, dimensions >=2 |
| Singlets | (0,1), (2,3), ... | Adjacent pairs along `square(nx,ny).order` |
| Local storage | Two-qubit blocks of dimension 16 | One-qubit Pauli dimension 4 |
| Symmetry | Dense tensors, charge-block QR/SVD | Block-sparse symmray Z2Z2 arrays |
| Gate convention | Historical backward sweep: even gate, all-site noise, odd gate, all-site noise | Forward sublayers: horizontal-even, horizontal-odd, vertical-even, vertical-odd; default noise after gates |
| Noise placement | All sites after each backward half-layer | Touched sites by default; `layer` damps all sites |

PEPS physical site id is `y*nx+x`. Its snake order changes with rectangle shape;
the pairing is explicitly defined above. Matching only N and number of steps
does not make the two geometries or noise conventions equivalent. MPS retains
the historical convention used in the repository's blocked-MPS/Q workflows.
The independent MPS exact-network regression explicitly reverses the physical
layer list and uses `noise_timing='before'` to reproduce this backward ordering.

## Residual correction and convergence

**MPS:** `mps_residual.run` contracts actual discarded two-site matrices with
backward guides. The base forward evolution is unchanged. An exact adjoint makes
the telescope exact; finite guide bonds are an approximation. The final zero,
one or two terminal layers receive a separate untruncated-rank suffix correction
(rank bound 16**terminal). The CLI restricts corrected runs to terminal depths
0–2. Vary both `--cap` and `--guide-cap`; inspect the correction and individual
contributions in JSON. The low-level API also accepts `guide_cache_mb` to trade
memory for recomputation. Memory and runtime can grow sharply with bond size.

**PEPS:** `peps_boundary_residual.streaming_residuals` corrects forward
boundary-MPS truncations using backward row environments. It retains backward
environments and scalar corrections, not all forward snapshots. With exact
suffix environments (`environment_cap=None` in Python), the telescope gives the
exact contraction **of the fixed, already evolved PEPS**. It does not correct
the spatial simple-update/gate truncations. Independently converge `--cap`
(spatial evolution), `--boundary-cap` (forward contraction), and
`--environment-cap` (correction guide). Exact environments are for small tests.

Neither finite-bond method supplies a certified error bound or guarantees that
corrected results are closer to the physical answer. Separate correctness tests
from large-system bond convergence.

## CPU and Metal

| Option | Work on GPU | Work remaining on CPU |
|---|---|---|
| MPS default | None | Numba transfer kernel; NumPy/SciPy QR/SVD |
| `--mps-backend metal` | Randomized-SVD large matrix products | QR, reduced SVD, tensor evolution/bookkeeping |
| PEPS default | None | quimb/symmray and NumPy/SciPy |
| `--gate-backend metal` | Packing, gate products, tensor reconstruction | Gate QR/SVD and other evolution work |
| `--boundary-backend metal-jacobi` | Boundary tensor blocks, contractions, custom QR/Jacobi SVD, streamed residual contractions | Topology, sector bookkeeping, scalar reads |

Metal products use float32; CPU reference calculations use float64. Small MPS
blocks may use full CPU SVD even with Metal selected. These options do not imply
a fully GPU-controlled simulator or guaranteed speedup. Increase
`--jacobi-sweeps` (default 32) when checking GPU convergence. The custom reduced
Jacobi core currently has a dimension limit of 512. Avoid concurrent Python
threads using the PEPS contexts, which temporarily patch process-global library
functions; use separate processes for independent jobs.

For example, append `--gate-backend metal --boundary-backend metal-jacobi` to the
PEPS command above after installing MLX. The MPS residual workflow also accepts
`--mps-backend metal`; validate precision against the CPU result at your bond size.

## Python API and tests

```python
from tensor_zne import curves, evaluate_mps, evaluate_peps

result = curves(evaluate_mps, [0, 1, 2], mode="both",
                n=6, steps=3, sites=(2, 4), cap=4, guide_cap=64, terminal=1)
print(result["rows"])

peps = evaluate_peps(nx=2, ny=3, steps=2, sites=(0, 1),
                     cap=8, boundary_cap=4, environment_cap=16)
print(peps["approximate"], peps["corrected"])
```

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q -W error::RuntimeWarning
# Optional Apple Silicon GPU regression suite:
TEST_PEPS_GPU_BOUNDARY=1 TEST_PEPS_METAL=1 TEST_MPS_METAL=1 python -m pytest tests -q
```

Tests include independent exact signed/unsigned contractions, unchanged MPS
baselines, exact residual telescopes, streamed-versus-retained PEPS corrections,
charge conservation, backend restoration, Q normalization and existing CSV-loader
compatibility. See [validation record](tensor-network-validation.md).
