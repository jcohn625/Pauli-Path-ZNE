# Hybrid Apple execution and bidirectional boundary refinement

This pass adds an opt-in `metal-hybrid` PEPS boundary backend, fixes a numerical
problem in Metal Householder QR, and tests bidirectional finite boundary-message
refinement. The standard CPU FP64 `blas` backend remains the default.

## Hybrid backend

`peps_hybrid_decomp.py` keeps contractions in MLX/Metal and sends selected real
FP32 matrix batches to SciPy LAPACK. QR routes to CPU at `min(m,n) >= 64` and SVD
at `min(m,n) >= 16`; smaller matrices use the existing Metal kernels. Routing uses
actual charge-sector matrix dimensions, not the requested tensor bond cap.
Factors return as FP32 MLX arrays and the existing global sector rank competition
still selects the retained bond. No Gram approximation or rank preselection is
introduced by the CPU factorization path.

This is a synchronous hybrid route. Unified memory avoids a discrete GPU bus,
but synchronization, NumPy views, LAPACK workspace/outputs, and returned device
arrays still have costs. The benchmark includes those costs. `cpu_boundary_seconds`
tracks host-boundary work, while `cpu_boundary_array_bytes` counts input/output
array bytes handled across the boundary; it is neither peak memory nor measured
physical bus traffic.

The Apple/MLX installation requirements are unchanged. Both corrected and raw
PEPS workflows accept the new backend:

```python
from tensor_zne import evaluate_peps
result = evaluate_peps(
    nx=6, ny=6, sites=(7, 14), steps=2,
    cap=16, boundary_cap=32, environment_cap=32,
    gate_backend='batched-cpu', boundary_backend='metal-hybrid',
    noise_placement='after_cnot', pair_rates=.001,
)
```

CLI: `--boundary-backend metal-hybrid`. `details.boundary_gpu_stats` reports
routing counters in residual mode. The ordinary CPU backend keeps FP64, while
this path uses FP32; comparisons must account for that precision difference.
The thresholds are an explicit policy tested on this M4 Pro, not a universal
crossover rule or runtime autotuner. Low-level experiments can override them in
`gpu_boundary(decomposition_backend='hybrid', cpu_qr_min=..., cpu_svd_min=...)`.

### Measured performance after the QR fix

Noise follows each CNOT layer, with single rates (.001,.002,.003), pair rate
.001, and Z observables at sites (5,10) for 4×4 or (7,14) for 6×6. Small cases
use six steps, spatial cap 8 and boundary/guide cap 32; the larger case uses two
steps, spatial cap 16 and boundary/guide cap 32. One warm-up and three repetitions
are used per variant. Boundary-stage times reuse the evolved PEPS and GPU input:

| Case | CPU FP64 ms | CPU FP32 ms | Metal ms | Hybrid ms |
|---|---:|---:|---:|---:|
| 4×4, O, scale 0 | 29.45 | 25.96 | 53.38 | 48.91 |
| 4×4, O, scale 1 | 29.94 | 26.53 | 49.78 | 47.12 |
| 4×4, U, scale 1 | 30.67 | 26.14 | 50.79 | 46.00 |
| 6×6, O, scale 1 | 1081.09 | 947.17 | 2043.23 | 727.41 |

On the 6×6 boundary, hybrid is **1.49× faster than CPU FP64**,
**1.30× faster than CPU FP32**, and **2.81× faster than Metal alone**.
Its corrected observable differs from the FP64 boundary result by **3.08 ppm**.
That comparison measures implementation/precision differences, not error relative
to the exact circuit. Small 4×4 boundaries still favor CPU.

A separate serial test of the complete `evaluate_peps` API, including evolution,
conversion and residual correction, measured **3.531 s CPU FP64 versus
3.209 s hybrid**: **1.100× speedup**, or **9.1% less elapsed time**.
The returned corrected values were -6.57601113017e-05 and
-6.57603140398e-05, respectively. This full-workflow measurement,
not the isolated factorization speedup, is the practical gain for this case.

The benchmark also tested routing only very large factors (QR>=256/SVD>=64)
and routing every factor to CPU. The former missed useful opportunities; the
latter was close to the chosen policy on the large case and slower on the small
cases. The chosen thresholds preserve the tested small-kernel path without
requiring a runtime calibration run.


## Metal QR correctness fix and panel experiments

A replay of actual captured boundary batches found that the previous Metal QR
could return a badly non-orthogonal Q while reconstructing its input reasonably
well. Weak intermediate columns underflowed in the direct sum of squared entries.
That compromises the canonical gauge used by later tensor truncations.

The Householder shader now finds the maximum absolute column entry first,
computes the norm in scaled coordinates, and computes both tau and the reflector
vector in those same scaled coordinates. Zero columns get an identity reflector.
This is applied to both direct and blocked QR, including the small GPU QRs used
by the hybrid backend. It does not substitute a polar factorization for QR.

For the captured batch of two 1026×1024 matrices, panel-32 QR's aggregate
`||Q.T Q - I||_F` fell from **386.5 to 4.48e-5**. CPU FP32 LAPACK gave **4.83e-5**.
The corrected reconstruction relative error was **2.10e-6**. A second batch of
two 1022×1024 matrices fell from **1026.9 to 4.16e-5**. These are norms across
whole batches, not maximum entry errors.

Actual captured decomposition replay, with device-resident input and synchronized
outputs (milliseconds):

| Operation and batch | Metal panel 16 | Metal panel 32 | CPU LAPACK with device boundary |
|---|---:|---:|---:|
| QR: 2 × 1026×1024 | 44.75 | 49.54 | 33.41 |
| QR: 4 × 2048×256 | 27.79 | 30.65 | 28.67 |
| QR: 2 × 1022×1024 | 45.15 | 48.83 | 32.27 |
| SVD: 1 × 128×240 | — | 29.13 | 1.21 |
| SVD: 1 × 128×160 | — | 26.23 | 0.64 |

Panel widths 64 and 128 were substantially slower here. Width 16 is available
for experiments as `GPUDecompositions(qr_panel_width=16)` or the same keyword
in `gpu_boundary`. The existing width-32 default is retained; three replayed QR
batches do not establish a universal optimum. The input batches were captured
from the earlier Metal trajectory and are replayed unchanged before/after the
fix, so this isolates factorization behavior on identical matrices.

## Bidirectional message refinement

`peps_message_refinement.py` initializes ordinary top and bottom boundary MPS
messages, then alternates downward and upward sweeps. Every update absorbs an
original PEPS row into the newly computed incoming message and uses the opposite
directed message to guide KQ/RTM compression. It does not repeatedly compress an
old message without rebuilding its source. Messages preserve Z2×Z2 charges,
remain signed, and are not scalar-fitted or divided by overlaps.

This finite-strip construction follows the local environment-guided compression
idea from [CT-bMPS, Sec. 2.3](https://arxiv.org/html/2610.06574v1#S2.SS3). It does
not implement arbitrary contraction-tree optimization or the variational MP-BP
stationarity equations. The new bottom messages can guide the existing ordinary
forward residual trajectory through `streaming_residuals(backward_guides=...)`.

We tested 4×4, spatial cap 8, steps 2/6, signed/unsigned, scales 0/1 and matched
boundary/guide caps 4/8/16/32: **32 settings**, with 1, 3 or 6 complete
bidirectional sweeps. References exactly contract the same fixed evolved PEPS;
they do not remove its spatial simple-update error. We also compared doubling
only the ordinary guide cap and doubling both ordinary caps.

| Variant | Strict accuracy wins vs original caps | Median time / ordinary residual |
|---|---:|---:|
| 1 refinement sweep | 10/32 | 3.23× |
| 3 refinement sweeps | 9/32 | 6.03× |
| 6 refinement sweeps | 8/32 | 10.19× |
| Double ordinary guide cap | 32/32 | 1.00× |
| Double both ordinary caps | 32/32 | 1.01× |

Some strict wins are at rounding level. Six-sweep refinement beats the best
cheaper tested ordinary alternative in **0/32** settings. Nearly unchanged cost
for larger caps is specific to these small networks, not a claim about asymptotic
scaling. No refinement mode is selected by the main API.

A particularly useful failure case is the signed six-step, zero-noise network.
At cap 8, six refinement sweeps reduce disagreement among cut estimates to
**0.0262%**, yet their mean is **46.6%** wrong and the refined-guide residual is
**42.7%** wrong. At cap 16, cut disagreement is only **0.000879%**, while the cut
mean remains **13.2%** wrong and the residual **5.04%** wrong. Ordinary residuals
at cap 16 have **3.44%** error; raising only the guide cap to 32 reduces that to
**0.309%** at essentially the same measured cost.

Agreement of jointly adapted messages is therefore not a correctness test for
these signed, cancelling networks. The experiment supports spending the next
accuracy budget on ordinary guide/boundary-cap convergence before adding more
iterations of this particular refinement rule.

## Reproduction

All timing runs use one warm-up followed by three measured repetitions, and
NumPy/SciPy/Numba CPU thread limits of one. Final reported benchmarks were run
serially. The hardware is an M4 Pro with 24 GiB unified memory.

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/pauli_cnot_numba MPLCONFIGDIR=/tmp/pauli_cnot_mpl
export PYTHONPATH=cnot_zne
mkdir -p results/refinement_hybrid
.venv312/bin/python benchmarks/run_hybrid_boundary.py --output results/refinement_hybrid/boundary_stable.json
.venv312/bin/python benchmarks/run_hybrid_workflow.py --output results/refinement_hybrid/workflow_final.json
.venv312/bin/python benchmarks/run_captured_decompositions.py --input results/refinement_hybrid/boundary_stable.npz --output results/refinement_hybrid/decompositions_stable.json
.venv312/bin/python benchmarks/run_message_refinement.py --output results/refinement_hybrid/messages.json
```

The decomposition replay above used the original pre-fix `boundary.npz`; a fresh
capture can have different weak sectors because the corrected QR changes the
numerical trajectory. It must not be described as reproducing those exact
historical inputs. The archive includes the original capture's SHA-256.

## Validation

Final full regression: **269 passed, 1 skipped, 4 subtests passed in 23.59 s**,
with RuntimeWarnings treated as errors and all Metal test groups enabled.
New tests cover full QR/SVD factors for rank-deficient and zero inputs, weak
columns down to 1e-30, multiple QR panel widths, global charge-rank selection,
public residual/non-residual paths, after-CNOT ratio/Q results, bidirectional
exact limits, symmetry, and supplied-guide immutability.

```sh
TEST_PEPS_METAL=1 TEST_PEPS_GPU_BOUNDARY=1 TEST_MPS_METAL=1 .venv312/bin/python -m pytest tests cnot_zne/tests -q -W error::RuntimeWarning --maxfail=1
```


[Raw measurements and final source hashes](apple-silicon/hybrid-refinement-20261013.json)
include the pre-fix decomposition diagnostics separately from the corrected
measurements. Earlier overlapping preliminary large-case/workflow timings are
excluded from the published final comparisons.
