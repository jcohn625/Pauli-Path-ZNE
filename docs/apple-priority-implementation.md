# Priority implementation: row residuals, environment guidance, blocked Metal

Follow-up: [hybrid execution and bidirectional refinement](apple-hybrid-refinement-results.md) implements the next targets, including a correction to Metal QR for weak intermediate columns. The measurements below describe the earlier implementation.

Measured on the local M4 Pro / 24 GiB machine, 2026-10-13. CPU BLAS threads
were limited to one. Benchmarks ran sequentially, with an untimed warm-up and
three timed repetitions. Times below are medians. These measurements cover the
listed workloads; they are not universal CPU/GPU crossover thresholds.

## 1. Row telescoping is enabled

`tensor_zne.evaluate_peps` and its CLI now default to row residual records.
The low-level `streaming_residuals` retains its previous bond default. Explicit
choices are `residual_granularity='row'|'bond'` in the high-level API,
`--residual-granularity row|bond` in the CLI, and `granularity` in the low-level
API. CPU and Metal boundary paths both support the option.

For a row with a fixed backward guide E and successive truncated boundaries
B_0,...,B_m, the summed correction is

`sum_j <B_(j-1)-B_j,E> = <B_0,E> - <B_m,E>`.

The gate/noise schedule, charge sectors, global singular-value selection,
forward truncations, and backward guides are unchanged. Only the intermediate
scalar overlaps are omitted. Floating-point accumulation can differ at rounding
level. Signed and unsigned paths continue to use their distinct elementary
transfer schedules.

The new records report `granularity` and `overlap_count`. Row mode's
`absolute_correction_sum` sums absolute row corrections; it is not the old sum
of absolute bond corrections. Neither quantity is a certified error bound when
guides are approximate. Select bond mode when that detailed diagnostic is needed.

The benchmark uses noise after each CNOT layer, two evolution steps, signed
Z observables, spatial cap 8, singles (.001,.002,.003), pair rate .001, and scale 1.

| Lattice | Boundary/guide cap | Bond residual ms | Row residual ms | Stage speedup |
|---|---:|---:|---:|---:|
| 4×4 | 8 | 33.15 | 28.72 | 1.154× |
| 4×4 | 16 | 33.75 | 29.09 | 1.160× |
| 6×6 | 8 | 103.53 | 80.57 | 1.285× |
| 6×6 | 16 | 118.86 | 95.89 | 1.240× |
| 8×8 | 8 | 144.55 | 101.85 | 1.419× |
| 8×8 | 16 | 163.20 | 119.86 | 1.362× |

All six scaled corrected values were identical between modes in this run.
Including the separately measured evolution time, speedups are 1.003–1.015×.
The largest lattice uses 12 residual overlaps instead of 48. This optimization
helps boundary correction; it does not accelerate spatial PEPS evolution.

## 2. Two finite environment-guided pilots remain experimental

`peps_environment_experiment.py` contains two implementations. Both use the
actual signed single-layer closed network, preserve Z2×Z2 blocks, select ranks
globally across sectors, and avoid scalar fitting, inverse singular values,
and positivity assumptions.

* **Open-bond projector:** contract the row and backward guide leaving one
  forward bond open, producing generally non-Hermitian E. Compute its SVD and
  insert the orthogonal projector onto its leading left singular vectors.
  This is a simple full-bond environment test. Repeated open-bond contractions
  are relatively expensive and the result depends on the chosen gauge.
* **Finite KQ / RTM pilot:** left-canonicalize the forward boundary and opposite
  guide. Cache bilinear left-prefix contractions K. Sweep right to left; use
  the right singular vectors of KQ to select the emitted site and absorb QV
  into the remaining core. No Gram matrix or pseudoinverse is needed.

The second construction adapts the local compression idea in
[CT-bMPS, Sec. 2.3](https://arxiv.org/html/2610.06574v1#S2.SS3). It performs one
finite-row update with fixed backward guides. It does **not** implement the
paper's bidirectional contraction-tree iteration or a stationary MP-BP solve.
The paper also reports difficulties with strongly cancelling signed networks;
its positive-model gains do not establish gains for these Pauli networks.

Comparison: 4×4, spatial cap 8, steps 2/6, signed/unsigned, scales 0/1, boundary
and guide caps 4/8/16/32: **32 settings**. All methods contract the same saved
PEPS in each setting, with an independent greedy exact contraction as reference.
This reference does not remove error from earlier spatial simple updates.

The open-bond projector improved corrected accuracy in **3/32** settings and
took **1.12–1.85×** the ordinary row-residual time. The finite RTM pilot improved
it in **6/32** and took **0.89–1.13×** the time. These strict win counts include
near-roundoff differences, so they are not evidence of useful advantage.

The difficult signed six-step zero-noise case is more informative:

| Boundary/guide cap | Ordinary residual relative error | Projector residual | Finite RTM residual |
|---|---:|---:|---:|
| 4 | 105.68% | 123.44% | 112.67% |
| 8 | 33.97% | 51.28% | 78.53% |
| 16 | 3.439% | 14.48% | 11.25% |
| 32 | 0.0350% | 0.3031% | 0.1307% |

None of these pilots is promoted. A small residual after adapting a boundary to
an approximate guide can simply mean the boundary fits that guide. It is not
proof the true suffix contraction is accurate. A stronger next algorithmic test
is to update both messages, monitor independent cuts, and compare at equal total
runtime against increasing the ordinary guide cap.

## 3. Actual batch profiles and a blocked Metal prototype

`profile_ready_boundary.py` records actual same-shape groups inside sequential
boundary splits. The 4×4, six-step, spatial-cap-8 / boundary-cap-32 case produced
22 groups, all batch 4. The 6×6, two-step, spatial-cap-16 / boundary-cap-32 case
produced 185 groups: **128 batch-1, 45 batch-2, and 12 batch-4** groups.

Thus batch-16/32 microbenchmarks do not describe ready work in these runs.
Cross-split batching needs an algorithmic schedule exposing independent updates;
collecting a batch must not change sequential gauge or truncation dependencies.
The larger case also contains QR groups as large as 1026×1024 per sector despite
a nominal boundary cap of 32. Boundary absorption can expand intermediate bonds.
After warming kernels, the larger profile spent 0.948 s in QR and 1.183 s in
SVD factorization out of 2.328 s total. These profiled times include explicit
synchronization and are bottleneck diagnostics, not an uninstrumented speed test.

`apple_blocked_jacobi_experiment.py` explores small Gram-core rotations followed
by batched GEMMs on column panels. It reuses the custom shared-memory Metal
Jacobi kernel for local rotations. This is inspired by the blocked data movement
in [Abdelfattah and Fasi](https://arxiv.org/html/2601.17979v2), but is a prototype
with MLX gather/scatter and host-scheduled rounds, not their optimized CUDA/ROCm
solver or a new fused Metal implementation.

We tested block widths 16/32, outer sweeps 2/4/8/16, and an optional direct
one-sided Jacobi cleanup. Local rotations must keep columns unsorted; sorting
every pair can stall the cyclic block schedule. FP32 Gram cores remain
insufficient for weak singular directions in the captured matrices. Small
reconstruction error coexists with large active-U orthogonality error. Therefore
reconstruction alone must not qualify this path for tensor truncation.

Representative **captured matrices**, batch 16, including host input and output:

| Matrix order | CPU FP64 ms | CPU FP32 ms | Adaptive Metal ms | Blocked + cleanup ms |
|---|---:|---:|---:|---:|
| 128 | 13.01 | 7.58 | 7.96 | 14.01 |
| 256 | 52.25 | 36.26 | 145.09 | 137.49 |

The cleanup restores active-U orthogonality to about 2.5e-6 / 3.8e-6 in these
cases, but removes the attractive speed comparison. Longer Gram-only runs still
have order-one orthogonality errors on captured inputs. Zero/null columns are
not completed into a full isometry. Production dispatch remains unchanged.

The input batches repeat up to eight previously captured matrices; batch 16 is
a controlled kernel workload, not evidence of sixteen simultaneously ready
solver tasks. Synthetic condition-100 matrices were tested separately. Initial
locally sorted and short unsorted experiments are archived as exploratory data;
the refined benchmark uses the corrected unsorted local rotations.

## Reproduction and validation

From the repository root, use the installed Python environment with NumPy,
SciPy, Numba, quimb and symmray. Metal experiments additionally require MLX and
Apple GPU access. Thread limits and writable cache locations used here:

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/pauli_cnot_numba MPLCONFIGDIR=/tmp/pauli_cnot_mpl
export PYTHONPATH=cnot_zne
.venv312/bin/python benchmarks/run_boundary_priorities.py --mode row --output results/priority_implementation/row.json
.venv312/bin/python benchmarks/run_boundary_priorities.py --mode guided --output results/priority_implementation/guided_rtm.json
.venv312/bin/python benchmarks/profile_ready_boundary.py --output results/priority_implementation/ready_batches_warm.json
.venv312/bin/python benchmarks/run_blocked_jacobi.py --output results/priority_implementation/blocked_unsorted.json
.venv312/bin/python benchmarks/run_blocked_jacobi.py --refine --sizes 128 256 --output results/priority_implementation/blocked_refined.json
```

The GPU SVD benchmark reads actual matrices from
`results/exploration_20261013/adaptive_real.npz`; it never substitutes synthetic
matrices when capture is missing. Regenerate that input with:

```sh
.venv312/bin/python benchmarks/explore_adaptive_jacobi.py --deflation-tol 1e-7 --output results/exploration_20261013/adaptive_real.json
```

Tests cover row/bond equivalence with cutoff, short rectangles, exact suffixes,
non-mutating inputs, after-CNOT ratio/Q output, CPU/Metal row execution, symmetry
validity, dense independent environment projectors, dense KQ retained subspaces,
SVD cleanup accuracy, and zero/rank-deficient reconstruction. The regression
command enables all three existing Metal test groups. Final result: **253 passed,
1 skipped, 4 subtests passed** (18.15 s):

```sh
TEST_PEPS_METAL=1 TEST_PEPS_GPU_BOUNDARY=1 TEST_MPS_METAL=1 .venv312/bin/python -m pytest tests cnot_zne/tests -q -W error::RuntimeWarning --maxfail=1
```

Published measurements and source hashes are in
[priority-implementation-20261013.json](apple-silicon/priority-implementation-20261013.json).
