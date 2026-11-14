# CPU Numba kernels for symmetry-resolved tensor networks

Measured on Apple M4 Pro, 24 GiB unified memory, 14 October 2026. Python 3.12,
NumPy 1.26.4, SciPy 1.17.1, Numba 0.60.0. All measurements use one CPU thread
per numerical library, with numerical workloads run sequentially.

## Implementation

`cpu_numba_kernels.py` contains cached, strict floating-point CPU kernels. They
use neither `fastmath` nor internal parallel loops. BLAS/LAPACK still perform
large contractions, QR and SVD.

- **MPS gathering:** copy charge-allowed matrix entries into sector matrices.
- **MPS reconstruction:** write all retained columns/rows for each charge sector
  in one compiled call, including the singular-value multiplication.
- **Global rank selection:** merge descending sector spectra, preserving the
  existing global bond cap, stable tie ordering, and relative cutoff. Positive
  discarded values still trigger residual correction, including tiny values
  below the numerical cutoff. This is not independent truncation per sector.
- **Experimental transfer:** fuse physical permutations/signs or diagonal
  two-site noise into one output allocation. Operation ordering is unchanged.
- **PEPS scaling/packing:** preserve the order of gauge
  multiplications and write directly into the desired contiguous axis order.
  The second implementation prepares weight maps and activity once per tensor.
  Large blocks, non-FP64 data and unsupported tensor ranks retain the reference path.

MPS integration targets `FrameMPS`, the transported-parity after-CNOT solver.
It does not rewrite the historical Heisenberg-noise MPS solver. PEPS scaling is
integrated into the CPU `LayerEngine`; it does not replace all quimb/symmray
object management or generic contractions. CPU work shared by a hybrid MPS run
can also use these kernels; Metal shaders have not changed.

## Benchmark procedure

Whole-solver timings include evolution and residual correction. Each variant is
warmed once, then timed in alternating order. The final MPS confirmation uses
five samples per variant and compares against an unchanged source snapshot
from before this implementation. Exploratory MPS ablations use three samples.
Every MPS case has 16 sites, four steps, observable sites (3,10), equal state and
guide caps, and pair rate 0.001 per Pauli generator after each CNOT layer.

The paired PEPS cases use two steps and the `batched-cpu` gate backend, CPU BLAS
boundary contraction, residual correction, and pair rate 0.001. The 4x4 case
has spatial cap 8 and boundary/guide caps 16; the 6x6 case has spatial cap 16
and boundary/guide caps 32.

Kernel microbenchmarks include Python dispatch, output allocations and, for
selection, concatenation/offset preparation. They exclude compilation. Scaling
microbenchmarks require packed output on both sides. The full reference solver
can retain transpose views, making those microbenchmarks insufficient to select
production behavior. First-call observations include import/cache/setup work;
they are not isolated measurements of JIT compilation.

## MPS results and enabled policy

The enabled policy combines sector gathering, global singular-value selection,
and SVD-factor reconstruction. Fused CNOT/noise transfer remains experimental.

| State/guide cap | Previous CPU implementation | Enabled Numba policy | Less elapsed time |
|---|---:|---:|---:|
| 16 | 0.443 s | 0.413 s | 6.8% |
| 64 | 1.287 s | 1.160 s | 9.8% |
| 128 | 3.580 s | 3.303 s | 7.7% |
| 256 | 10.707 s | 10.143 s | 5.3% |

The reported approximate, correction, and corrected scalars matched exactly
at every cap; parity leakage was zero. These are local measurements, not a
universal prediction for other circuits, bond distributions, or CPUs.

The initial individual-kernel ablation found reconstruction the most useful
single candidate. At cap 128, its whole-solver time was 3.449 s versus 3.664 s
for the unchanged reference. Gathering alone took 3.472 s, selection 3.590 s,
and transfer 3.634 s. The disabled new reference branches took 3.710 s; thus
comparisons against an unchanged snapshot are more conservative than against
only the new code with flags disabled. Small differences should not be treated
as precise isolated speedups from three samples.

## PEPS results and enabled policy

The initial per-block wrapper was slower by 6.2% on 4x4 and 2.6% on 6x6.
Preparing the gauge-vector maps and activity outside the block loop reversed
that result. The final policy additionally requires every block in the tensor
to be FP64 and at most 4096 elements, with at most five tensor axes. Other cases
retain the original NumPy path. This is a conservative measured-size guard,
not a universal crossover estimate.

The final five-sample confirmation against the unchanged CPU snapshot gave:

| Lattice / spatial cap | Previous CPU implementation | Enabled Numba policy | Less elapsed time |
|---|---:|---:|---:|
| 4x4 / 8 | 1.194 s | 1.116 s | 6.5% |
| 6x6 / 16 | 3.890 s | 3.563 s | 8.4% |

These use `gate_backend='batched-cpu'`. Approximate values, residual corrections,
and corrected values matched exactly. The earlier prepared-wrapper comparison
showed 3.0% and 6.4% reductions, respectively; timings vary between batches.
All raw samples are retained rather than selecting the fastest batch.

A separate three-sample check of the default `gate_backend='quimb'` also
improved: 4x4 went from 1.228 s to 1.171 s (4.6% less time), and 6x6 from
4.028 s to 3.883 s (3.6%). In this mode only the custom CPU portions use the new
scaling kernel; generic quimb gate updates retain their existing implementation.
The reported scalars again matched exactly.

## Kernel observations and limitations

In the final synthetic microbenchmark, 64x64 sector gathering was 4.1x faster,
merging four length-64 spectra was 7.1x faster including preparation, and
reconstructing the corresponding selected factors was 10.7x faster. Those local
speedups do not describe the full solve, which still spends substantial time in
contractions and decompositions.

Scaling/packing was 1.6--2.6x faster for the tested 16--4096-element blocks, but
slower at 65536 elements. This motivated the production guard. The fused
permutation/noise kernel did not show a compelling whole-solver improvement and
remains disabled. The extra benchmark implementation is retained for future
batching experiments.

This pass does not add fully block-sparse MPS storage, change tensor bond caps,
or establish an overall peak-memory reduction. It preserves the existing
operation schedule, symmetry frames, signed/unsigned definitions, and residual
correction semantics.

## Usage and reproduction

No new dependencies are required: Numba is already pinned in
`requirements-mps.txt`. Run through the existing CPU CLI. Set
`PAULI_CPU_NUMBA=0` before starting Python to disable the production Numba policy.
JIT compilation or cache loading can increase first-call latency; steady-state
measurements below exclude that initialization. Tests and benchmarks may toggle
individual internal switches; these switches are process-wide and should not
be changed during concurrent solves.

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/pauli_cnot_numba
export MPLCONFIGDIR=/tmp/pauli_cnot_mpl PYTHONPATH=cnot_zne
.venv312/bin/python benchmarks/benchmark_cpu_numba.py --kind micro \
  --output results/cpu_numba/micro.json
.venv312/bin/python benchmarks/benchmark_cpu_numba.py --kind mps \
  --caps 16 64 128 256 --variants reference policy --repeats 5 \
  --output results/cpu_numba/mps_policy.json
.venv312/bin/python benchmarks/benchmark_cpu_numba.py --kind peps \
  --variants reference policy --repeats 5 --output results/cpu_numba/peps_policy.json
```

By default `reference` runs the current code with the Numba kernels disabled.
To compare against an earlier unchanged checkout instead, pass `--baseline DIR`
containing its `cnot_parity_mps.py` and `peps_metal_layer.py`. The original local
snapshot for this report is `/tmp/pauli-numba-baseline`; its source hashes are
recorded in the accompanying data. No Metal access is needed for these runs.

## Validation and archived evidence

The CPU suite passed **245 tests and four subtests**, with 47 optional-backend
or other explicitly skipped tests. Three additional tests then verified that
large, FP32, and complex PEPS blocks use the reference fallback. Five existing
Metal/hybrid integration checks also passed with explicit device access. Thus
248 CPU test cases plus five Metal/hybrid checks passed in this validation pass.

Checks cover strided inputs, axis permutations, input immutability, zero spectra,
empty sectors, tied singular values, cutoff behavior, forbidden-sector zeros,
long-range routed noise, and complete signed/unsigned ratio/Q workflows with
residual contributions. Existing independent dense-reference tests also passed.
Runtime warnings were treated as errors.

The [archived data](apple-silicon/cpu-numba-20261014.json) contains all measured
samples, version metadata, baseline hashes, and final source hashes. Local logs
and raw files are in `results/cpu_numba/`. Regenerate the compact archive with
`python benchmarks/summarize_cpu_numba.py` after running the corresponding
experiments. The archive includes initial rejected variants as well as the
final policies.
