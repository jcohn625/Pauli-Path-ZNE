# Second Apple Silicon optimization batch

13 October 2026; Apple M4 Pro, 24 GiB RAM. This batch improves MPS canonicalization and PEPS gauge scaling relative to the completed [first pass](apple-priority-results.md).

## Retained changes

- `mps_qr.py` calls SciPy's FP64 LAPACK `dgeqrf`/`dorgqr` directly and caches optimal workspace lengths for at most 256 matrix shapes. This removes repeated high-level dispatch and workspace queries around thousands of small sector factorizations. It retains Householder QR, finite-input checks, input ownership, and LAPACK status checks. Other dtypes and empty matrices use SciPy's existing economic QR. The cache holds lengths, not numerical tensors or work arrays.
- CPU PEPS gauge scaling builds each axis/charge broadcasting view once per tensor request. It applies the views directly to block copies, avoiding per-block vector/view setup and the intermediate CPU task list. Multiplication order and output transpose semantics are preserved. Metal scaling remains on its existing kernel path.

The installed quimb's `method='auto'` already selects QR for the two exact outer splits. Merely specifying QR would not improve them. Central truncated splits still use the existing SVD path.

## Whole-solver measurements

| Workload | Before | After | Runtime reduction |
|---|---:|---:|---:|
| MPS, cap/guide 16 | 0.268 s | 0.237 s | 11.8% |
| MPS, cap/guide 64 | 0.633 s | 0.588 s | 7.2% |
| PEPS, spatial cap 8 | 1.389 s | 1.248 s | 10.1% |
| PEPS, spatial cap 16 | 2.237 s | 2.112 s | 5.6% |

MPS: 12 sites, three steps. PEPS: 4×4, two steps, boundary cap 8, environment cap 16, batched CPU gates. All include correlated noise after CNOT layers and residual correction. Results are medians of five warmed runs per variant, alternating old/new order in one process with one numerical-library thread. These are local workload measurements, not universal speedups. The corrected scalar matched exactly in all four cases.

A second independent invocation confirmed reductions of 13.0%, 5.7%, 11.1%, and 5.9%, respectively, again with exactly matching corrected scalars. Both invocations are preserved in the raw data rather than selecting the faster timings.

The baseline is `/tmp/pauli-apple-batch2-baseline`, captured immediately before this batch; it includes the first pass and earlier uncommitted work. Only MPS canonical QR and PEPS gauge scaling differ in production. These gains are additional to the first-pass gains, rather than a comparison to the last Git commit.

## Validation

The full CPU/Metal regression suite passed **196 tests, with 1 optional skip and 4 subtests passed**, treating runtime warnings as errors. Added tests verify QR reconstruction, orthogonality, unchanged input, rank deficiency, strided/Fortran layouts, tiny/large finite values, nonfinite rejection, and float32/complex fallback. PEPS tests compare sparse scaling with independent dense broadcasts, including reordered axes, repeated charge dimensions, unscaled axes, and input ownership.

Separate old/new comparisons at noise scales 0, 1, and 3 found **zero numerical difference** in signed O, unsigned U, ratios, Q±, and residual corrections for both methods with finite truncation caps. MPS parity leakage stayed zero. Existing tests also cover independent dense circuit references, parity restoration, legacy noise placement, and GPU backends.

No new installation requirement or command-line option is needed. FP64 CPU arithmetic, gate/noise ordering, truncation decisions, and residual definitions are unchanged. PEPS residual correction still addresses boundary contraction of the evolved PEPS, not spatial evolution truncation. No overall peak-memory reduction is claimed: fewer temporary Python objects do not establish that measurement.

## Remaining costs

In the new cap-16 MPS profile, center movement takes about 44% of runtime, including about 18% in the QR helper; SVD takes about 15%. Guide construction and replay include these operations, so their totals overlap. Sector discovery and triangular-factor extraction remain visible small-block costs.

In cap-8 PEPS, splitting takes about 40%, gauge scaling about 21%, and direct contraction about 16%. Fusion/unfusion contributes substantially to splitting. These are instrumented profile fractions, not uninstrumented timing predictions. The next priorities are reducing repeated PEPS split-layout work and measuring larger MPS sectors before choosing a new compression algorithm or GPU batching policy.

## Reproduction

- `benchmarks/benchmark_apple_paired.py --baseline /tmp/pauli-apple-batch2-baseline --output results/apple_batch2_paired.json`
- `benchmarks/check_apple_batch2_curves.py --baseline /tmp/pauli-apple-batch2-baseline --output results/apple_batch2_curves.json`
- `benchmarks/benchmark_apple_priority.py --repeats 1 --output results/apple_batch2_profile.json` for profiles; the one-run timings are not the table's benchmark.

Use the project Python with `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1`. The snapshot must be saved before editing; the paired harness switches only its listed methods. [Raw measurements, profiles, curve comparisons, and source hashes](apple-silicon/apple_batch2_data.json) are retained with the report.
