# First Apple Silicon optimization pass

13 October 2026; Apple M4 Pro, 24 GiB RAM. This implements the first priorities in the [combined roadmap](apple-optimization-roadmap.md). Production arithmetic remains FP64 on CPU. No new dependencies, user options, gate reorderings, truncation rules, or GPU dispatch thresholds were introduced.

**Changes retained**

- Fixed matrix-product sequences for MPS bond absorption, site joining, physical transfers, overlap environments, and residual environments. The environment order is selected from actual matrix dimensions without a general einsum-path search. Bilinear semantics are preserved.
- Direct integer indexing for already-known MPS sector row/column arrays, avoiding repeated `np.ix_` setup around QR/SVD.
- Direct PEPS matrix multiplication and cached structural plans. Plan construction computes free/contracted dimensions once per block using Python integer products, rather than repeating NumPy reductions per compatible block pair.
- CPU gauge scaling skips unscaled axes without allocating all-ones vectors or multiplying by them. The Metal scaling path is retained.

**Alternating whole-solver measurements**

| Workload | Before | After | Speedup | Runtime reduction |
|---|---:|---:|---:|---:|
| MPS16 | 0.414 s | 0.264 s | 1.57× | 36.2% |
| MPS64 | 0.789 s | 0.616 s | 1.28× | 22.0% |
| PEPS8 | 2.024 s | 1.378 s | 1.47× | 31.9% |
| PEPS16 | 2.992 s | 2.232 s | 1.34× | 25.4% |

MPS: 12 sites, three steps, matching state/guide caps 16 or 64. PEPS: 4×4, two steps, spatial caps 8 or 16, boundary cap 8, environment cap 16, batched CPU updates. All cases include single-qubit and nearest-neighbor pair noise after CNOT layers and residual correction.

Maximum absolute change in the corrected scalar across these cases: 3.19e-16. The separate unsigned MPS comparison also agreed to roundoff.

[Raw timings, profiles, source hashes, and product experiments](apple-silicon/apple_priority_data.json).

**Experiments not promoted**

Captured 329 actual PEPS contraction requests per cap (4, 8, 16), spanning a 4×4, two-step correlated-noise evolution. Replayed every candidate on the same captured values and alternated timing order. Same-output packing and per-request reshape dictionaries were slower than direct products. Only 112 output groups per case had multiple contributions: 1.46% of groups at cap 4 and 0.91% at caps 8/16. The literature's grouped-reduction approach remains relevant to other workloads, but these cases expose little opportunity.

Accelerate BLAS was competitive and slightly faster in a separate alternating whole-PEPS comparison (about 1–2% versus matmul). That small margin, and the need to account for dispatch, dtype/layout handling, and library builds, does not justify a new automatic backend rule yet. Both BLAS implementations are still represented in the benchmark. Numerical differences in replay were at roundoff scale.

**Validation and interpretation**

The final CPU/Metal regression run passed **160 tests, with 1 optional skip and 4 subtests passed**. New tests compare fixed MPS contractions with independent einsum expressions for rectangular, strided, and complex bilinear inputs. Existing tests cover dense elementary-circuit references, parity restoration, correlated/long-range noise, residual corrections, legacy Heisenberg-noise behavior, ratio/Q workflows, and Metal backends.

Timing comparisons use the local source snapshot immediately before this pass, saved at `/tmp/pauli-apple-priority-baseline`. The rest of that snapshot includes earlier uncommitted optimizations; this is not a comparison to the last Git commit. Raw data record source hashes. Independent-process baselines were repeated, then the main cases were checked by alternating the old/new contraction methods within one process, with unchanged schedules and solver APIs. Each variant had a warmup and five timed runs. Numerical libraries used one thread. These are representative local measurements, not confidence intervals or a universal speedup guarantee.

The recorded process memory high-water marks accumulate across cases and include imports/caches. They do not establish a per-case peak-memory reduction. Removing temporary work is useful, but no overall memory-saving percentage is claimed.

**Remaining bottlenecks**

In the final cap-16 MPS profile, canonical-center movement accounts for about 52% of profiled runtime; QR itself is about 24%, and SVD about 13%. Guide building/replay also performs those sweeps, so these categories overlap. The next target is small-sector QR/canonicalization overhead, followed by a separate larger-bond comparison of QR-based truncation and existing SVD/RSVD.

For cap-8 PEPS, tensor splitting is about 36% and gauge scaling about 30% of profiled runtime. Direct contractions, including their structural work, are now about 14%. The next low-level target is fused scaling/packing and split overhead. Large GPU batches and alternate truncation algorithms should be judged against this faster CPU baseline.

Reproduction scripts: `benchmarks/benchmark_apple_priority.py` records whole-run values, timings, source hashes and selected profiles; `benchmark_apple_paired.py` performs the alternating-method check; `benchmark_apple_product_strategies.py` captures/replays real block requests. Use `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1` and the project Python environment. The paired script's baseline must be the specific pre-pass snapshot, because only the changed numerical methods are switched.
