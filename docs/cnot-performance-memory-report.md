# CNOT-noise performance and memory pass

Date: 2026-10-13. Comparison is against the local implementation immediately before this pass (the previously optimized fusion/scheduling version). Module SHA-256 hashes and raw observations are recorded in `cnot-performance-memory-data.json`. No GitHub publication was performed.

## Default-path timing

Five timed runs after warm-up, one thread for numerical libraries, residual correction enabled. MPS: 12 qubits, 3 steps, state/guide caps 16. PEPS: 4×4, 1 step, spatial/boundary caps 4, environment cap 8. Rates are (0.001,0.002,0.003), with pair cases adding 0.0001 for each of nine Pauli generators on each nearest-neighbor edge.

| Engine | Noise | Before seconds | After seconds | Speedup | Absolute change in corrected result |
|---|---|---:|---:|---:|---:|
| MPS | single | 0.0557 | 0.0480 | 1.16× | 3.82e-17 |
| MPS | single + pair | 0.4027 | 0.3609 | 1.12× | 8.46e-17 |
| PEPS | single | 0.0582 | 0.0605 | 0.96× | 0.00e+00 |
| PEPS | single + pair | 0.3663 | 0.3232 | 1.13× | 2.02e-17 |
| PEPS-batched | single | 0.0611 | 0.0577 | 1.06× | 0.00e+00 |
| PEPS-batched | single + pair | 0.3513 | 0.3277 | 1.07× | 4.27e-17 |

The default-path changes preserve the tested finite-cap answers to roundoff. Single-noise default PEPS is essentially unchanged/slightly slower in this sample; the diagonal-noise specialization targets correlated noise. These are small exploratory CPU measurements, not a guaranteed speedup for every size/cap.

Implemented: direct blockwise diagonal PEPS scaling; cached sector/contraction layouts with no cached numeric state; full-SVD residual skipping only when no positive singular values are discarded; compact diagonal/permutation operation storage; optional budgeted MPS snapshots and streamed diagnostics. Generic/Metal contraction paths remain supported.

## Retained guide memory

Eight qubits, one step, state cap 8, guide cap 16, correlated noise, signed observable Z0 Z1. Times are medians of three runs. This shallow example has only five residual contractions after 53 are skipped.

| Guide-cache setting | Peak retained array bytes | Seconds | Corrected value |
|---|---:|---:|---:|
| None MiB | 209400 | 0.0422 | -0.8274777266384964 |
| 0.0625 MiB | 47904 | 0.0293 | -0.8274777266384964 |
| 0.0 MiB | 0 | 0.0233 | -0.8274777266384964 |

`None` means the original square-root checkpoint schedule. The 0.0625 MiB option reduces retained guide arrays by about 77% in this example. The zero-cache path is fast here because very few residuals need a guide; it can be much slower in heavily truncated runs. The budget excludes active working/forward states, decomposition workspace, operation data, Python metadata and other caches, so these numbers are **not total peak RSS**. Diagnostics can be omitted or streamed separately.

## Experimental diagonal-noise MPO

Eight qubits, two steps, state cap 8, guide cap 32, phi=0.23, scales=1, single rates (0.001,0.002,0.003), pair rate 0.0005. Exact signed and unsigned references use dense elementary evolution. The long-range star has edges (0,5),(0,6),(0,7). Three timing repeats after warm-up.

| Noise graph | Transfer | Strategy | Seconds | Absolute corrected error | Retained guide bytes | SWAPs |
|---|---|---|---:|---:|---:|---:|
| nearest neighbor | signed | local | 0.1207 | 1.89e-15 | 1512200 | 0 |
| nearest neighbor | signed | mpo | 0.1502 | 1.11e-16 | 929632 | 0 |
| nearest neighbor | unsigned | local | 0.1203 | 1.55e-15 | 1512200 | 0 |
| nearest neighbor | unsigned | mpo | 0.1489 | 1.11e-16 | 929632 | 0 |
| long range star | signed | local | 0.2213 | 6.68e-05 | 2401080 | 168 |
| long range star | signed | mpo | 0.1570 | 5.13e-08 | 1022016 | 0 |
| long range star | unsigned | local | 0.2224 | 4.76e-05 | 2401080 | 168 |
| long range star | unsigned | mpo | 0.1577 | 2.21e-08 | 1022016 | 0 |

The MPO needs bond 4 and 39,936 coefficient-array bytes across the schedule in both graph examples. It is slower for nearest-neighbor noise but faster and more accurate at these fixed caps for the star. Retained guides are smaller, but this does not establish a reduction in total peak memory: applying an MPO temporarily enlarges MPS bonds. The MPO stays opt-in; `local` remains the default. A frontier-size guard rejects an MPO exceeding `noise_mpo_max_bond` before allocation.

## Usage and validation

See `cnot-parity-methods.md` for the Python and CLI options. Reproduce timings with `benchmarks/benchmark_cnot_optimizations.py` (optional `--module-dir` selects the saved baseline) and `benchmarks/benchmark_cnot_memory_mpo.py`, using the project Python environment and single-thread numerical-library settings.

The full CPU/Metal suite passed 146 tests with one optional skip and four subtests. A subsequently added CLI smoke test also passed. New tests cover signed/unsigned dense references, MPO residual telescoping, diagonal-block versus generic contraction, exact byte-budget limits, streamed diagnostics, compact adjoints, and retention of residuals from numerical-cutoff discards.
