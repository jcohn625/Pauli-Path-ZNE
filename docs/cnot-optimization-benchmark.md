# CNOT-noise optimization timing check

Comparison against commit `6664cc5`, on the same local CPU with single-thread numerical libraries. Each case has one warm-up and five timed runs; the table reports medians. Residual correction is enabled.

MPS: 12 qubits, 3 steps, bond/guide caps 16. PEPS: 4 by 4, 1 step, spatial/boundary caps 4, environment cap 8. Single rates are (0.001, 0.002, 0.003); correlated cases also use rate 0.0001 for each of the nine generators on every nearest-neighbor edge.

| Engine | Noise | Before (s) | After (s) | Speedup |
|---|---|---:|---:|---:|
| MPS | single | 0.488 | 0.066 | 7.42x |
| MPS | single + pair | 0.769 | 0.494 | 1.56x |
| PEPS | single | 0.143 | 0.070 | 2.04x |
| PEPS | single + pair | 0.485 | 0.460 | 1.05x |
| PEPS-batched | single | 0.151 | 0.074 | 2.03x |
| PEPS-batched | single + pair | 0.461 | 0.438 | 1.05x |

These are fixed-cap timings, not equal-accuracy timings. Fusion changes truncation locations. For example, single-noise PEPS at spatial cap 4 changes from 0.0010907104 to 0.0020543903; its boundary residual does not correct this spatial truncation. Neither is certified converged by this timing test. Exact small-system signed/Q checks and MPS residual checks pass. Correlated-noise PEPS has little measured improvement in this workload.

Shared-routing example: an eight-site chain with noise edges (0,5), (0,6), (0,7), one step, drops from 180 independent-routing SWAPs to 84. This measures operation count, not wall-clock speed.

Reproduction script: `benchmarks/benchmark_cnot_optimizations.py`. Pass `--module-dir` pointing to the three `cnot_parity*.py` files extracted from the baseline commit for the old implementation. Raw timings and finite-cap answers are included in `cnot-optimization-timings.json`.
