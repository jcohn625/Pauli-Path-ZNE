# Third Apple Silicon optimization batch

13 October 2026, Apple M4 Pro, 24 GiB RAM. Baseline: the completed second batch, saved at `/tmp/pauli-apple-batch3-baseline`. Numerical libraries use one thread. This batch tests PEPS split layouts, profiles larger MPS bonds, and begins custom Metal compute-shader experiments.

## PEPS split layout experiment

`peps_split_layout.py` is an experimental adapter, not a production backend. It caches fused-sector destination slices and unfusion slice/reshape instructions, while delegating QR/SVD to the original quimb/symmray routines. Each cache is limited to 512 structural plans and retains no numerical tensor values. It uses the pinned symmray 0.4 metadata interface.

Five alternating warmed whole-solver samples per variant gave:

| PEPS spatial cap | Existing implementation | Cached split prototype | Change |
|---|---:|---:|---:|
| 8 | 1.327 s | 1.395 s | 5.1% slower |
| 16 | 2.136 s | 2.311 s | 8.2% slower |

Both cases are 4×4, two steps, boundary cap 8, environment cap 16, correlated noise after CNOT layers, and residual correction. Corrected scalars matched exactly. Plan construction and cache-key work offset the reduced slice/reshape overhead. The adapter remains disabled; the existing PEPS path is retained.

One API detail matters to any future split replacement: quimb `Tensor.split` defaults to relative singular-value cutoff, whereas `array_split` defaults to relative discarded squared weight. The adapter explicitly sets `cutoff_mode='rel'`; an initial implementation exposed this mismatch in the long-range dense-reference test and was corrected before the reported measurements.

## Larger MPS workload

For 16 sites, four steps, observable sites (3,10), correlated noise after CNOT layers, and matching state/guide caps:

| Bond cap | Uninstrumented baseline time | Full SVD time in profile | QR time in profile |
|---|---:|---:|---:|
| 64 | 1.356 s | 0.52 s | 0.17 s |
| 128 | 4.265 s | 1.69 s | 0.45 s |
| 256 | 16.135 s | 5.27 s | 1.45 s |

The timing column is a single warmed observation for workload characterization, not a paired speed comparison. Profile times include instrumentation and must not be divided by that column to infer percentages. All three runs had zero parity leakage. Corrected values converge to approximately -0.001487812174174.

**None of these runs called randomized SVD.** The current requested sketch rank is `cap+16` per sector, clipped to the sector dimensions. Many dominant sector matrices are cap×cap, so the solver takes full SVD. A future sector-adaptive low-rank algorithm needs global rank allocation and a bound or fallback for missed singular directions; simply dividing the cap by four can change truncation quality.

At cap 256, the profile also spends 3.69 s in site joining and 2.02 s in right-bond absorption. These are large dense products even though QR/SVD work is separated into parity sectors. This motivated the additional CPU BLAS comparison below.

## Apple Accelerate bond products

The local NumPy uses OpenBLAS, while SciPy uses Apple Accelerate. The candidate routes large real FP64 site-joining and bond-absorption products through SciPy `dgemm`. Transposing the operand order yields C-contiguous output without an extra output copy. Small products, other dtypes, and unrecognized library/platform configurations retain NumPy.

The production policy is restricted to Apple Silicon macOS with this OpenBLAS/Accelerate pairing and matrix dimensions m, k, n all at least 64. This is a CPU matrix-size rule, not a GPU bond-cap threshold. Validation and paired policy measurements are recorded in the raw data.

The final selective policy, including residual bond products, gave these whole-MPS results on the 16-site/four-step workload:

| Cap | NumPy baseline | Selective Accelerate | Runtime reduction |
|---|---:|---:|---:|
| 16 | 0.439 s | 0.447 s | -1.8% (dispatch overhead) |
| 64 | 1.389 s | 1.228 s | 11.6% |
| 128 | 4.240 s | 3.388 s | 20.1% |
| 256 | 15.715 s | 10.285 s | 34.6% |

This confirmation uses a warmup and two timed runs per variant in alternating order. Maximum corrected-scalar difference was 6.94e-18. The preceding three-sample experiment that routed all main evolution bond products through Accelerate found reductions of 12.1%, 18.8%, and 35.0% at caps 64, 128, and 256, but a 4.0% regression at cap 16. The selective policy reduces that small-workload penalty; it does not eliminate dispatch overhead. Both experiments are retained. These local observations do not establish a universal cutoff for every Apple chip or workload.

## Custom Metal shader experiment

`apple_metal_ragged.py` contains a working Metal compute shader through MLX. A single dispatch handles differently shaped block products and sums multiple contributions to each output. It is deliberately a simple scalar baseline rather than a tuned, tiled matrix kernel. Production solvers do not select it.

Each row below replays 23 real PEPS contraction requests sampled throughout evolution. Median of five runs, compilation excluded:

| Cap | CPU FP64 | Shader including packing/readback | Shader with prepared resident inputs |
|---|---:|---:|---:|
| 8 | 0.734 ms | 6.598 ms | 4.940 ms |
| 16 | 0.990 ms | 6.667 ms | 4.583 ms |

Resident timing still includes Python dispatch and synchronization per request; it is not raw device execution time. End-to-end shader costs are about 9.0× and 6.7× CPU costs. Maximum absolute product differences were 1.14e-7 and 1.80e-7 from FP32 arithmetic. The experiment validates block products, not whole-solver mixed-precision observables.

These results reject this shader at the current request granularity. They do not rule out custom shaders for larger independent batches, persistent device data, tiled matrix kernels, or operations fused across multiple stages. There is already a custom Metal gauge-scaling/packing kernel in `LayerEngine`; replacing every small CPU call with a shader is not the target.

The next shader experiments should prioritize:

1. Persistent GPU batches of independent observables/noise scales, with gauge scaling, parity gather/scatter, and packing fused into the consumer's matrix layout. Measure total solve time including all synchronization.
2. Larger dense MPS site joining, with physical Pauli permutation/sign or diagonal noise fused into the output. Preserve the operation and truncation schedule. Compare against the improved FP64 CPU baseline and tuned MLX matrix multiplication.
3. Batched sector QR or polar orthogonalization only after enough independent work can remain resident. Check rank deficiency, orthogonality, reconstruction, and residual effects explicitly.

MLX's [custom Metal kernel API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) supports this approach. Its default contiguous-input handling can introduce copies; use the documented stride-aware option only with explicit indexing and tests. Experiments here use installed MLX 0.31.2, rather than assuming every feature in the newer online documentation is available.

## Reproduction and evidence

- `benchmarks/benchmark_apple_batch3.py --kind peps --output results/apple_batch3_peps.json`
- `benchmarks/benchmark_apple_batch3.py --kind mps --output results/apple_batch3_mps.json`
- `benchmarks/benchmark_mps_blas.py --policy --output results/apple_batch3_blas_policy.json`
- `benchmarks/benchmark_metal_ragged.py` (requires working Metal device access).

Use the project Python environment with `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1`. [Raw timings, shapes, profiles, and source hashes](apple-silicon/apple_batch3_data.json) accompany this report. No overall peak-memory reduction is claimed.

The full CPU/Metal suite passed **212 tests, 1 optional skip, and 4 subtests** with runtime warnings treated as errors. New checks cover cached splits against quimb, repeated/strided/ragged shader products with multiple contributions per output, and large FP64 BLAS dispatch including ownership, other dtypes, and NumPy fallback. The production PEPS implementation is identical to the pre-batch snapshot.

`benchmarks/check_apple_batch3_curves.py` additionally compares the enabled and disabled dispatch on a 12-site, three-step MPS with state/guide cap 64 at noise scales 0, 1, and 3. Signed/unsigned observables, ratios, Q±, and residual corrections matched exactly in that comparison, with zero parity leakage.
