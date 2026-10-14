# Apple Silicon hybrid design

Measured on Apple M4 Pro, 24 GiB unified memory, MLX 0.31.2. The recommended design is an adaptive CPU/Metal executor with reusable shared input buffers. Current small-cap MPS/PEPS workloads should remain CPU-first. This document describes a design and tested prototype, not an integrated solver speedup.

The [GPU tensor-network literature review](gpu-tensor-network-literature.md) extends this design with grouped output reductions, producer/consumer layout planning, QR-based truncation with controlled bond expansion, and longer-term subspace/polar methods. Its proposed experiment order incorporates the published evidence and distinguishes CUDA/TPU results from Apple measurements.

## What the measurements establish

CPU uses FP64, Metal uses FP32 with FP64 output conversion. Timings include input packing/conversion, GPU evaluation, and output conversion. CPU BLAS threads were limited to one; calibrate again for any other threading policy. Medians follow warmup and rotating backend order. They are synthetic-value kernel benchmarks, not solver timings.

| Batch and matrix dimensions | CPU FP64 | Shared Metal FP32 | Interpretation |
|---|---:|---:|---|
| 1 × (32×32 @ 32×32) | 4 μs | 143 μs | Keep on CPU |
| 32 × (64×64 @ 64×64) | 709 μs | 228 μs | Metal 3.1× faster |
| 8 × (256×256 @ 256×256) | 5.592 ms | 0.634 ms | Metal 8.8× faster |
| 16 × (32×4 @ 4×32), captured PEPS shape | 22 μs | 153 μs | Keep on CPU |
| 32 × (64×16 @ 16×8), captured PEPS shape | 45 μs | 200 μs | Keep on CPU |

All 12 sampled real PEPS batch shapes favored CPU matmul over shared Metal. Selection took the three most frequent and five largest aggregate-operation-count groups from each captured PEPS case, deduplicated. This samples the workload; it does not exhaust every shape. The existing direct CPU einsum path was also timed and was slower than CPU matmul for these samples. A shape-aware CPU contraction choice is consequently a nearer-term opportunity than GPU offload for these batches.

Captured MPS (12 sites, 3 steps, cap/guide cap 16) most frequently called QR on 16×4 matrices (3,583 calls). PEPS 4×4, two steps: at cap 8, 81,773 of 86,231 captured sector products had all dimensions at most 32; cap 16 still had 39,778 such products. These counts describe instrumented runs, not a universal distribution. MPS products were not captured by the PEPS instrumentation.

Installed MLX probes rejected GPU QR, SVD, and float64 arithmetic. These are observed capabilities of this installation. Large square-product FP32 relative errors were approximately 8e-8–4e-7; casting outputs back to FP64 does not recover precision. They cannot establish final observable or residual accuracy.

## Memory and scheduling architecture

MLX CPU/GPU operations access the same memory pool, but NumPy-to-MLX construction still copies. A NumPy view of an MLX array can share storage. Accessing pending results also entails evaluation/synchronization. See the official [unified-memory description](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst), [NumPy interoperability](https://ml-explore.github.io/mlx/build/html/usage/numpy.html), and [lazy evaluation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst).

1. Keep authoritative tensors, QR/SVD, gauge bookkeeping, charge labels, and small products on CPU in FP64. Cache immutable contraction plans and sector layouts separately from numerical values.
2. Group independent products by `(batch, m, k, n)` and dtype. Calibrate the complete path, including packing and synchronization. Require a repeatable margin before selecting GPU; unknown shapes use CPU. Do not dispatch solely on bond dimension or FLOP count.
3. Allocate MLX-owned FP32 input arenas with writable NumPy views. Fill these views directly from FP64 tensors, eliminating stacked temporary arrays and a second construction copy. Reuse arenas within a bounded memory budget.
4. Enforce slot states: idle → CPU fill → GPU submitted → resolved → idle. Never refill inputs until their consumer finishes. The prototype implements one slot. A future two-slot queue could overlap independent CPU work with GPU execution; overlap has not been measured and shared memory bandwidth may limit it.
5. Fuse gauge multiplication, diagonal noise scaling, and compatible permutations into a sufficiently large GPU contraction where possible. Small standalone scaling kernels are poor offload candidates. Batch only operations whose dependencies permit it; no reordering across overlapping gates or truncations.
6. Return only the data needed for CPU decomposition. A GPU contraction followed immediately by CPU QR/SVD introduces a synchronization boundary, even on unified memory. Keep larger randomized-SVD products resident when profitable, while accounting for the existing CPU skinny-factor normalization barriers.

The prototype defaults to a 128 MiB allowance for estimated FP32 input/output buffers, not a process-memory cap. FP64 outputs, tensors, guides, library caches, decomposition workspace, and temporary allocations need separate accounting. A production executor should bound the entire live queue and measure peak process memory. Do not reserve a large fraction of the machine's 24 GiB automatically.

## Preserving the noise model and parity

Execution placement must leave transported Z2×Z2 charge labels, gate order, and noise placement unchanged. GPU batches contain compatible numerical sector blocks; CPU metadata retains the charge routing. Single- and two-qubit Pauli generators, including cross-block couplings, continue to use the existing operation schedule. Do not assume the entire correlated channel is one tensor product across disjoint Heisenberg gates.

Preserve the signed and unsigned schedules independently: unsigned transfers are built from absolute elementary transfers, not the absolute value of a fused signed transfer. Residual corrections and ratio/Q outputs must see the same truncation and guide semantics as the CPU implementation.

## Implementation order and acceptance criteria

1. **CPU optimization first:** benchmark matmul versus direct einsum for actual PEPS shapes, then select the cheaper contraction. For MPS, reduce repeated contraction-path planning and Python overhead around small QR sweeps. Validate whole-run gains before changing defaults.
2. **Opt-in hybrid contraction integration:** connect measured large-product groups to the shared-buffer executor. Keep strict FP64 CPU as the default and expose mixed precision explicitly. The current prototype is not connected to solver execution.
3. **Batching and fusion:** collect independent sector work across disjoint updates without changing truncation order. Use persistent structural plans and avoid repeated reshape/packing allocations. Recalibrate after fusion because crossover thresholds change.
4. **Overlap:** only after profitable offload exists, evaluate two slots and independent CPU work between submit/resolve. Compare total solver time and peak memory with the synchronous hybrid path.

Promotion requires warmed repeated full-workload benchmarks over representative caps/depths/noise graphs; signed O, unsigned U, ratio and Q± checks against CPU plus small dense references; residual-correction accuracy; and peak-memory measurement. Mixed precision must meet an explicit observable-error target, not merely a kernel relative-error threshold. Custom GPU QR would be a separate project; forming a Gram matrix to replace QR can square conditioning and is not a safe general shortcut.

## Prototype and reproducibility

`apple_silicon_hybrid.py` provides `SharedBatch` (explicit submit/resolve) and `HybridProducts` (opt-in measured-shape dispatch). It guards in-flight buffer reuse and allocation budgets. Its shared writes are deliberately outside autograd and retained compiled graphs. Four tests passed on this machine, including changing inputs across eight buffer reuse cycles, CPU fallback, budget checks, and GPU dispatch.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv312/bin/python benchmarks/benchmark_apple_shared.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv312/bin/python benchmarks/capture_apple_blocks.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv312/bin/python benchmarks/benchmark_apple_rectangular.py
TEST_PEPS_METAL=1 .venv312/bin/python -m pytest tests/test_apple_shared.py -q
```

Metal requires a local session with GPU access. Raw calibration results: [square batches](apple-silicon/apple_shared_calibration.json), [captured rectangular shapes](apple-silicon/apple_rectangular_calibration.json). Capture output is written to `results/apple_block_shapes.json` and contains shapes/counts, not tensor values.
