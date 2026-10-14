# GPU tensor-network literature and implications for Pauli Path ZNE

Reviewed 13 October 2026. This is a targeted literature review covering GPU/hybrid MPS and DMRG, symmetry blocks, contraction execution and planning, PEPS contraction, low-rank decompositions, precision, and related accelerator methods. It is not an exhaustive bibliography of every tensor-network application. Sources below are research papers, author manuscripts, and official software documentation; newer preprints are identified. Selected full-text sections were checked for the main algorithmic recommendations; other entries are assessed from their abstracts/documentation. No published speedup is a prediction for our M4 Pro.

The principal recommendation is to optimize the **collection of contractions and its memory layout**, then consider changing the decomposition algorithm. Moving individual small products to Metal is only one component. This supplements the [local architecture design and measurements](apple-silicon-hybrid-design.md).

**Evidence closest to our symmetry-aware implementation**

| Reference | Result or method relevant here | Application and limitation |
|---|---|---|
| Menczer & Legeza, *Massively Parallel Tensor Network State Algorithms on Hybrid CPU-GPU Based Architectures* (2023), [paper](https://arxiv.org/abs/2305.05581) | Hybrid scheduling of tensor-network work on CPU/multiple GPUs. | Design the executor around a dependency graph of numerical tasks. Their quantum-chemistry DMRG workload is much larger and has different contractions. |
| Menczer & Legeza, *Boosting the effective performance … via non-Abelian symmetries* (2023), [paper](https://arxiv.org/abs/2309.16724) | Separates symmetry algebra from numerical execution; uses spatially local storage, strided batches, adaptive buffering, and packed products that incorporate reductions. | Especially relevant to PEPS blocks accumulating into the same output. Our Abelian charge rules are simpler; SU(2) recoupling machinery is unnecessary. |
| Menczer et al., *Parallel implementation … achieving a quarter petaFLOPS … DGX-H100* (2024), [paper](https://arxiv.org/abs/2407.07411) | Demonstrates that a carefully organized symmetry-adapted DMRG implementation can use modern GPUs effectively. | Supports the architecture, not a laptop speedup estimate. GPU count, precision hardware, problem sizes, and CPU baseline differ. |
| Jaschke et al., *Benchmarking Quantum Red TEA on CPUs, GPUs, and TPUs* (2024, revised 2025), [paper](https://arxiv.org/abs/2409.03818) | Compares backends, precision, threading, padding, and symmetry. Its Z2-blocked TTN benchmark did not gain from GPU execution at bond dimensions ≤64; manual padding also failed to improve its tested timings. | Closely supports measuring actual block sizes. This is a TTN benchmark, not a universal MPS crossover. Its dense calculations behaved differently. |
| Singh, Pfeifer & Vidal, *Tensor network states and algorithms in the presence of a global U(1) symmetry* (2011), [paper](https://arxiv.org/abs/1008.4774) | Foundational separation of charge structure and numerical degrees of freedom. | Architectural background for cached charge/reshape plans, rather than a GPU benchmark or a new symmetry for our model. |

**Concrete extension: combine products that share an output**

Our `peps_metal_layer.py::contract_direct` currently computes each compatible block product and adds it to a dictionary entry. If an output block receives

\[
C=A_1B_1+A_2B_2+\cdots+A_sB_s,
\]

we can equivalently form

\[
C=[A_1\ A_2\ \cdots\ A_s]
\begin{bmatrix}B_1\\B_2\\\vdots\\B_s\end{bmatrix}.
\]

This is a proposed adaptation of the packed reduction strategy in the non-Abelian hybrid paper, not an implemented optimization. Group only contributions with the same output charge key and compatible free-index order. Concatenation is along the contracted dimension; unrelated outputs stay separate. The possible gain is fewer launches and reductions, plus a larger inner dimension. Packing can erase the gain, so compare ordinary accumulation, reused packed buffers, and a direct fused kernel. Never replace independent products with an unrestricted dense multiply that introduces cross terms.

**Contraction order, data layout, and memory**

| Reference | Relevant contribution | What to explore here |
|---|---|---|
| Pan et al., *Efficient Quantum Circuit Simulation by Tensor Network Methods on Modern GPUs* (2023), [paper](https://arxiv.org/abs/2310.03978) | Reorders indices across connected contractions so expensive operations become GEMMs; studies mixed/extended precision. | Choose each intermediate's output layout for its next consumer. Avoid repeated transpose-and-pack steps in MPS environments and PEPS updates. Its CUDA precision mechanisms require separate Metal work. |
| Springer & Bientinesi, *Design of a high-performance GEMM-like Tensor-Tensor Multiplication*, [paper](https://arxiv.org/abs/1607.00145) | GETT packs sub-tensors within a cache-aware contraction rather than materializing whole transposed tensors. | Useful CPU design foundation for small/skinny blocks and compiled physical-index loops. The paper's CPU results are not Metal results. |
| Gray & Kourtis, *Hyper-optimized tensor network contraction*, Quantum 5, 410 (2021), [paper](https://quantum-journal.org/papers/q-2021-03-15-410/) | Hyper-optimizes contraction trees for general networks. | Search and cache paths for repeated PEPS environments or exact references. Ordinary MPS sweeps have more restricted ordering freedom. |
| Huang et al., *Efficient parallelization of tensor network contraction for simulating quantum computation* (2021), [paper](https://www.nature.com/articles/s43588-021-00119-7) | Index slicing plus contraction-order reconfiguration reduces memory and exposes independent work. | Consider for otherwise oversized PEPS/exact contractions. Slicing can repeat work and make GPU jobs too small; it is not a default optimization for current small MPS blocks. |
| Gray & Chan, *Hyperoptimized Approximate Contraction of Tensor Networks with Arbitrary Geometry*, PRX 14, 011009 (2024), [paper](https://arxiv.org/abs/2206.07044) | Jointly optimizes compression and contraction strategies. | Relevant to future PEPS boundary strategies; changes truncation locations, so residual bookkeeping must change with it. |
| Hérault et al., *Distributed-memory multi-GPU block-sparse tensor contraction for electronic structure* (2021), [paper](https://www.netlib.org/utk/people/JackDongarra/PAPERS/dist-block-ipdps-2021.pdf) | Task-based execution of irregular block-sparse contractions. | Borrow dependency scheduling and locality principles; no need to import a distributed runtime for one Mac. |
| Pan et al., *Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs* (2026 preprint), [paper](https://arxiv.org/abs/2606.01852) | Distribution planning and GEMM-oriented mode ordering as an alternative to relying solely on slicing. | Mode ordering transfers; inter-GPU distribution and reported cluster speedups do not apply to one M4 Pro. |

For the Apple executor, the proposed cost model should include measured compute time, bytes packed/written, allocation, and synchronization. FLOPs alone are insufficient. Cotengra already supports a combined FLOP/write objective, which is a useful starting point, not a calibration for our machine. [Official optimizer documentation](https://cotengra.readthedocs.io/en/stable/advanced.html).

**Decomposition algorithms: potentially larger changes**

| Reference | Relevant contribution | Applicability |
|---|---|---|
| Unfried, Hauschild & Pollmann, *Fast time evolution of matrix product states using the QR decomposition*, PRB 107, 155133 (2023), [paper](https://arxiv.org/abs/2212.09782) | QR-based TEBD truncation with controlled bond expansion; reduces physical-dimension scaling and demonstrates large A100 gains. | Highest-priority algorithmic experiment for larger MPS bonds. It differs from our existing QR canonicalization and randomized SVD. The authors caution about generic gates far from identity. |
| Kohn et al., *Probabilistic low-rank factorization accelerates tensor network simulations of critical quantum many-body ground states*, [paper](https://arxiv.org/abs/1710.01463) | Randomized low-rank factorizations accelerate TEBD/DMRG in tested regimes. | We already have randomized SVD. Next experiments should concern rank fraction, oversampling, iteration count, and avoiding full intermediate construction, not simply adding another RSVD backend. |
| Ganahl et al., *Density Matrix Renormalization Group with Tensor Processing Units*, PRX Quantum 4, 010317 (2023), [paper](https://arxiv.org/abs/2204.05693) | Uses matrix-multiplication-based polar decomposition when distributed QR/SVD were unavailable. | Research option for large-block canonicalization on Metal. It produces an isometric factor and a generally nontriangular bond factor. Requires normalization, convergence/orthogonality tests, rank-deficiency handling, and CPU fallback. Extra iterations likely lose on tiny blocks. |
| Zhang, Yang & Corboz, *Accelerating two-dimensional tensor network contractions using QR decompositions*, PRB 113, L201106 (2026; preprint 2025), [paper](https://arxiv.org/abs/2505.00494) | QR-based CTMRG for C4v-symmetric networks. | Our internal Z2×Z2 symmetry does not imply spatial C4v symmetry. This is not directly interchangeable with finite-PEPS boundary contraction. |
| Zhang & Corboz, *Fast two-dimensional tensor-network contraction via subspace iteration* (July 2026 preprint), [paper](https://arxiv.org/abs/2607.15158) | Builds projectors with subspace iteration and a smaller SVD; recycles converged bases in iPEPS CTMRG. | Investigate analogous reuse only when successive boundary problems vary slowly. Our finite, noisy, signed network lacks the same fixed-point setting; accuracy and convergence must be established anew. |

For a QR-based MPS experiment, expand and truncate parity sectors dynamically rather than freezing their old ranks. Prefer a small direct SVD for the final rank decision initially; a Gram-matrix eigensolve can lose small singular-value accuracy. Elementary CNOT transfers and noisy unsigned maps do not satisfy every assumption of small-step unitary TEBD. Benchmark both signed and unsigned paths. These are our proposed safeguards, not a claimed extension proved by the paper.

Our actual compression residual is `target - left_factor @ right_factor`. In principle that permits testing other factorizations while retaining residual correction, provided every new approximation is represented in the residual and guide evolution remains consistent. A finite guide does not guarantee correction of arbitrary decomposition or roundoff error. In particular, approximate polar canonicalization must not be silently treated as an exact gauge change. Our PEPS correction currently covers contraction of a fixed evolved PEPS, not its spatial evolution truncation.

**Precision, software, and secondary directions**

- Brower et al., *Mixed-precision ab initio tensor network state methods adapted for NVIDIA Blackwell technology via emulated FP64 arithmetic*, [manuscript](https://arxiv.org/abs/2510.04795), [2026 publication](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6c00203): motivates explicit precision/error studies. Its fixed-point FP64 emulation is hardware-specific. It does not demonstrate FP64-quality Metal products or guarantee accuracy of cancellation-sensitive residual corrections.
- Bayraktar et al., *cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science* (2023), [paper](https://arxiv.org/abs/2308.01999), and [cuTENSOR documentation](https://docs.nvidia.com/cuda/cutensor/2.3.0/index.html): useful architecture references for plan reuse, layout-aware contractions, and block-sparse/mixed-precision primitives. These are CUDA libraries, not Apple GPU backends.
- Liao et al., *Differentiable Programming Tensor Networks*, PRX 9, 031041 (2019), [paper](https://arxiv.org/abs/1903.09650): relevant if we later optimize PEPS variationally or differentiate noise parameters. Automatic differentiation is not needed to accelerate our present forward/residual evaluation; it can add memory costs.
- Secular et al., *Parallel time-dependent variational principle algorithm for matrix product states*, PRB 101, 235123 (2020), [paper](https://arxiv.org/abs/1912.06127): demonstrates algorithmic parallelism along an MPS. Adopting it would change the evolution method; it does not authorize parallelizing dependent steps of our current sweep.
- Deverapalli, *High-Performance Block-Sparse Tensor Contraction on GPUs* (2026 master's thesis), [institutional record](https://repository.lib.ncsu.edu/items/0b623de7-9602-43b2-877f-bfa7eebf1783): closely related recent work. Full-text access was blocked by the repository's bot check in this review, so its detailed kernel and performance claims are not used as evidence here.

The existing shared-buffer prototype remains useful infrastructure. MLX supports [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), while [NumPy interoperability](https://ml-explore.github.io/mlx/build/html/usage/numpy.html) explains the shared views used in our prototype. Neither removes precision conversion or synchronization costs. Installed MLX 0.31.2 locally rejected GPU QR/SVD and FP64 operations; that is a version-specific observation. TPU polar methods illustrate a possible workaround, not proof that it is faster or sufficiently accurate here.

**Revised experiment order**

1. Establish whole-run FP64 baselines with fixed contraction plans and shape-dependent CPU matmul/einsum/Accelerate selection. Record packing, contraction, decomposition, guide construction, and residual costs separately.
2. Test grouped same-output PEPS products and producer/consumer layout planning. Include allocation/packing in timings and preserve exact charge routing and gate/truncation order.
3. At larger bonds, compare existing SVD/RSVD against sector-aware QR truncation with controlled expansion. Include sparse sector populations, abrupt rank changes, long-range noise, and unsigned transfers. Require equal observable accuracy, not merely equal caps.
4. Batch compatible work across independent scales/observables or disjoint jobs; dispatch to Metal only where measured total cost wins. Never combine tasks with unmet dependencies.
5. Investigate subspace recycling for larger boundary problems and polar canonicalization for large, well-conditioned sectors as separate experimental algorithms.

For every candidate record warm and cold time, peak memory, precision mode, local residual norms, final signed O/unsigned U, ratio and Q±, and error against dense small-system and converged CPU references. A ratio near a zero denominator needs an absolute-error assessment as well. Preserve the existing distinction between absolute elementary transfers and the absolute value of a fused signed transfer. Published CUDA/TPU speedups are motivation to test these designs, not expected Apple speedups.
