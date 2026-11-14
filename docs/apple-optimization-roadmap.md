# Apple Silicon optimization roadmap

This combines the local optimization proposals and the [GPU tensor-network literature review](gpu-tensor-network-literature.md). Ranking follows measured whole-solver costs, expected benefit, and the amount of numerical change required. Work started on 13 October 2026. Research candidates advance only when timing and accuracy comparisons support them.

| Priority | Work | Status |
|---|---|---|
| 1 | Remove repeated MPS contraction planning; select efficient CPU products | Implemented fixed contractions in `mps_contractions.py` and matrix products in PEPS. Signed, unsigned, legacy, and residual paths are included in validation. |
| 2 | Reduce allocation/layout overhead around numerical work | Removed CPU all-ones gauge-vector work, repeated MPS indexing setup, and NumPy reductions for PEPS metadata. Batch 2 reuses PEPS gauge broadcasting views by axis/charge and avoids CPU block-task lists. More extensive producer/consumer layout changes remain candidates. |
| 3 | Pack/group parity blocks, reuse reshapes, compare Accelerate | PEPS grouping/reshape dictionaries were slower and remain disabled. Batch 3 found a substantial MPS gain from Accelerate for large FP64 site-joining/bond-absorption products; dispatch is limited to measured matrix sizes and the Apple Silicon OpenBLAS/Accelerate library pairing. |
| 4 | Improve large-bond compression and canonicalization | Batch 2 reduces MPS QR wrapper overhead. Batch 3 profiled caps 64/128/256: full SVD remains important, and no RSVD calls occur in these cases. Sector-adaptive rank allocation/controlled expansion remains an algorithmic experiment, with global truncation-quality checks required. |
| 5 | Batch independent work and fuse profitable CPU/GPU operations | Shared-buffer and custom ragged Metal shader prototypes are available. The batch-3 shader loses at current PEPS request sizes, even with prepared resident inputs. Next: persistent independent batches and fusion of packing/scaling with larger consumers. Require a crossover including synchronization and numerical validation. |
| 6 | Reuse dominant subspaces in repeated boundary/compression problems | Research candidate; not safe to assume slowly changing subspaces across elementary CNOTs or truncations. |
| 7 | Matrix-multiplication-based polar canonicalization on Metal | Research candidate for sufficiently large sectors. Requires rank-deficiency, reconstruction, and orthogonality checks; approximation errors must be accounted for. |
| 8 | Alternative PEPS contraction algorithms, slicing, custom SME/Metal kernels, broader parallel evolution | Use when the workload justifies an algorithm/backend change. Infinite-PEPS CTMRG and TDVP results are not direct replacements for our finite noisy evolution. |

The first changes preserve FP64 CPU arithmetic and the operation/truncation schedule. No GPU bond-dimension threshold has been introduced. A faster kernel alone does not qualify a default change; whole-run time and observable accuracy decide promotion. The packed-product experiment is retained as a benchmark, not a production backend.

For every algorithmic change, compare signed O, unsigned U, ratio and Q±, parity leakage, residual corrections, and memory as well as runtime. Preserve the distinction between unsigned elementary transfers and the absolute value of a fused signed transfer. MPS guide errors and PEPS spatial truncation errors retain their existing limitations.

See the accompanying `apple-priority-results.md` for measurements, validation, and remaining bottlenecks from the first implementation pass.

The [second batch report](apple-batch2-results.md) records the additional QR and gauge-scaling gains against the completed first pass. PEPS outer splits already select QR automatically in the installed quimb version; a method-option change is not an optimization. Remaining PEPS split costs include block fusion/unfusion as well as decomposition.

The [third batch report](apple-batch3-results.md) records larger-MPS profiles, Accelerate bond products, and the PEPS split-layout and custom-shader experiments. The split cache was numerically correct but slower and remains disabled.

## Deep exploration update

The [October research report](apple-deep-exploration.md) and [message-passing literature map](peps-message-passing-literature.md) now supersede the prospective status of several entries above. New experiments cover adaptive/shared-memory Jacobi, CPU worker pools and CPU/GPU overlap, hybrid randomized SVD, GPU polar canonicalization, finite Bethe estimation, and row-level residual telescoping. None changes the production backend default.

The [priority implementation report](apple-priority-implementation.md) records the next completed pass: row telescoping is now the high-level PEPS residual default, with bond diagnostics still available. Two finite environment-guided pilots and a blocked Gram/Jacobi Metal prototype were implemented and tested, but did not justify production adoption. Actual ready PEPS groups in the measured runs contained only 1–4 matrices. The next algorithmic experiment is bidirectional message refinement with independent-cut checks; the next hardware target is the large QR/SVD work exposed by warmed solver profiles, with complete singular-vector validation.


The [hybrid/refinement pass](apple-hybrid-refinement-results.md) is now complete:
`metal-hybrid` is an opt-in PEPS backend with selective CPU LAPACK factorizations,
and Metal Householder QR now uses scaled norms for weak columns. Large captured
QR panel widths and whole workflows were benchmarked. Bidirectional finite RTM
messages were implemented and tested, but cap increases beat them at lower cost
in the measured 4×4 cases; cut agreement alone proved unreliable. CPU FP64 remains
the default. Future refinement work needs a different update/selection criterion
and independent accuracy checks, rather than more iterations of this pilot.
