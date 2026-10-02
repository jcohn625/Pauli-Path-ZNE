# Validation (2026-10-02)

- 31 tests passed using Python 3.13, NumPy 2.3.0, SciPy 1.17.1.
- Built a wheel and installed it into an isolated temporary target. Ran the complete N6/L2 example from `/tmp`, outside this repository, using the installed package.
- The N6 reference O(0) is 0.0078094913686142 from statevector; the noisy curve uses independent density matrices.
- At K64, ratio intercept absolute error: -3.07e-14; Q intercept absolute error: -3.32e-14.
- Refit the installed-package run with two new per-point shot scenarios (1,000 paired trials each), without recomputing MPS curves.
- Checked both forward noise placements, anisotropic site rates, nearest-neighbor correlated rates, elementary unsigned paths, Q sector edge cases, selective/noiseless shot points, insufficient fitting grids, randomized low-rank recovery, and MPS reference convergence/exhausted-budget reporting.
- Tests emit Accelerate backend floating-point warnings in optimized NumPy contractions on this host. The independent small-system comparisons and explicit decomposition checks pass; this does not certify every large-bond calculation on this backend.
- N32/L12 is supplied as a configuration example. Its CNOT-resolved high-bond convergence has not been benchmarked in this validation pass.

Generated run artifacts are retained locally under `runs/` and excluded from Git. The package has no dependency on the original checkout or on these generated files.

## Notebook validation

`ZNE_workflow.ipynb` contains four code cells plus mathematical explanation. All four cells executed successfully in a fresh Jupyter kernel on the N6/L2 example, including 200 paired trials for each of four shot scenarios. The notebook schema was validated, no error outputs were present, and the six-panel figure was visually checked. Repeated numerical warnings are grouped and exported, rather than discarded. Saved outputs are included in the notebook.

## Workflow-function refactor

The four-cell notebook now calls `define_experiment`, `calculate_curves`, `fit_curves`, and `analyze_results` from `pauli_zne.workflow`. The simplified notebook executed in a fresh kernel and reproduced all pre-refactor noiseless intercept estimates within 1e-10 relative / 1e-12 absolute tolerance. The package suite now has 34 passing tests, including copied-input isolation, no-MPS refitting, exports, observation validation, and analysis with a zero target and no shots.
