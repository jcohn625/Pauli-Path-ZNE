# Tensor-network integration validation

Validated 2026-10-13 in an isolated Python 3.12 environment on Apple Silicon
macOS. Dependencies were installed from `requirements-dev.txt` and
`requirements-metal.txt`; `python -m pip check` reported no broken requirements.
The existing sampling and fitting implementations were not modified.

## Regression suite

```bash
TEST_PEPS_GPU_BOUNDARY=1 TEST_PEPS_METAL=1 TEST_MPS_METAL=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m pytest tests -q -W error::RuntimeWarning
```

**68 passed, plus 4 subtests passed.** This includes all optional GPU tests;
none were skipped in the final run. Without GPU flags, the GPU-specific tests
are skipped so the CPU suite can run on other platforms. Those platforms were
not independently tested in this release.

Checks include:

- Signed and unsigned MPS residual corrections against independently compiled
  exact six-qubit networks, at zero and positive noise and a nondefault angle.
- Unchanged finite-bond MPS baseline, zero correction without truncation, exact
  adjoint telescoping, and residual normalization/scale tracking.
- PEPS versus independent exact contractions with anisotropic scheduled noise,
  both noise placements/timings, signed and unsigned paths, and parity selection.
- PEPS residual telescoping against exact contraction of the same fixed PEPS;
  retained versus streamed residuals and approximate backward environments.
- Total charge-sector truncation, rectangular/rank-deficient GPU QR, Jacobi SVD,
  restoration of scoped backend modifications, and CPU/GPU residual agreement.
- Q normalization, observable/sign-mass identities, undefined-sector handling,
  CSV compatibility with the existing fitter, and Metal wrapper integration.

## Documented commands and existing fitter

Both quick-run commands in [the usage guide](tensor-network-methods.md) completed
for scales 0, 1, 2, 4, 8, creating curves, observables, Q groups and JSON diagnostics.
Their exports were passed to `run_zne_fit_compare` with `methods="layer"` for MPS
and `methods="gate"` for PEPS. Each completed a two-trial smoke run with a reduced
10-point dual-exponential rate grid and generated all summary, trial, curve and
plot outputs. This verifies integration, not statistical performance.

The new modules contain no author-specific absolute import paths. Dependency
installation uncovered and resolved the explicit `opt_einsum` requirement and
quimb's SciPy >=1.16 constraint. The final pins use NumPy 1.26.4, SciPy 1.17.1 and
Numba 0.60.0. The host's NumPy 2.3.0 stack emitted arithmetic warnings and is not
used as the validated reference environment.

## Interpretation

These are small-system correctness and integration tests, not a new
large-system accuracy claim. MPS still requires forward/guide bond convergence.
PEPS residual correction removes boundary truncation only in the exact-guide
limit; it does not undo spatial gate/simple-update truncation. GPU arithmetic
uses float32, and larger blocks require their own precision and Jacobi-sweep
checks. See the usage guide for circuit conventions and convergence controls.

## Per-CNOT-layer parity extension (2026-10-13)

The transported-frame implementation adds `--noise-placement after_cnot` to
both engines. It applies the full graph-wide channel after each nonempty CX
layer and supports scheduled single-qubit and arbitrary-edge weight-2 rates.
See [the CNOT parity guide](cnot-parity-methods.md).

Final regression command:

```bash
TEST_PEPS_METAL=1 TEST_PEPS_GPU_BOUNDARY=1 TEST_MPS_METAL=1 \
PYTHONPATH=cnot_zne NUMBA_CACHE_DIR=/tmp/pauli_cnot_numba \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m pytest tests cnot_zne/tests -q -W error::RuntimeWarning
```

Result: **123 passed, 1 skipped, plus 4 subtests passed**. The skipped pre-existing
analysis/plot test requires optional pandas. All requested Metal tests ran.

New exact checks cover four- and six-qubit signed/unsigned elementary circuits,
all 66 possible weight-1/2 generators on four qubits, rate variation at every
CX occurrence, idle-site damping, routing of longer-range correlations, and
phi=pi/4 unsigned transfers. The signed MPS also matches the independent forward
density-matrix solver from `cnot_zne`. With sufficiently large guides, actual
truncated MPS residuals recover the dense reference, including routing errors.
PEPS streamed residuals reproduce exact contraction of the same evolved PEPS
when exact backward environments are used.

Both documented CLI examples generated ratios, Q groups, observables and JSON.
NPZ loading, its SHA-256 provenance record, and the `cnot` Q-group label are
covered by a subprocess test. Both engines' Q/observable exports completed
small end-to-end runs through the existing fitter using `methods="cnot"`.

Additional execution checks completed for a 12-qubit, two-step MPS (forward and
guide cap 16) and a 4x4, one-step PEPS (spatial/boundary cap 4, environment cap 8)
with all nearest-neighbor Pauli pair rates .0001. These are bounded execution
checks, not bond-convergence or accuracy claims. No large-system performance
comparison with the former full-Heisenberg-noise model is claimed.
