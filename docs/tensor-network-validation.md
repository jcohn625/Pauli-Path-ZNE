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
