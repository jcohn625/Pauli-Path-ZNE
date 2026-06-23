# Pauli MPS Solver User Guide

This guide describes `pauli_mps_solver.py`, a deterministic blocked-MPS solver
for noisy 1D Heisenberg brickwall circuits in the local Pauli basis.

The solver evolves a final Pauli observable backward through the circuit and
contracts it with an initial product of two-qubit singlets on even bonds:

```text
O_final -> backward noisy Trotter evolution -> <singlet product | O(0) | singlet product>
```

The code uses blocked even bonds as MPS sites:

```text
MPS site b = qubits (2b, 2b+1)
local dimension = 16 = {II, IX, ..., ZZ}
```

## Quick Start

```python
import time
import numpy as np

from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz

n_qubits = 20
n_steps = 12
phi = np.pi / 12
lam_xyz = np.full((n_qubits, 3), 1e-3)

value, mps, info = evolve_observable_backward_mps(
    pauli_zz(n_qubits, 0, 1),
    n_qubits=n_qubits,
    phi=phi,
    lam_xyz=lam_xyz,
    n_steps=n_steps,
    chi_max=256,
    return_mps=True,
)

print(value)
print(info["bond_dims"])
print(info.get("active_blocks_by_backward_step"))
```

## Main Function

```python
evolve_observable_backward_mps(
    final_pauli,
    *,
    n_qubits,
    phi,
    lam_xyz,
    n_steps,
    chi_max=None,
    cutoff=1e-12,
    svd_method="auto",
    oversample=24,
    n_iter=1,
    canonicalize=True,
    parallel_kernels="auto",
    use_lightcone=True,
    enforce_global_parity=True,
    noise_model="legacy_sum",
    noise_placement="gate",
    return_mps=False,
)
```

### Required Inputs

`final_pauli`
: Target Pauli observable. Can be a string like `"ZZIIII"` or an integer array
  with encoding `I=0, X=1, Y=2, Z=3`.

`n_qubits`
: Number of qubits. Must be even for the blocked even-bond MPS.

`phi`
: Heisenberg gate angle. The current convention matches:
  `U(phi) = cos(phi) I - i sin(phi) SWAP`.

`lam_xyz`
: Single-qubit Pauli noise lambdas. Accepted shapes:

```text
(n_qubits, 3)              same noise after every half layer
(2, n_qubits, 3)           separate even/odd half-layer noise, repeated in time
(n_steps, 2, n_qubits, 3)  fully time-dependent schedule
(2*n_steps, n_qubits, 3)   flattened half-layer schedule
```

For the default sampler-compatible placement, the even/odd entries are used as
the damping after the corresponding touched-gate layer. With
`noise_placement="layer"`, they are interpreted as full half-layer damping.

The lambda columns are:

```text
[:, 0] = lambda_x
[:, 1] = lambda_y
[:, 2] = lambda_z
```

`n_steps`
: Number of full even+odd Trotter steps.

## Convenience Constructors

Use `pauli_zz` for two-point `Z_i Z_j` observables:

```python
target = pauli_zz(n_qubits, 10, 11)
```

You can also pass strings:

```python
value = evolve_observable_backward_mps(
    "ZZIIIIIIIIIIIIIIIIII",
    n_qubits=20,
    phi=np.pi / 12,
    lam_xyz=np.zeros((20, 3)),
    n_steps=8,
    chi_max=128,
)
```

## Defaults

The recommended defaults are already enabled:

```python
use_lightcone=True
enforce_global_parity=True
canonicalize=True
parallel_kernels="auto"
svd_method="auto"
noise_model="legacy_sum"
noise_placement="gate"
```

You usually only need to set:

```python
final_pauli, n_qubits, phi, lam_xyz, n_steps, chi_max
```

## Light-Cone Optimization

By default, `use_lightcone=True`.

The solver evolves only the active backward light cone of the target Pauli. For
nonlocal observables, it tracks multiple active islands and keeps them
factorized until they touch.

This matters because observable-based MPS cost depends strongly on the target:

```text
local Pauli       -> small light cone, low chi often works
nearby correlator -> moderate light cone
far correlator    -> two cones that may merge later
large Pauli string -> broad support from the start
```

Useful diagnostics:

```python
info["active_blocks_by_backward_step"]
info["active_intervals_by_backward_step"]
info["active_intervals"]
```

Disable light-cone mode only for validation:

```python
value_full = evolve_observable_backward_mps(..., use_lightcone=False)
```

## Global Parity Filter

By default, `enforce_global_parity=True`.

The Heisenberg Pauli transfer conserves the relative parity sector:

```text
parity(X) xor parity(Y)
parity(X) xor parity(Z)
```

The initial singlet product can only contract with the sector where:

```text
parity(X) = parity(Y) = parity(Z)
```

If the target Pauli is outside this sector, the solver returns exact zero
immediately.

Diagnostics:

```python
info["global_pauli_parity"]
info["parity_filtered"]
```

Example:

```python
# Single Z has parity (0, 0, 1), so it is filtered to zero.
value = evolve_observable_backward_mps(
    "ZIIIIIII",
    n_qubits=8,
    phi=np.pi / 12,
    lam_xyz=np.zeros((8, 3)),
    n_steps=4,
)
```

## Noise Models

The default is:

```python
noise_model="legacy_sum"
```

This matches `Pauli_path_Heis.pauli_diag_factors_from_lambda`:

```text
eta_X = exp(-2 lambda_y) + exp(-2 lambda_z) - 1
eta_Y = exp(-2 lambda_x) + exp(-2 lambda_z) - 1
eta_Z = exp(-2 lambda_x) + exp(-2 lambda_y) - 1
```

You can also use:

```python
noise_model="independent"
```

which applies independent Pauli noise channels:

```text
eta_X = exp(-2 lambda_y) exp(-2 lambda_z)
eta_Y = exp(-2 lambda_x) exp(-2 lambda_z)
eta_Z = exp(-2 lambda_x) exp(-2 lambda_y)
```

## Noise Placement

The default is:

```python
noise_placement="gate"
```

This matches `Pauli_path_Heis.evolve_many_paths_with_1q_noise`: after each
two-qubit gate update, only the two qubits touched by that gate receive
diagonal Pauli damping. Since gates inside one even/odd layer are disjoint, the
MPS applies this as one touched-qubit diagonal layer after the odd gate layer
and one after the even gate layer.

For the older MPS convention, use:

```python
noise_placement="layer"
```

This applies full-chain damping at each half-layer in the physical adjoint
order:

```text
odd noise -> odd gates -> even noise -> even gates
```

## Bond Dimension Sweeps

Use bond-dimension sweeps to check convergence:

```python
bond_dims = [32, 64, 128, 256, 512]
values = []

for chi in bond_dims:
    t0 = time.time()
    value, mps, info = evolve_observable_backward_mps(
        pauli_zz(n_qubits, 10, 11),
        n_qubits=n_qubits,
        phi=phi,
        lam_xyz=lam_xyz,
        n_steps=n_steps,
        chi_max=chi,
        return_mps=True,
    )
    dt = time.time() - t0
    values.append(value)
    print(
        "chi =", chi,
        "runtime =", dt,
        "value =", value,
        "max bond =", max(info["bond_dims"]),
        "active blocks =", info.get("active_blocks_by_backward_step"),
    )
```

Low `chi` can be nonmonotonic. Judge convergence by the change between larger
bond dimensions, not by monotonic approach.

For local weak-angle observables, `chi=32` or `64` may already be good. For
nonlocal observables or stronger `phi`, check `chi=128`, `256`, or higher.

## Step Series Without Repeated Runs

Use `evolve_observable_backward_series` to compute values for steps
`0, 1, ..., max_steps` in one pass:

```python
from pauli_mps_solver import evolve_observable_backward_series, pauli_zz

values, final_mps, info = evolve_observable_backward_series(
    pauli_zz(n_qubits, 0, 1),
    n_qubits=n_qubits,
    phi=phi,
    lam_xyz=lam_xyz,
    max_steps=12,
    chi_max=256,
    return_final_mps=True,
)

for step, value in enumerate(values):
    print(step, value)
```

This is faster than calling `evolve_observable_backward_mps` separately for
each step count because it reuses the previously evolved observable.

Important limitation: the one-pass series helper is for time-independent
circuits. `lam_xyz` must have shape:

```text
(n_qubits, 3)
(2, n_qubits, 3)
```

For a fully time-dependent schedule, the adjoint order depends on the requested
final step, so separate calls are safer unless you intentionally want repeated
identical steps.

## SVD Backends

`svd_method="auto"` is recommended.

Options:

```text
auto        full SVD for small matrices, randomized truncated SVD for large ones
full        exact dense SVD; slow but useful for small validation tests
sparse      SciPy svds backend; robust calibration mode, can be slow
randomized  force randomized truncated SVD
```

For final validation on small cases:

```python
value = evolve_observable_backward_mps(..., svd_method="full")
```

For production runs, prefer:

```python
svd_method="auto"
```

## Canonicalization

Keep:

```python
canonicalize=True
```

The solver right-canonicalizes before each left-to-right odd-layer TEBD sweep.
This is important because noise and Pauli-transfer maps are nonunitary; without
canonicalization, local SVD truncation is gauge-sensitive and convergence can
look erratic.

## Parallel Kernels

The solver includes Numba `prange` kernels for local Pauli transfer and diagonal
noise updates.

Recommended:

```python
parallel_kernels="auto"
```

Options:

```text
auto      choose serial or parallel kernels based on local tensor size
parallel  force Numba prange kernels
serial    force serial kernels
True      same as "parallel"
False     same as "serial"
```

Parallel kernels tend to help at larger bond dimensions, especially around
`chi >= 256`, but can be slightly slower at small bond dimensions.

## Interpreting Results

### State MPS vs Observable MPS

This is an observable-based Heisenberg MPS. Cost depends on the target Pauli.

State-based MPS:

```text
cost mostly depends on evolved state and circuit depth
measurement choice is usually cheap afterward
```

Observable MPS:

```text
cost depends strongly on final Pauli support and light-cone growth
local observables can be much cheaper
nonlocal observables may need larger chi
```

### Local `Z_i Z_{i+1}` Plateaus

For local even-bond observables like `Z_0 Z_1` or `Z_10 Z_11`, values may decay
quickly and then show only small oscillations. This can be physical, not a
solver bug. For example, with `lambda=0`, `phi=pi/12`, `Z_0 Z_1` on `N=20`
matches exact statevector dynamics and approaches a plateau near `-0.565`.

## Common Validation Checks

Compare light-cone and full-chain modes:

```python
v_lc = evolve_observable_backward_mps(..., use_lightcone=True)
v_full = evolve_observable_backward_mps(..., use_lightcone=False)
print(abs(v_lc - v_full))
```

Compare SVD methods at small sizes:

```python
v_auto = evolve_observable_backward_mps(..., svd_method="auto")
v_full = evolve_observable_backward_mps(..., svd_method="full")
```

Check bond convergence:

```python
for chi in [32, 64, 128, 256]:
    print(chi, evolve_observable_backward_mps(..., chi_max=chi))
```

Check stronger/weaker noise:

```python
for lam in [0.0, 1e-3, 1e-2]:
    lam_xyz = np.full((n_qubits, 3), lam)
    print(lam, evolve_observable_backward_mps(..., lam_xyz=lam_xyz))
```

## Useful Helper Functions

`pauli_zz(n_qubits, i, j)`
: Construct an integer Pauli vector for `Z_i Z_j`.

`pauli_parity_vector(pauli)`
: Return `(n_X mod 2, n_Y mod 2, n_Z mod 2)`.

`has_singlet_compatible_global_parity(pauli)`
: Check whether a target lies in the global relative-parity sector that can
  contract with the singlet product.

`mps_to_dense(mps)`
: Reconstruct a dense blocked tensor. Use only for small tests.

`uncompressed_mps_reference(...)`
: Small reference path that keeps all singular vectors. Useful for beta tests,
  not production.

## Performance Notes

- Light-cone mode gives the biggest gain for local observables, larger systems,
  and shorter times before cones fill the chain.
- Weak `phi`, local targets, and nonzero noise usually need much smaller
  `chi_max`.
- Stronger entangling angles, nonlocal targets, and broad Pauli strings need
  larger `chi_max`.
- `phi = pi/4` is the maximally entangling point for the current
  `cos(phi) I - i sin(phi) SWAP` gate family.
- `svd_method="full"` is often much slower at large `chi`.
- `return_mps=False` is slightly cheaper if you only need the scalar value.
