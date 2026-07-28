# Motivation And Workflow

[Back to README](../README.md)

## Goal

Estimate a zero-noise target observable for a 1D Heisenberg brickwall circuit,
for example:

```text
target = Z_6 Z_10
```

The workflow combines three data streams:

```text
Q data       Monte Carlo estimates of Q+(s) and Q-(s)
O data       noisy hardware, simulated shot-noisy, or MPS observable data O(s)
baseline     ordinary dual-exponential ZNE fit to O(s)
```

## Why Q-Assisted ZNE

Standard ZNE fits `O(s)` directly. That can be fragile when the observable
changes sign or when only a few noisy points determine the extrapolated
intercept.

The Q-assisted strategy uses classical Pauli-path sampling to learn the sector
damping structure first, then uses observable data only for a final two-parameter
linear fit.

The sector damping functions are:

```math
Q^\pm(s)
=
\frac{\sum_{\gamma \in \pm} |A_\gamma|D_\gamma^s}
{\sum_{\gamma \in \pm} |A_\gamma|}.
```

Here `+` and `-` label the signed Pauli-path sectors, `A_gamma` is the noiseless
path amplitude, and `D_gamma^s` is the path damping factor at noise scale `s`.

## End-To-End Workflow

1. Choose the target problem.

```text
N          number of qubits
L          number of Heisenberg layers
target     for example Z_6 Z_10
phi        Heisenberg circuit angle
lambda     base noise strength
s_grid     noise scale points, often 1,2,4,8
```

2. Generate or load `O(s)`.

For benchmarks, compute an MPS curve with:

```text
noise_model       independent
noise_placement   layer
```

For hardware, load measured values and error bars if available.

3. Estimate `Q+(s)` and `Q-(s)`.

Use both final-layer DP samplers:

```text
gate_qmom     gate-wise triggered prefix sampler plus final-layer DP
layer_qmom    layer-wise DP prefix sampler plus final-layer DP
```

4. Diagnose Q convergence.

Compare pooled, group mean, and median-of-means (MoM). If they disagree, the Q
estimate is still dominated by rare prefix batches.

5. Fit Q modes.

The fitter converts `Q+` and `Q-` into average and difference modes and fits a
smooth log ansatz.

6. Fit the observable.

The final Q-assisted model is:

```math
O_{\mathrm{fit}}(s)
=
uQ_{\mathrm{ave}}(s)
+
vQ_{\mathrm{diff}}(s).
```

At zero noise:

```math
O_{\mathrm{fit}}(0) = u.
```

In output files, read this as `u_O0`.

## Final-Layer DP

The final-layer DP exactly sums all nonzero final-layer continuations
conditioned on the sampled prefix. This removes final-layer branch noise.

The remaining hard part is prefix rare-event variance. That is why grouped Q
diagnostics are important.
