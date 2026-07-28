# Fitting Procedure

[Back to README](../README.md)

## Q Summaries

For grouped Monte Carlo data, each group gives:

```math
Q^\pm_g(s)
=
\frac{N^\pm_g(s)}{D^\pm_g}.
```

The default summary is pointwise median-of-means:

```math
\widehat{Q}^\pm(s)
=
\mathrm{median}_g Q^\pm_g(s).
```

This is robust to rare denominator-heavy prefix groups. For large enough group
sizes, pooled, group mean, and MoM should agree.

## Log-Mode Fit

The Q-assisted fit does not fit `Q+` and `Q-` independently. It first forms:

```math
\ell(s)
=
\frac{1}{2}
\left[
\log Q^+(s)
+
\log Q^-(s)
\right],
```

```math
\delta(s)
=
\frac{1}{2}
\left[
\log Q^+(s)
-
\log Q^-(s)
\right].
```

Then it fits:

```math
\ell(s)
=
-a_1s + a_2s^2 - a_3s^3,
```

```math
\delta(s)
=
d_1s + d_2s^2.
```

These define:

```math
Q_{\mathrm{ave}}(s)
=
e^{\ell(s)}\cosh(\delta(s)),
```

```math
Q_{\mathrm{diff}}(s)
=
e^{\ell(s)}\sinh(\delta(s)).
```

## Final Observable Fit

The observable model is:

```math
O_{\mathrm{fit}}(s)
=
uQ_{\mathrm{ave}}(s)
+
vQ_{\mathrm{diff}}(s).
```

At `s=0`, `Q_ave(0)=1` and `Q_diff(0)=0`, so:

```math
O_{\mathrm{fit}}(0)=u.
```

The reported zero-noise estimate is `u_O0`.

With noisy observable data, the final fit minimizes:

```math
\sum_s
\frac{
\left[
uQ_{\mathrm{ave}}(s)
+
vQ_{\mathrm{diff}}(s)
-
O_{\mathrm{obs}}(s)
\right]^2
}{\sigma_O(s)^2}
+
\alpha_v v^2.
```

Only `v` is regularized. The zero-noise coefficient `u` is not regularized.

If `use_observed=False`, the script simulates shot noise from exact/MPS values:

```math
\sigma_O(s)
\approx
\sqrt{\frac{1-O(s)^2}{N_{\mathrm{shots}}}}.
```

If `use_observed=True`, loaded observable values are fit directly. If the CSV
contains `sigma`, `se`, `stderr`, or `std_error`, those error bars are used.

## Dual-Exponential Baseline

The baseline ignores Q data and fits:

```math
O_{\mathrm{dual}}(s)
=
c_1e^{-b_1s}
+
c_2e^{-b_2s}.
```

Use this as a baseline, not as the preferred estimator in difficult
sign-changing cases.
