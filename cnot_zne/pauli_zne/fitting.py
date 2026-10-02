"""Equal-weight direct-mode and exponential fits, with paired shot noise."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


class ExponentialFits:
    def __init__(self, s, options):
        self.s = np.array(s, float)
        self.lo, self.hi = options["rate_bounds"]
        self.bound = options["amplitude_bound"]
        rates = np.unique(np.r_[self.lo, np.geomspace(max(self.lo, 1e-5), self.hi, 48)])
        i, j = np.triu_indices(len(rates), 1)
        self.seeds = np.column_stack([rates[i], rates[j]])
        self.x = np.exp(-self.s[None, :, None] * self.seeds[:, None, :])
        self.pinv = np.linalg.pinv(self.x)

    def single(self, y):
        if len(y) < 2:
            return {"status": "insufficient_points"}

        def solve(b):
            x = np.exp(-b * self.s)
            a = float(np.clip(x @ y / (x @ x), -self.bound, self.bound))
            return float(np.sum((a * x - y) ** 2)), a

        fit = minimize_scalar(
            lambda b: solve(b)[0],
            bounds=(self.lo, self.hi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        rate = min([self.lo, fit.x, self.hi], key=lambda b: solve(b)[0])
        cost, a = solve(rate)
        return dict(
            status="ok" if fit.success else "optimizer_non_success",
            estimate=a,
            sse=cost,
            rates=[float(rate)],
            amplitudes=[a],
            boundary=bool(
                abs(a) >= 0.99999 * self.bound
                or rate <= self.lo + 1e-7
                or rate >= self.hi - 1e-7
            ),
        )

    def amplitudes(self, rates, y):
        x = np.exp(-self.s[:, None] * rates)
        a = np.linalg.lstsq(x, y, rcond=None)[0]
        if max(abs(a)) > self.bound:
            candidates = []
            for col in (0, 1):
                other = 1 - col
                for edge in (-self.bound, self.bound):
                    aa = np.zeros(2)
                    aa[col] = edge
                    aa[other] = np.clip(
                        x[:, other]
                        @ (y - edge * x[:, col])
                        / (x[:, other] @ x[:, other]),
                        -self.bound,
                        self.bound,
                    )
                    candidates.append(aa)
            a = min(candidates, key=lambda v: np.sum((x @ v - y) ** 2))
        return a, x @ a - y

    def dual(self, y, extra_seed=None):
        if len(y) < 4:
            return {"status": "insufficient_points"}
        amplitudes = np.einsum("pij,j->pi", self.pinv, y, optimize=False)
        residual = np.einsum("pij,pj->pi", self.x, amplitudes, optimize=False) - y
        costs = np.sum(residual**2, axis=1)
        bad = np.max(abs(amplitudes), axis=1) > self.bound
        if bad.any():
            x = self.x[bad]
            cost = np.full(len(x), np.inf)
            for col in (0, 1):
                other = 1 - col
                for edge in (-self.bound, self.bound):
                    rem = y - edge * x[:, :, col]
                    a = np.clip(
                        np.sum(x[:, :, other] * rem, axis=1)
                        / np.sum(x[:, :, other] ** 2, axis=1),
                        -self.bound,
                        self.bound,
                    )
                    cost = np.minimum(
                        cost, np.sum((rem - a[:, None] * x[:, :, other]) ** 2, axis=1)
                    )
            costs[bad] = cost
        seed = self.seeds[np.argmin(costs)]
        starts = [
            seed,
            [self.lo + 0.05 * (self.hi - self.lo), self.lo + 0.5 * (self.hi - self.lo)],
        ]
        if extra_seed is not None:
            starts.append(extra_seed)
        fits = []
        for start in starts:
            f = least_squares(
                lambda r: self.amplitudes(r, y)[1] / max(0.01, np.linalg.norm(y)),
                np.clip(start, self.lo + 1e-10, self.hi - 1e-10),
                bounds=(self.lo, self.hi),
                max_nfev=600,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            rates = np.sort(f.x)
            a, err = self.amplitudes(rates, y)
            fits.append(
                dict(
                    status="ok" if f.success else "optimizer_non_success",
                    estimate=float(sum(a)),
                    sse=float(err @ err),
                    rates=rates.tolist(),
                    amplitudes=a.tolist(),
                    boundary=bool(
                        max(abs(a)) >= 0.99999 * self.bound
                        or rates[0] <= self.lo + 1e-7
                        or rates[1] >= self.hi - 1e-7
                    ),
                )
            )
        return min(fits, key=lambda f: f["sse"])


def designs(curves):
    """Maps curve labels to designs; input arrays include the zero point."""
    out = {}
    for cap, m in curves.items():
        if m["ratio"] is not None:
            out[f"ratio_K{cap}"] = m["ratio"][1:, None]
        else:
            out[f"ratio_K{cap}"] = None
        if m["Q_avg"] is not None:
            d = m["Q_contrast"][1:]
            scale = max(abs(d))
            out[f"Q_K{cap}"] = (
                np.column_stack([m["Q_avg"][1:], d / scale]) if scale > 1e-14 else None
            )
        else:
            out[f"Q_K{cap}"] = None
    return out


def fit_linear(x, y):
    if x is None:
        return {"status": "undefined_or_degenerate_modes"}
    if len(y) < x.shape[1]:
        return {"status": "insufficient_points"}
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return {"status": "rank_deficient"}
    w = np.linalg.pinv(x)[0]
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return dict(
        status="ok",
        estimate=float(coef[0]),
        sse=float(np.sum((x @ coef - y) ** 2)),
        noise_gain=float(np.linalg.norm(w)),
        condition=float(np.linalg.cond(x)),
        weights=w.tolist(),
    )


def benchmark(s, y, target, curves, options, scenarios, progress=lambda x: None):
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or (abs(y) > 1 + 1e-10).any():
        raise ValueError("Input Pauli means must lie in [-1,1]")
    y = np.clip(y, -1, 1)
    matrices = designs(curves)
    exp = ExponentialFits(s, options)
    baseline = {name: fit_linear(x, y) for name, x in matrices.items()}
    baseline["single_exp"] = exp.single(y)
    baseline["dual_exp"] = exp.dual(y)
    for f in baseline.values():
        if "estimate" in f:
            f["absolute_error"] = f["estimate"] - target
            f["relative_error_pct"] = (
                100 * (f["estimate"] - target) / abs(target)
                if abs(target) > 1e-12
                else None
            )
    results = []
    samples = []
    rng = np.random.default_rng(options["seed"])
    for name, shots in scenarios.items():
        draws = np.tile(y, (options["trials"], 1))
        for j, n in enumerate(shots):
            if n is not None:
                draws[:, j] = (
                    2 * rng.binomial(n, (1 + y[j]) / 2, size=options["trials"]) / n - 1
                )
        estimates = {}
        statuses = {}
        bounds = {}
        for model, f in baseline.items():
            if model in matrices and f["status"] == "ok":
                estimates[model] = np.einsum(
                    "ij,j->i", draws, np.array(f["weights"]), optimize=False
                )
                statuses[model] = [True] * len(draws)
                bounds[model] = [False] * len(draws)
        for model, func in [("single_exp", exp.single), ("dual_exp", exp.dual)]:
            if "estimate" not in baseline[model]:
                continue
            fs = [
                (
                    func(row, baseline[model]["rates"])
                    if model == "dual_exp"
                    else func(row)
                )
                for row in draws
            ]
            estimates[model] = np.array([f["estimate"] for f in fs])
            statuses[model] = [f["status"] == "ok" for f in fs]
            bounds[model] = [f["boundary"] for f in fs]
            for i, f in enumerate(fs):
                samples.append(
                    dict(
                        scenario=name,
                        trial=i,
                        method=model,
                        observations=draws[i].tolist(),
                        **f,
                    )
                )
        for model, est in estimates.items():
            if not np.isfinite(est).all():
                raise FloatingPointError(f"Nonfinite estimates in {model}")
            err = est - target
            scale = 100 / abs(target) if abs(target) > 1e-12 else None
            results.append(
                dict(
                    scenario=name,
                    method=model,
                    trials=len(est),
                    shots_per_point=shots,
                    total_shots_per_trial=sum(v or 0 for v in shots),
                    rmse=float(np.sqrt(np.mean(err**2))),
                    relative_rmse_pct=(
                        float(np.sqrt(np.mean(err**2)) * scale) if scale else None
                    ),
                    median_abs_relative_error_pct=(
                        float(np.median(abs(err)) * scale) if scale else None
                    ),
                    mean_estimate=float(np.mean(est)),
                    optimizer_non_success=int(np.sum(np.logical_not(statuses[model]))),
                    boundary_fraction=float(np.mean(bounds[model])),
                )
            )
        progress(f'Finished shot scenario {name}: {options["trials"]} paired trials')
    return baseline, results, samples
