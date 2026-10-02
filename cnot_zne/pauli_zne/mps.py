"""Pauli-basis MPS with elementary-gate absolute paths and stable SVD.

Physical dimension 4 per qubit. Unsigned paths take abs at EACH elementary
rotation/CNOT transfer and at the singlet terminal functional, before summing.
"""

import time
import numpy as np
from scipy.linalg import qr, svd
from .circuit import transfers, DIRECTIONS, gates, noise_diagonals


class MPS:
    def __init__(self, observable, cap, settings):
        self.a = [
            np.eye(4)["IXYZ".index(p)].reshape(1, 4, 1).copy() for p in observable
        ]
        self.cap = cap
        self.settings = settings
        self.log_norm = 0.0
        self.discarded = 0.0
        self.max_bond = 1
        self.rng = np.random.default_rng(settings["seed"])

    def right_canonical(self):
        for q in range(len(self.a) - 1, 0, -1):
            a = self.a[q]
            u, r = qr(a.reshape(a.shape[0], -1).T, mode="economic", check_finite=True)
            self.a[q] = u.T.reshape(-1, 4, a.shape[2])
            self.a[q - 1] = np.einsum("lpm,mr->lpr", self.a[q - 1], r.T, optimize=True)

    def split(self, mat):
        cfg = self.settings
        rank = min(self.cap + cfg["oversample"], min(mat.shape))
        if cfg["svd"] == "randomized" and rank < min(mat.shape):
            omega = self.rng.normal(size=(mat.shape[1], rank))
            q = qr(mat @ omega, mode="economic", check_finite=True)[0]
            for _ in range(cfg["power_iterations"]):
                z = qr(mat.T @ q, mode="economic", check_finite=True)[0]
                q = qr(mat @ z, mode="economic", check_finite=True)[0]
            small = q.T @ mat
        else:
            q = None
            small = mat
        u, s, v = svd(
            small, full_matrices=False, lapack_driver="gesvd", check_finite=True
        )
        # Check the decomposition before truncation on three deterministic columns.
        ix = np.linspace(0, small.shape[1] - 1, min(3, small.shape[1]), dtype=int)
        rebuilt = np.einsum("ik,k,kj->ij", u, s, v[:, ix], optimize=False)
        err = np.linalg.norm(rebuilt - small[:, ix])
        if err > 1e-9 * max(np.linalg.norm(small), 1e-300):
            raise FloatingPointError("SVD reconstruction check failed")
        if q is not None:
            u = q @ u
        keep = (
            min(self.cap, max(1, int(np.sum(s > cfg["cutoff"] * s[0]))))
            if len(s)
            else 1
        )
        total = float(np.linalg.norm(mat) ** 2)
        self.discarded += max(0.0, total - float(s[:keep] @ s[:keep])) / max(
            total, 1e-300
        )
        self.max_bond = max(self.max_bond, keep)
        return u[:, :keep], s[:keep, None] * v[:keep]

    def sweep(self, operators, diagonal=False):
        if not operators:
            return
        self.right_canonical()
        for q in range(len(self.a) - 1):
            a, b = self.a[q : q + 2]
            if q in operators:
                theta = np.einsum("lam,mbr->labr", a, b, optimize=True)
                if diagonal:
                    theta *= operators[q][None, :, :, None]
                else:
                    theta = np.einsum(
                        "uvab,labr->luvr", operators[q], theta, optimize=True
                    )
                left, right = theta.shape[0], theta.shape[-1]
                u, v = self.split(theta.reshape(left * 4, 4 * right))
                self.a[q] = u.reshape(left, 4, -1)
                self.a[q + 1] = v.reshape(u.shape[1], 4, right)
            else:
                u, r = qr(
                    a.reshape(-1, a.shape[-1]), mode="economic", check_finite=True
                )
                self.a[q] = u.reshape(a.shape[0], 4, -1)
                self.a[q + 1] = np.einsum("lm,mpr->lpr", r, b, optimize=True)
        norm = float(np.linalg.norm(self.a[-1]))
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("MPS norm vanished or became nonfinite")
        self.a[-1] /= norm
        self.log_norm += np.log(norm)

    def rotate(self, qs, block):
        for q in qs:
            for side in (0, 1):
                self.a[q + side] = np.einsum(
                    "pq,lqr->lpr", block[side], self.a[q + side], optimize=True
                )

    def noise(self, singles, pairs, scale):
        if scale == 0:
            return
        local, pair = noise_diagonals(singles, pairs, scale)
        for q in range(len(self.a)):
            self.a[q] *= local[q][None, :, None]
        active = {q: pair[q] for q in range(len(pair)) if np.any(pairs[q])}
        self.sweep(active, diagonal=True)

    def expectation(self, unsigned):
        env = np.ones(1)
        terminal = np.diag([1, 1, 1, 1] if unsigned else [1, -1, -1, -1])
        # Independent non-BLAS contraction: avoid the earlier unchecked final dot path.
        for q in range(0, len(self.a), 2):
            mid = np.einsum("l,lpm->pm", env, self.a[q], optimize=False)
            env = np.einsum(
                "pm,mqr,pq->r", mid, self.a[q + 1], terminal, optimize=False
            )
        scalar = float(env[0])
        if not np.isfinite(scalar):
            raise FloatingPointError("Nonfinite terminal contraction")
        logabs = float(np.log(abs(scalar)) + self.log_norm) if scalar else None
        if unsigned and scalar <= 0:
            return None, None
        if logabs is None:
            value = 0.0
        elif logabs > np.log(np.finfo(float).max):
            value = None
        else:
            value = float(np.sign(scalar) * np.exp(logabs))
        return value, logabs


def evaluate(config, singles, pairs, scale, cap, unsigned=False):
    start = time.perf_counter()
    c = config["circuit"]
    m = MPS(c["O"], cap, config["mps"])
    rotations, cnots = transfers(c["phi"])
    if unsigned:
        rotations = tuple(tuple(abs(t) for t in b) for b in rotations)
        cnots = tuple(abs(t) for t in cnots)
    for t in range(c["L"] - 1, -1, -1):
        for parity in (1, 0):
            qs = gates(c["N"], parity)
            if not qs:
                continue
            m.rotate(qs, rotations[3])
            for stage in range(2, -1, -1):
                direction = DIRECTIONS[stage]
                ch = 2 * parity + direction
                if config["noise"]["placement"] == "after_cnot":
                    m.noise(singles[t, ch], pairs[t, ch], scale)
                m.sweep({q: cnots[direction] for q in qs})
                if config["noise"]["placement"] == "before_cnot":
                    m.noise(singles[t, ch], pairs[t, ch], scale)
                m.rotate(qs, rotations[stage])
    value, logabs = m.expectation(unsigned)
    if not unsigned and (value is None or abs(value) > 1 + 1e-6):
        raise FloatingPointError(
            "Signed expectation outside [-1,1]; increase bond cap / inspect numerics"
        )
    return dict(
        value=value,
        log_abs=logabs,
        max_bond=m.max_bond,
        discarded_relative_sum=m.discarded,
        runtime_s=time.perf_counter() - start,
    )


def modes(signed, log_unsigned, zero_tol=1e-12):
    """First point is s=0. Return stable Q fit basis even when Q+≈Q-."""
    o = np.asarray(signed, float)
    ratio = o / o[0] if abs(o[0]) > zero_tol else None
    if any(v is None for v in log_unsigned):
        return dict(
            ratio=ratio,
            Q_plus=None,
            Q_minus=None,
            Q_avg=None,
            Q_diff=None,
            Q_contrast=None,
            status="zero_or_nonpositive_absolute_mass",
        )
    a = np.asarray(log_unsigned, float)
    qabs = np.exp(a - a[0])
    inv_a0 = np.exp(-a[0]) if a[0] < 745 else 0.0
    z0 = o[0] * inv_a0
    zs = o * inv_a0
    den = 1 - z0 * z0
    ratio = o / o[0] if abs(o[0]) > zero_tol else None
    if den <= 1e-12:
        return dict(
            ratio=ratio,
            Q_plus=None,
            Q_minus=None,
            Q_avg=None,
            Q_diff=None,
            Q_contrast=None,
            status="undefined_sector_normalization",
        )
    avg = (qabs - zs * z0) / den
    contrast = (
        o - qabs * o[0]
    )  # Qdiff = contrast/[A(0)*(1-z0^2)], scale before fitting.
    diff = contrast * inv_a0 / den
    return dict(
        ratio=ratio,
        Q_plus=avg + diff,
        Q_minus=avg - diff,
        Q_avg=avg,
        Q_diff=diff,
        Q_contrast=contrast,
        status="ok",
    )
