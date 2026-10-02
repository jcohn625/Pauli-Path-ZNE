"""Independent forward statevector / density-matrix references."""

import numpy as np
from .circuit import (
    PAULIS,
    DIRECTIONS,
    gates,
    cnot,
    rotation_blocks,
    apply_axes,
    singlet_state,
    observable_action,
)


def statevector_value(config):
    n = config["circuit"]["N"]
    state = singlet_state(n).reshape((2,) * n)
    phi = config["circuit"]["phi"]
    swap = np.eye(4)[[0, 2, 1, 3]]
    # Equivalent to exp[-i phi/2 (XX+YY+ZZ)] up to global phase.
    u = np.cos(phi) * np.eye(4) - 1j * np.sin(phi) * swap
    for _ in range(config["circuit"]["L"]):
        for parity in (0, 1):
            for q in gates(n, parity):
                state = apply_axes(state, u, (q, q + 1))
    flat = state.reshape(-1)
    return float(np.vdot(flat, observable_action(flat, config["circuit"]["O"])).real)


def density_value(config, singles, pairs, scale):
    n = config["circuit"]["N"]
    psi = singlet_state(n)
    rho = np.outer(psi, psi.conj()).reshape((2,) * (2 * n))
    blocks = rotation_blocks(config["circuit"]["phi"])

    def sandwich(r, u, axes):
        return apply_axes(apply_axes(r, u, axes), u.conj(), tuple(q + n for q in axes))

    def noise(r, t, ch):
        for q in range(n):
            for a in range(3):
                rate = singles[t, ch, q, a] * scale
                if rate:
                    p = -0.5 * np.expm1(-2 * rate)
                    r = (1 - p) * r + p * sandwich(r, PAULIS[a + 1], (q,))
        for q in range(n - 1):
            for a in range(3):
                for b in range(3):
                    rate = pairs[t, ch, q, a, b] * scale
                    if rate:
                        p = -0.5 * np.expm1(-2 * rate)
                        changed = sandwich(
                            sandwich(r, PAULIS[a + 1], (q,)), PAULIS[b + 1], (q + 1,)
                        )
                        r = (1 - p) * r + p * changed
        return r

    def rotations(r, qs, block):
        for q in qs:
            r = sandwich(r, blocks[block][0], (q,))
            r = sandwich(r, blocks[block][1], (q + 1,))
        return r

    for t in range(config["circuit"]["L"]):
        for parity in (0, 1):
            qs = gates(n, parity)
            if not qs:
                continue
            rho = rotations(rho, qs, 0)
            for stage, d in enumerate(DIRECTIONS):
                ch = 2 * parity + d
                if config["noise"]["placement"] == "before_cnot":
                    rho = noise(rho, t, ch)
                for q in qs:
                    rho = sandwich(rho, cnot(d), (q, q + 1))
                if config["noise"]["placement"] == "after_cnot":
                    rho = noise(rho, t, ch)
                rho = rotations(rho, qs, stage + 1)
    # Tr(O rho) without constructing the full observable matrix.
    changed = rho
    for q, p in enumerate(config["circuit"]["O"]):
        if p != "I":
            changed = apply_axes(changed, PAULIS["IXYZ".index(p)], (q,))
    value = np.trace(changed.reshape(2**n, 2**n))
    trace = np.trace(rho.reshape(2**n, 2**n))
    if abs(value.imag) > 1e-9 or abs(trace - 1) > 1e-9:
        raise FloatingPointError("Density reference lost Hermiticity or normalization")
    return float(value.real)
