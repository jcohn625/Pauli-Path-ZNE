"""Open-chain Heisenberg circuit, elementary gates, and Pauli-Lindblad channels."""

from functools import lru_cache
import numpy as np

PAULIS = np.array(
    [np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex
)
CHANNELS = ("even_lr", "even_rl", "odd_lr", "odd_rl")
DIRECTIONS = (1, 0, 1)  # right->left, left->right, right->left
ANTI = np.array([[int(a != 0 and a != b) for b in (1, 2, 3)] for a in range(4)])


def rotation(axis, angle):
    return (
        np.cos(angle / 2) * PAULIS[0]
        - 1j * np.sin(angle / 2) * PAULIS["IXYZ".index(axis)]
    )


def rotation_blocks(phi):
    return (
        (PAULIS[0], rotation("Z", -np.pi / 2)),
        (PAULIS[0], rotation("Y", np.pi / 2 - phi)),
        (rotation("Z", phi - np.pi / 2), rotation("Y", phi - np.pi / 2)),
        (rotation("Z", np.pi / 2), PAULIS[0]),
    )


def cnot(direction):
    return np.eye(4, dtype=complex)[[0, 1, 3, 2] if direction == 0 else [0, 3, 2, 1]]


def ptm(u):
    """Heisenberg transfer, out/in Pauli indices; elementary gates only."""
    basis = (
        PAULIS
        if u.shape == (2, 2)
        else np.array([np.kron(a, b) for a in PAULIS for b in PAULIS])
    )
    out = np.empty((len(basis), len(basis)))
    for j, p in enumerate(basis):
        evolved = np.einsum("ji,jk,kl->il", u.conj(), p, u, optimize=False)
        out[:, j] = (
            np.einsum("aij,ji->a", basis, evolved, optimize=False).real / u.shape[0]
        )
    out[abs(out) < 1e-14] = 0
    return out


@lru_cache(maxsize=32)
def transfers(phi):
    return tuple(tuple(ptm(u) for u in block) for block in rotation_blocks(phi)), tuple(
        ptm(cnot(d)).reshape(4, 4, 4, 4) for d in (0, 1)
    )


def gates(n, parity):
    return tuple(range(parity, n - 1, 2))


def noise_diagonals(singles, pairs, scale):
    local = np.exp(-2 * scale * np.einsum("pa,qa->qp", ANTI, singles))
    xor = np.logical_xor(ANTI[:, None, :, None], ANTI[None, :, None, :])
    pair = np.exp(-2 * scale * np.einsum("pqab,jab->jpq", xor, pairs))
    return local, pair


def singlet_state(n):
    state = np.array([1.0 + 0j])
    singlet = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    for _ in range(n // 2):
        state = np.kron(state, singlet)
    return state


def apply_axes(array, unitary, axes):
    """Apply a small matrix to selected tensor axes, preserving their order."""
    axes = tuple(axes)
    rest = tuple(i for i in range(array.ndim) if i not in axes)
    order = axes + rest
    shape = [array.shape[i] for i in order]
    front = array.transpose(order).reshape(unitary.shape[1], -1)
    evolved = np.einsum("ij,jk->ik", unitary, front, optimize=False)
    return evolved.reshape(shape).transpose(np.argsort(order))


def observable_action(state, observable):
    out = state.reshape((2,) * len(observable))
    for q, p in enumerate(observable):
        if p != "I":
            out = apply_axes(out, PAULIS["IXYZ".index(p)], (q,))
    return out.reshape(-1)
