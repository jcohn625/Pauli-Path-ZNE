"""
Blocked Pauli-basis MPS solver for 1D noisy Heisenberg brickwall circuits.

The solver evolves a final Pauli observable backward in the local Pauli basis
and contracts the result against an initial product of two-qubit singlets on
even bonds (0,1), (2,3), ...

Encoding:
    single-site Pauli: I=0, X=1, Y=2, Z=3
    blocked even bond: code = 4 * left + right, local dimension 16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import numba as nb
from scipy.sparse.linalg import svds


CHAR_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
INT_TO_CHAR = np.array(["I", "X", "Y", "Z"])

NoiseModel = Literal["legacy_sum", "independent"]
NoisePlacement = Literal["gate", "touched_gate", "layer"]
KernelMode = Literal["auto", "parallel", "serial"]


def encode_2q(s: str) -> int:
    return 4 * CHAR_TO_INT[s[0]] + CHAR_TO_INT[s[1]]


def decode_2q(code: int) -> str:
    return str(INT_TO_CHAR[code // 4]) + str(INT_TO_CHAR[code % 4])


P2 = {
    "II": ["II"],
    "XX": ["XX"],
    "YY": ["YY"],
    "ZZ": ["ZZ"],
    "XI": ["XI", "IX", "YZ", "ZY"],
    "YI": ["YI", "IY", "ZX", "XZ"],
    "ZI": ["ZI", "IZ", "XY", "YX"],
    "IX": ["IX", "XI", "ZY", "YZ"],
    "IY": ["IY", "YI", "XZ", "ZX"],
    "IZ": ["IZ", "ZI", "YX", "XY"],
    "XY": ["XY", "YX", "ZI", "IZ"],
    "YX": ["YX", "XY", "IZ", "ZI"],
    "ZX": ["ZX", "XZ", "YI", "IY"],
    "XZ": ["XZ", "ZX", "IY", "YI"],
    "YZ": ["YZ", "ZY", "XI", "IX"],
    "ZY": ["ZY", "YZ", "XI", "IX"],
}

COMMUTING_2Q = ("II", "XX", "YY", "ZZ")


@dataclass
class PauliMPS:
    tensors: list[np.ndarray]
    log_scale: float = 0.0

    @property
    def n_blocks(self) -> int:
        return len(self.tensors)

    @property
    def n_qubits(self) -> int:
        return 2 * len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        dims = [self.tensors[0].shape[0]]
        dims.extend(int(t.shape[2]) for t in self.tensors)
        return dims


@dataclass
class _MPSIsland:
    left: int
    right: int
    mps: PauliMPS


def build_heisenberg_pauli_transfer(phi: float):
    """
    Build sparse local Pauli-transfer data for a two-qubit Heisenberg gate.

    Returns
    -------
    trans_codes : (16, 4) int8
        Output code for each input code and local branch.
    coeffs : (16, 4) float64
        Signed transfer coefficient for each branch.
    n_branches : (16,) int64
        Number of valid branches.

    This follows the sign convention used by the existing Pauli-path modules.
    Passing ``-phi`` gives the inverse local transfer.
    """
    trans_codes = np.full((16, 4), -1, dtype=np.int8)
    coeffs = np.zeros((16, 4), dtype=np.float64)
    n_branches = np.zeros(16, dtype=np.int64)
    is_commuting = np.zeros(16, dtype=np.bool_)
    branch_sign = np.ones((16, 4), dtype=np.int8)

    c = np.cos(phi)
    s = np.sin(phi)
    sin2 = np.sin(2.0 * phi)
    abs_sin2 = abs(sin2)

    for s2q in COMMUTING_2Q:
        is_commuting[encode_2q(s2q)] = True

    for key, outs in P2.items():
        in_code = encode_2q(key)
        for k, s_out in enumerate(outs):
            trans_codes[in_code, k] = encode_2q(s_out)

    neg_pos = ["XI", "YI", "ZI", "IX", "IY", "IZ", "ZY"]
    pos_neg = ["XY", "YX", "ZX", "XZ", "YZ"]

    for key in neg_pos:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = -1
        branch_sign[in_code, 3] = +1

    for key in pos_neg:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = +1
        branch_sign[in_code, 3] = -1

    if sin2 < 0.0:
        branch_sign[:, 2] *= -1
        branch_sign[:, 3] *= -1

    for code in range(16):
        if is_commuting[code]:
            trans_codes[code, 0] = np.int8(code)
            coeffs[code, 0] = 1.0
            n_branches[code] = 1
        else:
            coeffs[code, 0] = c * c
            coeffs[code, 1] = s * s
            coeffs[code, 2] = 0.5 * abs_sin2 * int(branch_sign[code, 2])
            coeffs[code, 3] = 0.5 * abs_sin2 * int(branch_sign[code, 3])
            n_branches[code] = 4

    return trans_codes, coeffs, n_branches


def build_odd_pair_transfer_tables(trans_codes, coeffs, n_branches):
    """
    Precompute odd-bond updates between neighboring blocked sites.

    Odd gates act on the right Pauli of the left block and the left Pauli of
    the right block. The tight Numba kernel sees every pair of block-local
    codes many times, so decoding and recoding those pairs once avoids a large
    amount of repeated integer work.
    """
    odd_new_a = np.empty((16, 16, 4), dtype=np.int8)
    odd_new_b = np.empty((16, 16, 4), dtype=np.int8)
    odd_coeffs = np.zeros((16, 16, 4), dtype=np.float64)
    odd_n_branches = np.empty((16, 16), dtype=np.int64)

    for code_a in range(16):
        a_left = code_a // 4
        a_right = code_a - 4 * a_left
        for code_b in range(16):
            b_left = code_b // 4
            b_right = code_b - 4 * b_left
            mid_code = 4 * a_right + b_left
            n = int(n_branches[mid_code])
            odd_n_branches[code_a, code_b] = n
            for k in range(n):
                out_mid = int(trans_codes[mid_code, k])
                odd_new_a[code_a, code_b, k] = 4 * a_left + (out_mid // 4)
                odd_new_b[code_a, code_b, k] = 4 * (out_mid % 4) + b_right
                odd_coeffs[code_a, code_b, k] = coeffs[mid_code, k]

    return odd_new_a, odd_new_b, odd_coeffs, odd_n_branches


def parse_pauli_string(final_pauli: str | Sequence[int] | np.ndarray, n_qubits: int) -> np.ndarray:
    """Return an int8 Pauli vector of length ``n_qubits``."""
    if isinstance(final_pauli, str):
        s = final_pauli.replace(" ", "").upper()
        if len(s) != n_qubits:
            raise ValueError(f"Pauli string has length {len(s)}, expected {n_qubits}.")
        try:
            return np.array([CHAR_TO_INT[ch] for ch in s], dtype=np.int8)
        except KeyError as exc:
            raise ValueError("Pauli string must contain only I, X, Y, Z.") from exc

    arr = np.asarray(final_pauli, dtype=np.int8)
    if arr.shape != (n_qubits,):
        raise ValueError(f"final_pauli must have shape ({n_qubits},).")
    if np.any((arr < 0) | (arr > 3)):
        raise ValueError("final_pauli entries must be in {0, 1, 2, 3}.")
    return arr


def pauli_zz(n_qubits: int, i: int, j: int) -> np.ndarray:
    """Convenience constructor for a Z_i Z_j target observable."""
    p = np.zeros(n_qubits, dtype=np.int8)
    p[i] = 3
    p[j] = 3
    return p


def pauli_parity_vector(pauli: np.ndarray) -> tuple[int, int, int]:
    """Return ``(n_X mod 2, n_Y mod 2, n_Z mod 2)``."""
    px = int(np.count_nonzero(pauli == 1) & 1)
    py = int(np.count_nonzero(pauli == 2) & 1)
    pz = int(np.count_nonzero(pauli == 3) & 1)
    return px, py, pz


def has_singlet_compatible_global_parity(pauli: np.ndarray) -> bool:
    """
    Check the global Heisenberg relative-parity sector.

    Local Heisenberg Pauli updates can flip all three Pauli-count parities
    together, so the conserved data are ``px xor py`` and ``px xor pz``. The
    singlet-product contraction lies in the sector ``px == py == pz``.
    """
    px, py, pz = pauli_parity_vector(pauli)
    return px == py and px == pz


def normalize_lambda_schedule(
    lam_xyz: np.ndarray,
    n_steps: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Normalize noise inputs to shape ``(n_steps, 2, n_qubits, 3)``.

    The half-layer index is ``0`` for noise after the even layer and ``1`` for
    noise after the odd layer in the forward circuit.
    """
    lam = np.asarray(lam_xyz, dtype=np.float64)
    if lam.shape == (n_qubits, 3):
        out = np.empty((n_steps, 2, n_qubits, 3), dtype=np.float64)
        out[:, 0, :, :] = lam
        out[:, 1, :, :] = lam
        return out
    if lam.shape == (2, n_qubits, 3):
        out = np.empty((n_steps, 2, n_qubits, 3), dtype=np.float64)
        out[:, :, :, :] = lam
        return out
    if lam.shape == (n_steps, 2, n_qubits, 3):
        return np.ascontiguousarray(lam)
    if lam.shape == (2 * n_steps, n_qubits, 3):
        return np.ascontiguousarray(lam.reshape(n_steps, 2, n_qubits, 3))
    raise ValueError(
        "lam_xyz must have shape (n_qubits,3), (2,n_qubits,3), "
        "(n_steps,2,n_qubits,3), or (2*n_steps,n_qubits,3)."
    )


def pauli_diag_factors_from_lambda(lam_xyz: np.ndarray, noise_model: NoiseModel = "legacy_sum") -> np.ndarray:
    """
    Convert lambda_x/y/z rates to Pauli-basis damping factors.

    ``independent`` applies the three single-Pauli channels sequentially:
        eta_X = exp(-2 lambda_y) exp(-2 lambda_z), etc.

    ``legacy_sum`` matches several existing Pauli-path files:
        eta_X = exp(-2 lambda_y) + exp(-2 lambda_z) - 1, etc.
    """
    lam = np.asarray(lam_xyz, dtype=np.float64)
    eta = np.empty_like(lam)
    ex = np.exp(-2.0 * lam[:, 0])
    ey = np.exp(-2.0 * lam[:, 1])
    ez = np.exp(-2.0 * lam[:, 2])
    if noise_model == "independent":
        eta[:, 0] = ey * ez
        eta[:, 1] = ex * ez
        eta[:, 2] = ex * ey
    elif noise_model == "legacy_sum":
        eta[:, 0] = ey + ez - 1.0
        eta[:, 1] = ex + ez - 1.0
        eta[:, 2] = ex + ey - 1.0
    else:
        raise ValueError("noise_model must be 'independent' or 'legacy_sum'.")
    return eta


def normalize_noise_placement(noise_placement: NoisePlacement) -> Literal["gate", "layer"]:
    """Normalize noise-placement aliases."""
    if noise_placement == "touched_gate":
        return "gate"
    if noise_placement in ("gate", "layer"):
        return noise_placement
    raise ValueError("noise_placement must be 'gate', 'touched_gate', or 'layer'.")


def build_block_noise_diag(eta_xyz: np.ndarray) -> np.ndarray:
    """Return block-local diagonal noise factors with shape ``(n_blocks, 16)``."""
    n_qubits = eta_xyz.shape[0]
    if n_qubits % 2 != 0:
        raise ValueError("Blocked even-bond MPS requires an even number of qubits.")
    out = np.ones((n_qubits // 2, 16), dtype=np.float64)
    for b in range(n_qubits // 2):
        for left in range(4):
            lf = 1.0 if left == 0 else eta_xyz[2 * b, left - 1]
            for right in range(4):
                rf = 1.0 if right == 0 else eta_xyz[2 * b + 1, right - 1]
                out[b, 4 * left + right] = lf * rf
    return out


def build_touched_layer_noise_diag(eta_xyz: np.ndarray, layer: Literal["even", "odd"]) -> np.ndarray:
    """
    Return block-local noise factors for qubits touched by one gate layer.

    This matches ``Pauli_path_Heis.evolve_many_paths_with_1q_noise``: after
    each two-qubit gate, only the two qubits in that gate receive damping.
    Gates inside one even/odd layer are disjoint, so this is equivalent to one
    diagonal layer over the touched qubits.
    """
    n_qubits = eta_xyz.shape[0]
    if n_qubits % 2 != 0:
        raise ValueError("Blocked even-bond MPS requires an even number of qubits.")
    touched = np.zeros(n_qubits, dtype=np.bool_)
    if layer == "even":
        for q in range(0, n_qubits - 1, 2):
            touched[q] = True
            touched[q + 1] = True
    elif layer == "odd":
        for q in range(1, n_qubits - 1, 2):
            touched[q] = True
            touched[q + 1] = True
    else:
        raise ValueError("layer must be 'even' or 'odd'.")

    out = np.ones((n_qubits // 2, 16), dtype=np.float64)
    for b in range(n_qubits // 2):
        for left in range(4):
            lf = 1.0
            if touched[2 * b] and left != 0:
                lf = eta_xyz[2 * b, left - 1]
            for right in range(4):
                rf = 1.0
                if touched[2 * b + 1] and right != 0:
                    rf = eta_xyz[2 * b + 1, right - 1]
                out[b, 4 * left + right] = lf * rf
    return out


def product_pauli_mps(pauli: np.ndarray) -> PauliMPS:
    """Create a rank-1 blocked MPS for one Pauli string."""
    if pauli.shape[0] % 2 != 0:
        raise ValueError("Blocked even-bond MPS requires an even number of qubits.")
    tensors = []
    for b in range(pauli.shape[0] // 2):
        code = 4 * int(pauli[2 * b]) + int(pauli[2 * b + 1])
        a = np.zeros((1, 16, 1), dtype=np.float64)
        a[0, code, 0] = 1.0
        tensors.append(a)
    return PauliMPS(tensors=tensors)


def _identity_block_tensor() -> np.ndarray:
    a = np.zeros((1, 16, 1), dtype=np.float64)
    a[0, 0, 0] = 1.0
    return a


def _is_identity_block_tensor(A: np.ndarray) -> bool:
    return (
        A.shape == (1, 16, 1)
        and A[0, 0, 0] == 1.0
        and np.count_nonzero(A) == 1
    )


def _identity_block_mps(n_blocks: int) -> PauliMPS:
    return PauliMPS([_identity_block_tensor() for _ in range(n_blocks)])


def _zero_block_mps(n_blocks: int) -> PauliMPS:
    mps = _identity_block_mps(n_blocks)
    if n_blocks > 0:
        mps.tensors[0] = np.zeros_like(mps.tensors[0])
    return mps


def _initial_islands_from_pauli(pauli: np.ndarray) -> list[_MPSIsland]:
    active = []
    for b in range(pauli.shape[0] // 2):
        code = 4 * int(pauli[2 * b]) + int(pauli[2 * b + 1])
        if code != 0:
            active.append(b)

    if not active:
        return []

    intervals = []
    start = active[0]
    prev = active[0]
    for b in active[1:]:
        if b == prev + 1:
            prev = b
        else:
            intervals.append((start, prev))
            start = b
            prev = b
    intervals.append((start, prev))

    islands = []
    for left, right in intervals:
        local_pauli = pauli[2 * left : 2 * (right + 1)]
        islands.append(_MPSIsland(left, right, product_pauli_mps(local_pauli)))
    return islands


def _expand_island(island: _MPSIsland, n_blocks: int) -> _MPSIsland:
    left = island.left
    right = island.right
    tensors = island.mps.tensors
    if left > 0:
        left -= 1
        tensors = [_identity_block_tensor()] + tensors
    if right + 1 < n_blocks:
        right += 1
        tensors = tensors + [_identity_block_tensor()]
    return _MPSIsland(left, right, PauliMPS(tensors, island.mps.log_scale))


def _merge_touching_islands(islands: list[_MPSIsland]) -> list[_MPSIsland]:
    if not islands:
        return []
    islands = sorted(islands, key=lambda x: x.left)
    merged = [islands[0]]
    for island in islands[1:]:
        cur = merged[-1]
        if island.left <= cur.right + 1:
            skip = max(0, cur.right - island.left + 1)
            tensors = cur.mps.tensors + island.mps.tensors[skip:]
            merged[-1] = _MPSIsland(
                cur.left,
                max(cur.right, island.right),
                PauliMPS(tensors, cur.mps.log_scale + island.mps.log_scale),
            )
        else:
            merged.append(island)
    return merged


def _full_mps_from_islands(islands: list[_MPSIsland], n_blocks: int) -> PauliMPS:
    tensors = []
    pos = 0
    log_scale = 0.0
    for island in sorted(islands, key=lambda x: x.left):
        while pos < island.left:
            tensors.append(_identity_block_tensor())
            pos += 1
        tensors.extend(island.mps.tensors)
        log_scale += island.mps.log_scale
        pos = island.right + 1
    while pos < n_blocks:
        tensors.append(_identity_block_tensor())
        pos += 1
    if not tensors:
        return _identity_block_mps(n_blocks)
    return PauliMPS(tensors, log_scale)


def _contract_islands_with_singlet_product(islands: list[_MPSIsland]) -> float:
    value = 1.0
    for island in islands:
        value *= contract_with_singlet_product(island.mps)
    return float(value)


@nb.njit(cache=True, parallel=True)
def _apply_one_site_transfer(A, trans_codes, coeffs, n_branches):
    chi_l, _, chi_r = A.shape
    out = np.zeros_like(A)
    for lr in nb.prange(chi_l * chi_r):
        l = lr // chi_r
        r = lr - l * chi_r
        for code in range(16):
            val = A[l, code, r]
            if val == 0.0:
                continue
            for k in range(n_branches[code]):
                out_code = int(trans_codes[code, k])
                out[l, out_code, r] += coeffs[code, k] * val
    return out


@nb.njit(cache=True)
def _apply_one_site_transfer_serial(A, trans_codes, coeffs, n_branches):
    chi_l, _, chi_r = A.shape
    out = np.zeros_like(A)
    for l in range(chi_l):
        for r in range(chi_r):
            for code in range(16):
                val = A[l, code, r]
                if val == 0.0:
                    continue
                for k in range(n_branches[code]):
                    out_code = int(trans_codes[code, k])
                    out[l, out_code, r] += coeffs[code, k] * val
    return out


@nb.njit(cache=True, parallel=True)
def _apply_one_site_diag_inplace(A, diag):
    n = A.shape[0] * A.shape[1] * A.shape[2]
    chi_r = A.shape[2]
    for idx in nb.prange(n):
        r = idx % chi_r
        t = idx // chi_r
        p = t % 16
        l = t // 16
        A[l, p, r] *= diag[p]


@nb.njit(cache=True)
def _apply_one_site_diag_inplace_serial(A, diag):
    chi_r = A.shape[2]
    for l in range(A.shape[0]):
        for p in range(16):
            d = diag[p]
            for r in range(chi_r):
                A[l, p, r] *= d


@nb.njit(cache=True, parallel=True)
def _apply_odd_transfer_theta(theta, odd_new_a, odd_new_b, odd_coeffs, odd_n_branches):
    chi_l, _, _, chi_r = theta.shape
    out = np.zeros_like(theta)
    for lr in nb.prange(chi_l * chi_r):
        l = lr // chi_r
        r = lr - l * chi_r
        for code_a in range(16):
            for code_b in range(16):
                val = theta[l, code_a, code_b, r]
                if val == 0.0:
                    continue
                for k in range(odd_n_branches[code_a, code_b]):
                    new_a = int(odd_new_a[code_a, code_b, k])
                    new_b = int(odd_new_b[code_a, code_b, k])
                    out[l, new_a, new_b, r] += odd_coeffs[code_a, code_b, k] * val
    return out


@nb.njit(cache=True)
def _apply_odd_transfer_theta_serial(theta, odd_new_a, odd_new_b, odd_coeffs, odd_n_branches):
    chi_l, _, _, chi_r = theta.shape
    out = np.zeros_like(theta)
    for l in range(chi_l):
        for r in range(chi_r):
            for code_a in range(16):
                for code_b in range(16):
                    val = theta[l, code_a, code_b, r]
                    if val == 0.0:
                        continue
                    for k in range(odd_n_branches[code_a, code_b]):
                        new_a = int(odd_new_a[code_a, code_b, k])
                        new_b = int(odd_new_b[code_a, code_b, k])
                        out[l, new_a, new_b, r] += odd_coeffs[code_a, code_b, k] * val
    return out


def apply_even_layer_inplace(
    mps: PauliMPS,
    trans_codes,
    coeffs,
    n_branches,
    parallel_kernels: bool | KernelMode = "auto",
) -> None:
    """Apply all backward even-bond Heisenberg gates as one-site updates."""
    for i, A in enumerate(mps.tensors):
        if _is_identity_block_tensor(A):
            continue
        use_parallel = parallel_kernels is True or parallel_kernels == "parallel"
        if parallel_kernels == "auto":
            use_parallel = A.shape[0] * A.shape[2] >= 32768
        if use_parallel:
            mps.tensors[i] = _apply_one_site_transfer(A, trans_codes, coeffs, n_branches)
        else:
            mps.tensors[i] = _apply_one_site_transfer_serial(A, trans_codes, coeffs, n_branches)


def apply_block_noise_inplace(
    mps: PauliMPS,
    block_diag: np.ndarray,
    parallel_kernels: bool | KernelMode = "auto",
) -> None:
    """Apply diagonal single-qubit Pauli noise factors to every blocked site."""
    for i, A in enumerate(mps.tensors):
        if _is_identity_block_tensor(A):
            continue
        use_parallel = parallel_kernels is True or parallel_kernels == "parallel"
        if parallel_kernels == "auto":
            use_parallel = A.size >= 524288
        if use_parallel:
            _apply_one_site_diag_inplace(A, block_diag[i])
        else:
            _apply_one_site_diag_inplace_serial(A, block_diag[i])


def right_canonicalize_inplace(mps: PauliMPS) -> None:
    """
    Put the MPS in right-canonical form by QR sweeps from right to left.

    This is important before a left-to-right two-site compression sweep after
    nonunitary one-site Pauli transfer/noise maps have changed the MPS gauge.
    """
    for site in range(mps.n_blocks - 1, 0, -1):
        A = mps.tensors[site]
        if _is_identity_block_tensor(A):
            continue
        chi_l, d, chi_r = A.shape
        mat = A.reshape(chi_l, d * chi_r)
        Q_t, R_t = np.linalg.qr(mat.T, mode="reduced")
        chi_new = Q_t.shape[1]
        mps.tensors[site] = Q_t.T.reshape(chi_new, d, chi_r)

        L = R_t.T
        B = mps.tensors[site - 1]
        mps.tensors[site - 1] = np.einsum("lpr,ra->lpa", B, L, optimize=True)


def _svd_truncate(
    mat: np.ndarray,
    chi_max: int | None,
    cutoff: float,
    svd_method: Literal["auto", "full", "sparse", "randomized"] = "auto",
    oversample: int = 24,
    n_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    target = None
    if chi_max is not None:
        target = int(chi_max)
        if cutoff > 0.0:
            target = min(target + oversample, min(mat.shape))

    use_randomized = False
    use_sparse = False
    if target is not None and target < min(mat.shape):
        if svd_method == "sparse":
            use_sparse = True
        elif svd_method == "randomized":
            use_randomized = True
        elif svd_method == "auto" and min(mat.shape) >= 4 * max(1, int(chi_max)):
            use_randomized = True

    if use_sparse:
        U, S, Vh = _sparse_svd(mat, rank=target)
    elif use_randomized:
        U, S, Vh = _randomized_svd(mat, rank=target, n_iter=n_iter)
    else:
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

    return _truncate_svd_factors(U, S, Vh, chi_max=chi_max, cutoff=cutoff)


def _truncate_svd_factors(
    U: np.ndarray,
    S: np.ndarray,
    Vh: np.ndarray,
    chi_max: int | None,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if S.size == 0:
        return U, S, Vh, 0.0

    keep = S.size
    if cutoff > 0.0:
        keep = int(np.count_nonzero(S > cutoff))
        keep = max(1, keep)
    if chi_max is not None:
        keep = min(keep, int(chi_max))

    discarded = float(np.sum(S[keep:] * S[keep:]))
    return U[:, :keep], S[:keep], Vh[:keep, :], discarded


def _randomized_svd(
    mat: np.ndarray,
    rank: int,
    n_iter: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate leading singular vectors using a randomized range finder.

    This is meant for TEBD truncation when ``chi_max`` is far below the full
    two-site matrix dimension. It returns at most ``rank`` singular values.
    """
    m, n = mat.shape
    rank = max(1, min(int(rank), m, n))
    rng = np.random.default_rng(0)

    if m >= n:
        omega = rng.standard_normal((n, rank), dtype=np.float64)
        Y = mat @ omega
        for _ in range(max(0, n_iter)):
            Y = mat @ (mat.T @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ mat
        Uh, S, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uh
    else:
        omega = rng.standard_normal((m, rank), dtype=np.float64)
        Y = mat.T @ omega
        for _ in range(max(0, n_iter)):
            Y = mat.T @ (mat @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = mat @ Q
        U, S, Vht = np.linalg.svd(B, full_matrices=False)
        Vh = Vht @ Q.T

    return U[:, :rank], S[:rank], Vh[:rank, :]


def _sparse_svd(mat: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Leading singular triplets via ARPACK/PROPACK-backed SciPy ``svds``.

    This is usually slower than randomized SVD but more stable for observables
    whose contraction is sensitive to local truncation errors.
    """
    m, n = mat.shape
    rank = max(1, min(int(rank), min(m, n) - 1))
    try:
        U, S, Vh = svds(mat, k=rank, which="LM", solver="propack")
    except Exception:
        # Low-rank local matrices can make PROPACK report an invariant subspace.
        # Full SVD is acceptable for these cases and keeps the solver robust.
        return np.linalg.svd(mat, full_matrices=False)
    order = np.argsort(S)[::-1]
    return U[:, order], S[order], Vh[order, :]


def apply_odd_layer_inplace(
    mps: PauliMPS,
    trans_codes,
    coeffs,
    n_branches,
    chi_max: int | None = None,
    cutoff: float = 1e-12,
    svd_method: Literal["auto", "full", "sparse", "randomized"] = "auto",
    oversample: int = 24,
    n_iter: int = 1,
    canonicalize: bool = True,
    parallel_kernels: bool | KernelMode = "auto",
    odd_pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """
    Apply all backward odd-bond Heisenberg gates as two-block TEBD updates.

    Returns the total discarded singular-value weight for this odd layer.
    """
    if canonicalize and mps.n_blocks > 1:
        right_canonicalize_inplace(mps)

    if odd_pair_tables is None:
        odd_pair_tables = build_odd_pair_transfer_tables(trans_codes, coeffs, n_branches)
    odd_new_a, odd_new_b, odd_coeffs, odd_n_branches = odd_pair_tables

    total_discarded = 0.0
    for site in range(mps.n_blocks - 1):
        A = mps.tensors[site]
        B = mps.tensors[site + 1]
        if _is_identity_block_tensor(A) and _is_identity_block_tensor(B):
            continue
        theta = np.einsum("lam,mbr->labr", A, B, optimize=True)
        use_parallel = parallel_kernels is True or parallel_kernels == "parallel"
        if parallel_kernels == "auto":
            use_parallel = A.shape[0] * B.shape[2] >= 32768
        if use_parallel:
            theta = _apply_odd_transfer_theta(
                theta, odd_new_a, odd_new_b, odd_coeffs, odd_n_branches
            )
        else:
            theta = _apply_odd_transfer_theta_serial(
                theta, odd_new_a, odd_new_b, odd_coeffs, odd_n_branches
            )

        chi_l, d1, d2, chi_r = theta.shape
        mat = theta.reshape(chi_l * d1, d2 * chi_r)
        U, S, Vh, discarded = _svd_truncate(
            mat,
            chi_max=chi_max,
            cutoff=cutoff,
            svd_method=svd_method,
            oversample=oversample,
            n_iter=n_iter,
        )
        total_discarded += discarded

        chi_new = S.shape[0]
        mps.tensors[site] = U.reshape(chi_l, d1, chi_new)
        mps.tensors[site + 1] = (S[:, None] * Vh).reshape(chi_new, d2, chi_r)
    return total_discarded


def singlet_block_functional() -> np.ndarray:
    """Local contraction vector for an initial two-qubit singlet."""
    f = np.zeros(16, dtype=np.float64)
    f[encode_2q("II")] = 1.0
    f[encode_2q("XX")] = -1.0
    f[encode_2q("YY")] = -1.0
    f[encode_2q("ZZ")] = -1.0
    return f


def contract_with_singlet_product(mps: PauliMPS) -> float:
    """Contract a blocked Pauli-basis MPS with the singlet-product state."""
    f = singlet_block_functional()
    env = np.ones(1, dtype=np.float64)
    for A in mps.tensors:
        env = np.einsum("l, lpr, p -> r", env, A, f, optimize=True)
    return float(env[0] * np.exp(mps.log_scale))


def _evolve_observable_backward_lightcone(
    pauli: np.ndarray,
    *,
    n_qubits: int,
    trans_codes,
    coeffs,
    n_branches,
    odd_pair_tables: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lam_schedule: np.ndarray,
    n_steps: int,
    chi_max: int | None,
    cutoff: float,
    svd_method: Literal["auto", "full", "sparse", "randomized"],
    oversample: int,
    n_iter: int,
    canonicalize: bool,
    parallel_kernels: bool | KernelMode,
    noise_model: NoiseModel,
    noise_placement: Literal["gate", "layer"],
    return_mps: bool,
) -> float | tuple[float, PauliMPS, dict]:
    n_blocks = n_qubits // 2
    islands = _initial_islands_from_pauli(pauli)

    if not islands:
        value = 1.0
        if return_mps:
            info = {
                "discarded_by_backward_step": np.zeros((n_steps, 1), dtype=np.float64),
                "max_bond_by_backward_step": np.ones(n_steps + 1, dtype=np.int64),
                "bond_dims": np.ones(n_blocks + 1, dtype=np.int64),
                "noise_model": noise_model,
                "noise_placement": noise_placement,
                "svd_method": svd_method,
                "parallel_kernels": parallel_kernels,
                "use_lightcone": True,
                "active_intervals": [],
            }
            return value, _identity_block_mps(n_blocks), info
        return value

    discarded_by_step = np.zeros((n_steps, 1), dtype=np.float64)
    max_bond_by_step = np.zeros(n_steps + 1, dtype=np.int64)
    active_counts = np.zeros(n_steps + 1, dtype=np.int64)
    interval_history: list[list[tuple[int, int]]] = []

    max_bond_by_step[0] = max(max(island.mps.bond_dims) for island in islands)
    active_counts[0] = sum(island.right - island.left + 1 for island in islands)
    interval_history.append([(island.left, island.right) for island in islands])

    for t in range(n_steps - 1, -1, -1):
        islands = [_expand_island(island, n_blocks) for island in islands]
        islands = _merge_touching_islands(islands)

        eta_odd = pauli_diag_factors_from_lambda(lam_schedule[t, 1], noise_model)
        eta_even = pauli_diag_factors_from_lambda(lam_schedule[t, 0], noise_model)
        if noise_placement == "layer":
            odd_diag = build_block_noise_diag(eta_odd)
            even_diag = build_block_noise_diag(eta_even)
        else:
            odd_diag = build_touched_layer_noise_diag(eta_odd, "odd")
            even_diag = build_touched_layer_noise_diag(eta_even, "even")

        discarded = 0.0
        new_islands = []
        for island in islands:
            sl = slice(island.left, island.right + 1)
            if noise_placement == "layer":
                apply_block_noise_inplace(
                    island.mps, odd_diag[sl], parallel_kernels=parallel_kernels
                )
                discarded += apply_odd_layer_inplace(
                    island.mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    chi_max=chi_max,
                    cutoff=cutoff,
                    svd_method=svd_method,
                    oversample=oversample,
                    n_iter=n_iter,
                    canonicalize=canonicalize,
                    parallel_kernels=parallel_kernels,
                    odd_pair_tables=odd_pair_tables,
                )
                apply_block_noise_inplace(
                    island.mps, even_diag[sl], parallel_kernels=parallel_kernels
                )
                apply_even_layer_inplace(
                    island.mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    parallel_kernels=parallel_kernels,
                )
            else:
                discarded += apply_odd_layer_inplace(
                    island.mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    chi_max=chi_max,
                    cutoff=cutoff,
                    svd_method=svd_method,
                    oversample=oversample,
                    n_iter=n_iter,
                    canonicalize=canonicalize,
                    parallel_kernels=parallel_kernels,
                    odd_pair_tables=odd_pair_tables,
                )
                apply_block_noise_inplace(
                    island.mps, odd_diag[sl], parallel_kernels=parallel_kernels
                )
                apply_even_layer_inplace(
                    island.mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    parallel_kernels=parallel_kernels,
                )
                apply_block_noise_inplace(
                    island.mps, even_diag[sl], parallel_kernels=parallel_kernels
                )
            new_islands.append(island)

        islands = _merge_touching_islands(new_islands)
        step_index = n_steps - 1 - t
        discarded_by_step[step_index, 0] = discarded
        max_bond_by_step[step_index + 1] = max(max(island.mps.bond_dims) for island in islands)
        active_counts[step_index + 1] = sum(
            island.right - island.left + 1 for island in islands
        )
        interval_history.append([(island.left, island.right) for island in islands])

    value = _contract_islands_with_singlet_product(islands)
    if return_mps:
        full_mps = _full_mps_from_islands(islands, n_blocks)
        info = {
            "discarded_by_backward_step": discarded_by_step,
            "max_bond_by_backward_step": max_bond_by_step,
            "active_blocks_by_backward_step": active_counts,
            "active_intervals_by_backward_step": interval_history,
            "active_intervals": interval_history[-1],
            "bond_dims": np.array(full_mps.bond_dims, dtype=np.int64),
            "noise_model": noise_model,
            "noise_placement": noise_placement,
            "svd_method": svd_method,
            "parallel_kernels": parallel_kernels,
            "use_lightcone": True,
        }
        return value, full_mps, info
    return value


def evolve_observable_backward_mps(
    final_pauli: str | Sequence[int] | np.ndarray,
    *,
    n_qubits: int,
    phi: float,
    lam_xyz: np.ndarray,
    n_steps: int,
    chi_max: int | None = None,
    cutoff: float = 1e-12,
    svd_method: Literal["auto", "full", "sparse", "randomized"] = "auto",
    oversample: int = 24,
    n_iter: int = 1,
    canonicalize: bool = True,
    parallel_kernels: bool | KernelMode = "auto",
    use_lightcone: bool = True,
    enforce_global_parity: bool = True,
    noise_model: NoiseModel = "legacy_sum",
    noise_placement: NoisePlacement = "gate",
    return_mps: bool = False,
) -> float | tuple[float, PauliMPS, dict]:
    """
    Evolve ``final_pauli`` backward and contract with the initial singlet product.

    Parameters
    ----------
    final_pauli
        Pauli string as ``"IXYZ..."`` or integer array with I=0, X=1, Y=2, Z=3.
    n_qubits
        Even number of qubits.
    phi
        Heisenberg Pauli-transfer angle. Use ``-phi`` if you need the inverse
        local convention relative to the existing Monte Carlo tables.
    lam_xyz
        Noise lambdas. Accepted shapes are ``(n_qubits,3)``, ``(2,n_qubits,3)``,
        ``(n_steps,2,n_qubits,3)``, or ``(2*n_steps,n_qubits,3)``.
    n_steps
        Number of full even+odd Trotter steps.
    chi_max, cutoff
        TEBD compression controls for odd-layer SVDs. The small default cutoff
        removes numerically zero singular vectors that otherwise inflate bonds.
    svd_method, oversample, n_iter
        ``"auto"`` uses randomized truncated SVD when ``chi_max`` is much
        smaller than the full two-site matrix dimension. Use ``"full"`` for
        exact SVD beta tests, ``"sparse"`` for a robust truncated backend, and
        ``"randomized"`` to force the fastest approximate path.
    canonicalize
        If true, right-canonicalize before every odd-layer two-site sweep.
        This is recommended for stable compression after nonunitary noise maps.
    parallel_kernels
        ``"auto"`` chooses Numba ``prange`` kernels only for large local
        contractions. Use ``"parallel"``/``True`` or ``"serial"``/``False`` to
        force a mode for benchmarking or debugging.
    use_lightcone
        If true, evolve only active blocked intervals touched by the backward
        light cone of the final Pauli string. Multiple disjoint cones are kept
        factorized until they touch.
    enforce_global_parity
        If true, return exact zero when the target Pauli is outside the global
        relative-parity sector that can contract with an even-bond singlet
        product.
    noise_model
        ``"legacy_sum"`` matches ``Pauli_path_Heis.pauli_diag_factors_from_lambda``.
        ``"independent"`` uses sequential single-Pauli channels.
    noise_placement
        ``"gate"`` matches ``Pauli_path_Heis.evolve_many_paths_with_1q_noise``:
        apply damping only to the two qubits touched by each gate after that
        gate update. ``"layer"`` keeps the older MPS convention of applying
        full-chain damping before each backward layer.
    return_mps
        If true, return ``(value, mps, info)``.
    """
    if n_qubits % 2 != 0:
        raise ValueError("Blocked even-bond MPS requires an even number of qubits.")
    noise_placement_norm = normalize_noise_placement(noise_placement)

    pauli = parse_pauli_string(final_pauli, n_qubits)
    parity = pauli_parity_vector(pauli)
    if enforce_global_parity and not has_singlet_compatible_global_parity(pauli):
        if return_mps:
            n_blocks = n_qubits // 2
            info = {
                "discarded_by_backward_step": np.zeros((n_steps, 1), dtype=np.float64),
                "max_bond_by_backward_step": np.ones(n_steps + 1, dtype=np.int64),
                "bond_dims": np.ones(n_blocks + 1, dtype=np.int64),
                "noise_model": noise_model,
                "noise_placement": noise_placement_norm,
                "svd_method": svd_method,
                "parallel_kernels": parallel_kernels,
                "use_lightcone": use_lightcone,
                "global_pauli_parity": np.array(parity, dtype=np.int8),
                "parity_filtered": True,
            }
            return 0.0, _zero_block_mps(n_blocks), info
        return 0.0

    lam_schedule = normalize_lambda_schedule(lam_xyz, n_steps, n_qubits)
    trans_codes, coeffs, n_branches = build_heisenberg_pauli_transfer(phi)
    odd_pair_tables = build_odd_pair_transfer_tables(trans_codes, coeffs, n_branches)

    if use_lightcone:
        result = _evolve_observable_backward_lightcone(
            pauli,
            n_qubits=n_qubits,
            trans_codes=trans_codes,
            coeffs=coeffs,
            n_branches=n_branches,
            odd_pair_tables=odd_pair_tables,
            lam_schedule=lam_schedule,
            n_steps=n_steps,
            chi_max=chi_max,
            cutoff=cutoff,
            svd_method=svd_method,
            oversample=oversample,
            n_iter=n_iter,
            canonicalize=canonicalize,
            parallel_kernels=parallel_kernels,
            noise_model=noise_model,
            noise_placement=noise_placement_norm,
            return_mps=return_mps,
        )
        if return_mps:
            value, out_mps, info = result
            info["global_pauli_parity"] = np.array(parity, dtype=np.int8)
            info["parity_filtered"] = False
            return value, out_mps, info
        return result

    mps = product_pauli_mps(pauli)

    discarded_by_step = np.zeros((n_steps, 1), dtype=np.float64)
    max_bond_by_step = np.zeros(n_steps + 1, dtype=np.int64)
    max_bond_by_step[0] = max(mps.bond_dims)

    # Backward through each forward step:
    #   layer placement: physical adjoint of full half-layer damping.
    #   gate placement: matches Pauli_path_Heis' gate-update-then-touched-noise loop.
    for t in range(n_steps - 1, -1, -1):
        eta_odd = pauli_diag_factors_from_lambda(lam_schedule[t, 1], noise_model)
        eta_even = pauli_diag_factors_from_lambda(lam_schedule[t, 0], noise_model)
        if noise_placement_norm == "layer":
            apply_block_noise_inplace(
                mps, build_block_noise_diag(eta_odd), parallel_kernels=parallel_kernels
            )
            discarded = apply_odd_layer_inplace(
                mps,
                trans_codes,
                coeffs,
                n_branches,
                chi_max=chi_max,
                cutoff=cutoff,
                svd_method=svd_method,
                oversample=oversample,
                n_iter=n_iter,
                canonicalize=canonicalize,
                parallel_kernels=parallel_kernels,
                odd_pair_tables=odd_pair_tables,
            )
            apply_block_noise_inplace(
                mps, build_block_noise_diag(eta_even), parallel_kernels=parallel_kernels
            )
            apply_even_layer_inplace(
                mps, trans_codes, coeffs, n_branches, parallel_kernels=parallel_kernels
            )
        else:
            discarded = apply_odd_layer_inplace(
                mps,
                trans_codes,
                coeffs,
                n_branches,
                chi_max=chi_max,
                cutoff=cutoff,
                svd_method=svd_method,
                oversample=oversample,
                n_iter=n_iter,
                canonicalize=canonicalize,
                parallel_kernels=parallel_kernels,
                odd_pair_tables=odd_pair_tables,
            )
            apply_block_noise_inplace(
                mps,
                build_touched_layer_noise_diag(eta_odd, "odd"),
                parallel_kernels=parallel_kernels,
            )
            apply_even_layer_inplace(
                mps, trans_codes, coeffs, n_branches, parallel_kernels=parallel_kernels
            )
            apply_block_noise_inplace(
                mps,
                build_touched_layer_noise_diag(eta_even, "even"),
                parallel_kernels=parallel_kernels,
            )

        step_index = n_steps - 1 - t
        discarded_by_step[step_index, 0] = discarded
        max_bond_by_step[step_index + 1] = max(mps.bond_dims)

    value = contract_with_singlet_product(mps)
    if return_mps:
        info = {
            "discarded_by_backward_step": discarded_by_step,
            "max_bond_by_backward_step": max_bond_by_step,
            "bond_dims": np.array(mps.bond_dims, dtype=np.int64),
            "noise_model": noise_model,
            "noise_placement": noise_placement_norm,
            "svd_method": svd_method,
            "parallel_kernels": parallel_kernels,
            "use_lightcone": False,
            "global_pauli_parity": np.array(parity, dtype=np.int8),
            "parity_filtered": False,
        }
        return value, mps, info
    return value


def evolve_observable_backward_series(
    final_pauli: str | Sequence[int] | np.ndarray,
    *,
    n_qubits: int,
    phi: float,
    lam_xyz: np.ndarray,
    max_steps: int,
    chi_max: int | None = None,
    cutoff: float = 1e-12,
    svd_method: Literal["auto", "full", "sparse", "randomized"] = "auto",
    oversample: int = 24,
    n_iter: int = 1,
    canonicalize: bool = True,
    parallel_kernels: bool | KernelMode = "auto",
    use_lightcone: bool = True,
    enforce_global_parity: bool = True,
    noise_model: NoiseModel = "legacy_sum",
    noise_placement: NoisePlacement = "gate",
    return_final_mps: bool = False,
) -> np.ndarray | tuple[np.ndarray, PauliMPS, dict]:
    """
    Compute observable values for Trotter steps ``0, 1, ..., max_steps``.

    This avoids rerunning the full solver for each step count. It is intended
    for time-independent circuits, where ``lam_xyz`` has shape ``(n_qubits, 3)``
    or ``(2, n_qubits, 3)``. For a fully time-dependent noise schedule, the
    adjoint order depends on the requested final step, so use separate calls to
    ``evolve_observable_backward_mps`` unless you deliberately want repeated
    identical steps.
    """
    if n_qubits % 2 != 0:
        raise ValueError("Blocked even-bond MPS requires an even number of qubits.")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative.")
    noise_placement_norm = normalize_noise_placement(noise_placement)

    lam = np.asarray(lam_xyz, dtype=np.float64)
    if lam.shape not in ((n_qubits, 3), (2, n_qubits, 3)):
        raise ValueError(
            "evolve_observable_backward_series currently requires time-independent "
            "noise with shape (n_qubits,3) or (2,n_qubits,3)."
        )

    pauli = parse_pauli_string(final_pauli, n_qubits)
    parity = pauli_parity_vector(pauli)
    n_blocks = n_qubits // 2
    if enforce_global_parity and not has_singlet_compatible_global_parity(pauli):
        values = np.zeros(max_steps + 1, dtype=np.float64)
        if return_final_mps:
            info = {
                "max_bond_by_step": np.ones(max_steps + 1, dtype=np.int64),
                "active_blocks_by_step": np.zeros(max_steps + 1, dtype=np.int64),
                "bond_dims": np.ones(n_blocks + 1, dtype=np.int64),
                "noise_model": noise_model,
                "noise_placement": noise_placement_norm,
                "svd_method": svd_method,
                "parallel_kernels": parallel_kernels,
                "use_lightcone": use_lightcone,
                "global_pauli_parity": np.array(parity, dtype=np.int8),
                "parity_filtered": True,
            }
            return values, _zero_block_mps(n_blocks), info
        return values

    lam_schedule = normalize_lambda_schedule(lam, 1, n_qubits)
    eta_even = pauli_diag_factors_from_lambda(lam_schedule[0, 0], noise_model)
    eta_odd = pauli_diag_factors_from_lambda(lam_schedule[0, 1], noise_model)
    if noise_placement_norm == "layer":
        even_diag = build_block_noise_diag(eta_even)
        odd_diag = build_block_noise_diag(eta_odd)
    else:
        even_diag = build_touched_layer_noise_diag(eta_even, "even")
        odd_diag = build_touched_layer_noise_diag(eta_odd, "odd")
    trans_codes, coeffs, n_branches = build_heisenberg_pauli_transfer(phi)
    odd_pair_tables = build_odd_pair_transfer_tables(trans_codes, coeffs, n_branches)

    values = np.empty(max_steps + 1, dtype=np.float64)
    max_bond_by_step = np.zeros(max_steps + 1, dtype=np.int64)
    active_counts = np.zeros(max_steps + 1, dtype=np.int64)
    interval_history: list[list[tuple[int, int]]] = []
    discarded_by_step = np.zeros(max_steps, dtype=np.float64)

    if use_lightcone:
        islands = _initial_islands_from_pauli(pauli)
        if not islands:
            values[:] = 1.0
            final_mps = _identity_block_mps(n_blocks)
            max_bond_by_step[:] = 1
            interval_history = [[] for _ in range(max_steps + 1)]
        else:
            values[0] = _contract_islands_with_singlet_product(islands)
            max_bond_by_step[0] = max(max(island.mps.bond_dims) for island in islands)
            active_counts[0] = sum(island.right - island.left + 1 for island in islands)
            interval_history.append([(island.left, island.right) for island in islands])

            for step in range(1, max_steps + 1):
                islands = [_expand_island(island, n_blocks) for island in islands]
                islands = _merge_touching_islands(islands)

                discarded = 0.0
                for island in islands:
                    sl = slice(island.left, island.right + 1)
                    if noise_placement_norm == "layer":
                        apply_block_noise_inplace(
                            island.mps, odd_diag[sl], parallel_kernels=parallel_kernels
                        )
                        discarded += apply_odd_layer_inplace(
                            island.mps,
                            trans_codes,
                            coeffs,
                            n_branches,
                            chi_max=chi_max,
                            cutoff=cutoff,
                            svd_method=svd_method,
                            oversample=oversample,
                            n_iter=n_iter,
                            canonicalize=canonicalize,
                            parallel_kernels=parallel_kernels,
                            odd_pair_tables=odd_pair_tables,
                        )
                        apply_block_noise_inplace(
                            island.mps, even_diag[sl], parallel_kernels=parallel_kernels
                        )
                        apply_even_layer_inplace(
                            island.mps,
                            trans_codes,
                            coeffs,
                            n_branches,
                            parallel_kernels=parallel_kernels,
                        )
                    else:
                        discarded += apply_odd_layer_inplace(
                            island.mps,
                            trans_codes,
                            coeffs,
                            n_branches,
                            chi_max=chi_max,
                            cutoff=cutoff,
                            svd_method=svd_method,
                            oversample=oversample,
                            n_iter=n_iter,
                            canonicalize=canonicalize,
                            parallel_kernels=parallel_kernels,
                            odd_pair_tables=odd_pair_tables,
                        )
                        apply_block_noise_inplace(
                            island.mps, odd_diag[sl], parallel_kernels=parallel_kernels
                        )
                        apply_even_layer_inplace(
                            island.mps,
                            trans_codes,
                            coeffs,
                            n_branches,
                            parallel_kernels=parallel_kernels,
                        )
                        apply_block_noise_inplace(
                            island.mps, even_diag[sl], parallel_kernels=parallel_kernels
                        )

                islands = _merge_touching_islands(islands)
                values[step] = _contract_islands_with_singlet_product(islands)
                discarded_by_step[step - 1] = discarded
                max_bond_by_step[step] = max(max(island.mps.bond_dims) for island in islands)
                active_counts[step] = sum(
                    island.right - island.left + 1 for island in islands
                )
                interval_history.append([(island.left, island.right) for island in islands])

            final_mps = _full_mps_from_islands(islands, n_blocks)
    else:
        mps = product_pauli_mps(pauli)
        values[0] = contract_with_singlet_product(mps)
        max_bond_by_step[0] = max(mps.bond_dims)

        for step in range(1, max_steps + 1):
            if noise_placement_norm == "layer":
                apply_block_noise_inplace(mps, odd_diag, parallel_kernels=parallel_kernels)
                discarded_by_step[step - 1] = apply_odd_layer_inplace(
                    mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    chi_max=chi_max,
                    cutoff=cutoff,
                    svd_method=svd_method,
                    oversample=oversample,
                    n_iter=n_iter,
                    canonicalize=canonicalize,
                    parallel_kernels=parallel_kernels,
                    odd_pair_tables=odd_pair_tables,
                )
                apply_block_noise_inplace(mps, even_diag, parallel_kernels=parallel_kernels)
                apply_even_layer_inplace(
                    mps, trans_codes, coeffs, n_branches, parallel_kernels=parallel_kernels
                )
            else:
                discarded_by_step[step - 1] = apply_odd_layer_inplace(
                    mps,
                    trans_codes,
                    coeffs,
                    n_branches,
                    chi_max=chi_max,
                    cutoff=cutoff,
                    svd_method=svd_method,
                    oversample=oversample,
                    n_iter=n_iter,
                    canonicalize=canonicalize,
                    parallel_kernels=parallel_kernels,
                    odd_pair_tables=odd_pair_tables,
                )
                apply_block_noise_inplace(mps, odd_diag, parallel_kernels=parallel_kernels)
                apply_even_layer_inplace(
                    mps, trans_codes, coeffs, n_branches, parallel_kernels=parallel_kernels
                )
                apply_block_noise_inplace(mps, even_diag, parallel_kernels=parallel_kernels)
            values[step] = contract_with_singlet_product(mps)
            max_bond_by_step[step] = max(mps.bond_dims)

        final_mps = mps

    if return_final_mps:
        info = {
            "discarded_by_step": discarded_by_step,
            "max_bond_by_step": max_bond_by_step,
            "active_blocks_by_step": active_counts,
            "active_intervals_by_step": interval_history,
            "bond_dims": np.array(final_mps.bond_dims, dtype=np.int64),
            "noise_model": noise_model,
            "noise_placement": noise_placement_norm,
            "svd_method": svd_method,
            "parallel_kernels": parallel_kernels,
            "use_lightcone": use_lightcone,
            "global_pauli_parity": np.array(parity, dtype=np.int8),
            "parity_filtered": False,
        }
        return values, final_mps, info
    return values


def mps_to_dense(mps: PauliMPS) -> np.ndarray:
    """Reconstruct the dense blocked coefficient tensor. Use only for tests."""
    T = mps.tensors[0][0, :, :]
    for A in mps.tensors[1:]:
        T = np.tensordot(T, A, axes=([-1], [0]))
    return np.squeeze(T, axis=-1) * np.exp(mps.log_scale)


def uncompressed_mps_reference(
    final_pauli: str | Sequence[int] | np.ndarray,
    *,
    n_qubits: int,
    phi: float,
    lam_xyz: np.ndarray,
    n_steps: int,
    noise_model: NoiseModel = "legacy_sum",
    noise_placement: NoisePlacement = "gate",
) -> float:
    """
    Uncompressed MPS reference for small beta tests.

    This keeps all singular vectors, including numerically zero ones, so it is
    intentionally slower than the default solver.
    """
    value, mps, _ = evolve_observable_backward_mps(
        final_pauli,
        n_qubits=n_qubits,
        phi=phi,
        lam_xyz=lam_xyz,
        n_steps=n_steps,
        chi_max=None,
        cutoff=0.0,
        svd_method="full",
        noise_model=noise_model,
        noise_placement=noise_placement,
        return_mps=True,
    )
    return value


# Backward-compatible alias for scratch notebooks that want a small reference.
dense_backward_reference = uncompressed_mps_reference
