import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Literal, Optional
def build_basis_u64(N: int, K: int) -> np.ndarray:
    masks = np.empty(_nCk(N, K), dtype=np.uint64)
    t = 0
    for occ in combinations(range(N), K):
        x = np.uint64(0)
        for p in occ:
            x |= (np.uint64(1) << np.uint64(p))
        masks[t] = x
        t += 1
    masks.sort()
    return masks

def _nCk(n,k):
    # exact integer comb without scipy
    if k < 0 or k > n: return 0
    k = min(k, n-k)
    num = 1
    den = 1
    for i in range(1, k+1):
        num *= n - (k - i)
        den *= i
    return num // den

def build_basis_index(basis: List[int]) -> Dict[int, int]:
    """Map determinant bitstring -> index in CI vector."""
    return {d: k for k, d in enumerate(basis)}


import numba as nb

@nb.njit
def _pop_bit(x, i):
    return (x >> i) & 1

@nb.njit
def build_swap_cache_for_pair(basis: np.ndarray, i: int, j: int):
    """
    Returns:
      A, B : int32 arrays of same length P, with A[p] < B[p] and swap connects them
      fixed: int32 array of indices where swap acts as identity
    """
    M = basis.shape[0]

    # First pass: count how many pairs and fixed points
    npairs = 0
    nfixed = 0
    for idx in range(M):
        x = basis[idx]
        bi = _pop_bit(x, i)
        bj = _pop_bit(x, j)
        if bi == bj:
            nfixed += 1
        else:
            # it will be paired with its partner; count only once
            y = x ^ (np.uint64(1) << np.uint64(i)) ^ (np.uint64(1) << np.uint64(j))
            partner = np.searchsorted(basis, y)
            if idx < partner:
                npairs += 1

    A = np.empty(npairs, dtype=np.int32)
    B = np.empty(npairs, dtype=np.int32)
    fixed = np.empty(nfixed, dtype=np.int32)

    # Second pass: fill
    p = 0
    f = 0
    for idx in range(M):
        x = basis[idx]
        bi = _pop_bit(x, i)
        bj = _pop_bit(x, j)
        if bi == bj:
            fixed[f] = idx
            f += 1
        else:
            y = x ^ (np.uint64(1) << np.uint64(i)) ^ (np.uint64(1) << np.uint64(j))
            partner = np.searchsorted(basis, y)
            if idx < partner:
                A[p] = idx
                B[p] = partner
                p += 1

    return A, B, fixed


import numpy as np
import numba as nb

@nb.njit
def init_singlet_product_state(basis: np.ndarray, N: int):
    """
    Initialize psi for (|01>-|10>)^{⊗ (N/2)} on pairs (0,1),(2,3),...
    Assumes N even and basis corresponds to K = N/2 sector.
    basis must be sorted uint64.
    Returns complex128 psi.
    """
    M = basis.shape[0]
    psi = np.zeros(M, dtype=np.complex128)

    npairs = N // 2
    # normalization: (1/sqrt(2))^{npairs}
    amp0 = 1.0
    for _ in range(npairs):
        amp0 *= 0.7071067811865475  # 1/sqrt(2)

    nterms = 1 << npairs  # 2^(N/2)

    for choice in range(nterms):
        x = np.uint64(0)
        sign = 1.0
        for p in range(npairs):
            if (choice >> p) & 1:
                # choose |01> on pair p: set site 2p+1 and add a minus sign
                x |= (np.uint64(1) << np.uint64(2*p + 1))
                sign = -sign
            else:
                # choose |10> on pair p: set site 2p
                x |= (np.uint64(1) << np.uint64(2*p))
        idx = np.searchsorted(basis, x)
        # optional safety: if basis[idx] != x then something is wrong
        psi[idx] = sign * amp0

    return psi


@nb.njit
def apply_swap_gate_inplace(psi: np.ndarray, A: np.ndarray, B: np.ndarray, fixed: np.ndarray, c: float, s: float):
    """
    psi is complex128 vector, updated in-place.
    """
    # fixed points: multiply by e^{-i theta} = c - i s
    phase = complex(c, -s)  # c - i s
    for t in range(fixed.shape[0]):
        idx = fixed[t]
        psi[idx] *= phase

    # changing pairs: 2x2 mixing
    # (a,b) -> ( c*pa - i*s*pb,  c*pb - i*s*pa )
    is_ = complex(0.0, -s)  # -i*s
    for p in range(A.shape[0]):
        a = A[p]
        b = B[p]
        pa = psi[a]
        pb = psi[b]
        psi[a] = c * pa + is_ * pb
        psi[b] = c * pb + is_ * pa


class SwapCache:
    def __init__(self, basis: np.ndarray):
        self.basis = basis
        self._cache = {}  # (i,j) -> (A,B,fixed)

    def get(self, i: int, j: int):
        if i > j:
            i, j = j, i
        key = (i, j)
        val = self._cache.get(key)
        if val is None:
            A, B, fixed = build_swap_cache_for_pair(self.basis, i, j)
            self._cache[key] = (A, B, fixed)
        return self._cache[key]


def apply_gate(cache: SwapCache, psi: np.ndarray, i: int, j: int, theta: float):
    A, B, fixed = cache.get(i, j)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    apply_swap_gate_inplace(psi, A, B, fixed, c, s)