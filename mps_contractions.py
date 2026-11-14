"""Fixed binary contraction orders for MPS sweeps and scalar environments.

These are bilinear products, matching the real Pauli solver's einsums; they do
not conjugate either operand. NumPy handles noncontiguous inputs on reshape.
Choose the environment order from actual dimensions, not a nominal bond cap.
"""
import platform
import sys
import numpy as np
import scipy
from scipy.linalg.blas import dgemm


def _blas_name(module):
    return getattr(module.__config__, 'CONFIG', {}).get(
        'Build Dependencies', {}).get('blas', {}).get('name', '').lower()


# Measured for Apple Silicon with NumPy/OpenBLAS and SciPy/Accelerate.
# Unknown library builds retain NumPy rather than assuming a faster backend.
_ACCELERATE = (sys.platform == 'darwin' and platform.machine() == 'arm64'
               and _blas_name(scipy) == 'accelerate'
               and 'openblas' in _blas_name(np))


def bond_product(a, b):
    """Keep tiny/other-dtype products on NumPy; use Accelerate for large FP64."""
    if (_ACCELERATE and a.dtype == np.float64 and b.dtype == np.float64
            and min(a.shape[0], a.shape[1], b.shape[1]) >= 64):
        # Transposed operands produce a C-contiguous result without a copy.
        return dgemm(1., b.T, a.T).T
    return a @ b


def absorb_left(link, tensor):
    return bond_product(link, tensor.reshape(tensor.shape[0], -1)).reshape(
        link.shape[0], tensor.shape[1], tensor.shape[2])


def absorb_right(tensor, link):
    return bond_product(tensor.reshape(-1, tensor.shape[2]), link).reshape(
        tensor.shape[0], tensor.shape[1], link.shape[1])


def join_sites(left, right):
    return bond_product(left.reshape(-1, left.shape[2]), right.reshape(right.shape[0], -1)).reshape(
        left.shape[0], left.shape[1], right.shape[1], right.shape[2])


def environment_left(env, left, right):
    """ab,apr,bps->rs without repeating einsum path discovery."""
    a, p, r = left.shape
    b, _, s = right.shape
    if a*b*s + a*r*s <= a*b*r + b*r*s:
        tmp = env @ right.reshape(b, p*s)
        return left.reshape(a*p, r).T @ tmp.reshape(a*p, s)
    tmp = env.T @ left.reshape(a, p*r)
    return tmp.reshape(b*p, r).T @ right.reshape(b*p, s)


def environment_right(env, left, right):
    """apr,bps,rs->ab with the cheaper of the two binary orders."""
    a, p, r = left.shape
    b, _, s = right.shape
    if b*r*s + a*b*r <= a*r*s + a*b*s:
        tmp = right @ env.T
        return left.reshape(a, p*r) @ tmp.reshape(b, p*r).T
    tmp = left @ env
    return tmp.reshape(a, p*s) @ right.reshape(b, p*s).T
