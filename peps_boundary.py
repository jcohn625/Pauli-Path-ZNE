"""Scoped float64/complex128 BLAS contractions for PEPS boundary sweeps.

The context is consulted by the stable dispatch function in pauli_peps so that
cotengra's cached expressions cannot retain the wrong numerical backend.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import numpy as np
from scipy.linalg.blas import get_blas_funcs

_BACKEND = ContextVar('peps_boundary_backend', default='einsum')


@contextmanager
def boundary_backend(name):
    if name not in ('einsum', 'blas'):
        raise ValueError('boundary backend must be einsum or blas')
    token = _BACKEND.set(name)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def use_blas():
    return _BACKEND.get() == 'blas'


def blas_tensordot(a, b, axes=2):
    """Tensor contraction via SciPy GEMM, without conjugating either operand."""
    a, b = np.asarray(a), np.asarray(b)
    dtype = np.result_type(a, b)
    if dtype not in (np.dtype('float32'), np.dtype('float64'),
                     np.dtype('complex64'), np.dtype('complex128')):
        # Preserve integer and extended precision semantics.
        return np.tensordot(a, b, axes=axes)
    if isinstance(axes, (int, np.integer)):
        if axes < 0 or axes > min(a.ndim, b.ndim):
            raise ValueError('invalid number of contracted axes')
        left = list(range(a.ndim - axes, a.ndim))
        right = list(range(axes))
    else:
        left, right = axes
        left, right = list(np.atleast_1d(left)), list(np.atleast_1d(right))
        if any(i < -a.ndim or i >= a.ndim for i in left) or any(i < -b.ndim or i >= b.ndim for i in right):
            raise ValueError('contracted axis out of bounds')
        left = [int(i % a.ndim) for i in left]
        right = [int(i % b.ndim) for i in right]
    if len(left) != len(right) or len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError('invalid contracted axes')
    if any(a.shape[i] != b.shape[j] for i, j in zip(left, right)):
        raise ValueError('contracted dimensions do not match')
    ao = [i for i in range(a.ndim) if i not in left]
    bo = [i for i in range(b.ndim) if i not in right]
    shape = tuple(a.shape[i] for i in ao) + tuple(b.shape[i] for i in bo)
    m = math.prod(a.shape[i] for i in ao)
    k = math.prod(a.shape[i] for i in left)
    n = math.prod(b.shape[i] for i in bo)
    if not m or not n or not k:
        return np.zeros(shape, dtype=dtype)
    am = a.transpose(ao + left).reshape(m, k)
    bm = b.transpose(right + bo).reshape(k, n)
    gemm = get_blas_funcs('gemm', (am, bm))
    result = gemm(1., am, bm)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('nonfinite BLAS boundary contraction')
    return result.reshape(shape)
