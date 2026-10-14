"""Economic Householder QR with bounded workspace-query caching for real sectors."""
from functools import lru_cache
import numpy as np
import scipy.linalg as la
from scipy.linalg.lapack import dgeqrf, dorgqr


@lru_cache(maxsize=256)
def _workspace(m, n):
    # Cache only workspace lengths, never arrays or tensor values.
    a = np.empty((m, n), order='F')
    k = min(m, n)
    _, _, work, info = dgeqrf(a, lwork=-1)
    if info != 0:
        raise ValueError(f'dgeqrf workspace query failed: {info}')
    _, qwork, info = dorgqr(a[:, :k], np.empty(k), lwork=-1)
    if info != 0:
        raise ValueError(f'dorgqr workspace query failed: {info}')
    return int(work[0]), int(qwork[0])


def economic_qr(a):
    """Same LAPACK factorization as SciPy QR; preserve input and finite checks."""
    if a.dtype != np.float64 or a.ndim != 2 or min(a.shape) == 0:
        return la.qr(a, mode='economic', check_finite=True)
    if not np.isfinite(a).all():
        raise ValueError('QR input must contain only finite values')
    m, n = a.shape
    k = min(m, n)
    work, qwork = _workspace(m, n)
    # Never overwrite the caller, including Fortran-contiguous inputs.
    packed, tau, _, info = dgeqrf(a, lwork=work, overwrite_a=False)
    if info != 0:
        raise ValueError(f'dgeqrf failed: {info}')
    r = np.triu(packed[:k, :])
    q, _, info = dorgqr(packed[:, :k], tau, lwork=qwork, overwrite_a=True)
    if info != 0:
        raise ValueError(f'dorgqr failed: {info}')
    return q, r
