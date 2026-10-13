"""Opt-in CPU/Metal randomized SVD; existing reference solver is unchanged.

LU-normalized power iteration retains QR for the final orthonormal basis.
Caches contain only deterministic random sketches, never state-dependent factors.
"""
from collections import OrderedDict
import time
import numpy as np
import scipy.linalg as la


class RandomizedSVD:
    def __init__(self, normalizer='qr', driver='gesdd', cache_bytes=64*1024**2,
                 backend='cpu', audit=False):
        if normalizer not in ('qr','lu') or driver not in ('gesvd','gesdd') or backend not in ('cpu','metal'):
            raise ValueError('Invalid randomized SVD configuration')
        self.normalizer, self.driver = normalizer, driver
        self.cache_bytes, self.backend, self.audit = cache_bytes, backend, audit
        self.cache = OrderedDict()
        self.stats = dict(calls=0, cache_hits=0, cache_bytes=0, seconds=0.)
        if backend == 'metal':
            import mlx.core as mx
            if not mx.metal.is_available():
                raise RuntimeError('Metal device unavailable')
            self.mx = mx

    def sketch(self, n, rank, dtype):
        key = (n, rank, np.dtype(dtype).str)
        if key in self.cache:
            self.stats['cache_hits'] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        rng = np.random.default_rng(0)
        omega = rng.standard_normal((n, rank))
        if np.issubdtype(dtype, np.complexfloating):
            omega = (omega + 1j*rng.standard_normal((n, rank)))/np.sqrt(2)
        omega = omega.astype(dtype)
        if omega.nbytes <= self.cache_bytes:
            while self.stats['cache_bytes'] + omega.nbytes > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.stats['cache_bytes'] -= old.nbytes
            omega.flags.writeable = False
            self.cache[key] = omega
            self.stats['cache_bytes'] += omega.nbytes
        return omega

    def normalize(self, a):
        if self.normalizer == 'lu':
            return la.lu(a, permute_l=True, check_finite=True)[0]
        return la.qr(a, mode='economic', check_finite=True)[0]

    def __call__(self, mat, rank, n_iter=3):
        start = time.perf_counter()
        if mat.ndim!=2 or min(mat.shape)==0 or rank<1:
            raise ValueError('SVD needs a nonempty matrix and positive rank')
        rank = min(int(rank), *mat.shape)
        if not np.isfinite(mat).all():
            raise ValueError('Nonfinite SVD input')
        omega = self.sketch(mat.shape[1], rank, mat.dtype)
        if self.backend == 'metal':
            # Keep the large matrix resident; only skinny iterates cross the API.
            mx = self.mx
            dtype = mx.complex64 if np.iscomplexobj(mat) else mx.float32
            device_mat = mx.array(mat, dtype=dtype)
            def product(x, adjoint=False):
                a = mx.conj(device_mat.T) if adjoint else device_mat
                out = a @ mx.array(x, dtype=dtype)
                mx.eval(out)
                return np.array(out).astype(mat.dtype)
            def project(q):
                out = mx.conj(mx.array(q, dtype=dtype).T) @ device_mat
                mx.eval(out)
                return np.array(out).astype(mat.dtype)
        else:
            def product(x, adjoint=False):
                return (mat.conj().T if adjoint else mat) @ x
            def project(q):
                return q.conj().T @ mat
        y = product(omega)
        for _ in range(max(0, n_iter)):
            y = self.normalize(y)
            z = self.normalize(product(y, adjoint=True))
            y = product(z)
        q = la.qr(y, mode='economic', check_finite=True)[0]
        small = project(q)
        u, s, vh = la.svd(small, full_matrices=False, lapack_driver=self.driver,
                           check_finite=True)
        u = q @ u
        if not np.isfinite(s).all():
            raise ValueError('Nonfinite singular values')
        if self.audit:
            np.testing.assert_allclose(u.conj().T@u, np.eye(rank), atol=2e-9, rtol=0)
            np.testing.assert_allclose(vh@vh.conj().T, np.eye(rank), atol=2e-9, rtol=0)
        self.stats['calls'] += 1
        self.stats['seconds'] += time.perf_counter()-start
        return u, s, vh
