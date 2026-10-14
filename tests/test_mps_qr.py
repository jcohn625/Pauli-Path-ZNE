"""Numerical and ownership contracts for the cached LAPACK QR path."""
import numpy as np
import pytest
import scipy.linalg as la
from mps_qr import economic_qr


@pytest.mark.parametrize('shape', [(1, 1), (4, 1), (16, 4), (4, 16), (64, 16)])
@pytest.mark.parametrize('case', ['strided', 'fortran', 'rank_deficient', 'tiny', 'large'])
def test_qr_reconstruction_and_orthogonality(shape, case):
    rng = np.random.default_rng(720)
    a = rng.normal(size=(shape[0]*2, shape[1]*2))[::2, ::2]
    if case == 'fortran':
        a = np.asfortranarray(a)
    elif case == 'rank_deficient':
        a[:, -1] = 0.
    elif case == 'tiny':
        a *= 1e-150
    elif case == 'large':
        a *= 1e150
    original = a.copy()
    q, r = economic_qr(a)
    scale = np.max(np.abs(a)) or 1.
    np.testing.assert_allclose((q @ r)/scale, a/scale, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(q.T @ q, np.eye(min(shape)), atol=3e-14)
    np.testing.assert_array_equal(a, original)
    reference = la.qr(a, mode='economic')
    for actual, expected in zip((q, r), reference):
        np.testing.assert_allclose(actual, expected, atol=3e-14*max(1., np.max(abs(expected))))


@pytest.mark.parametrize('dtype', [np.float32, np.complex128])
def test_qr_fallback(dtype):
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
    if dtype == np.complex128:
        a[0, 0] += 2j
    q, r = economic_qr(a)
    tol = 20*np.finfo(dtype).eps*np.linalg.norm(a)
    np.testing.assert_allclose(q @ r, a, atol=tol, rtol=0.)
    expected = la.qr(a, mode='economic')
    for actual, reference in zip((q, r), expected):
        np.testing.assert_array_equal(actual, reference)
    assert q.dtype == dtype


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_qr_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        economic_qr(np.array([[value]]))
