import numpy as np
import unittest
from optimized_pauli_svd import RandomizedSVD

def check_low_rank_and_adjoint(complex_values,normalizer):
    rng=np.random.default_rng(7)
    a=rng.normal(size=(70,12)); b=rng.normal(size=(12,55))
    if complex_values:
        a=a+1j*rng.normal(size=a.shape);b=b+1j*rng.normal(size=b.shape)
    m=a@b
    method=RandomizedSVD(normalizer,audit=True)
    u,s,v=method(m,16,2)
    assert np.linalg.norm((u*s)@v-m)/np.linalg.norm(m)<1e-12
    method(m,16,2)
    assert method.stats['cache_hits']==1

def check_full_rank_spectrum_and_bounded_cache():
    rng=np.random.default_rng(8)
    m=rng.normal(size=(51,37))
    method=RandomizedSVD('lu',cache_bytes=100)
    u,s,v=method(m,37,2)
    np.testing.assert_allclose(s,np.linalg.svd(m,compute_uv=False),rtol=1e-12)
    assert method.stats['cache_bytes']==0
    assert np.linalg.norm((u*s)@v-m)/np.linalg.norm(m)<1e-12


class RandomizedSVDTests(unittest.TestCase):
    def test_real_complex_qr_lu(self):
        for c in [False,True]:
            for method in ['qr','lu']:
                with self.subTest(complex=c,normalizer=method):
                    check_low_rank_and_adjoint(c,method)

    def test_full_rank_and_cache(self):
        check_full_rank_spectrum_and_bounded_cache()

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):RandomizedSVD('none')
        with self.assertRaises(ValueError):RandomizedSVD()(np.eye(3),0)
        with self.assertRaises(ValueError):RandomizedSVD()(np.array([[np.nan]]),1)

if __name__=='__main__':unittest.main()
