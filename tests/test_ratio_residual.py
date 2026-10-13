import sys
import unittest
import numpy as np

import pauli_mps_solver as solver
from mps_residual import run,residual_value
from parity_pauli_mps import run as baseline

class ResidualTests(unittest.TestCase):
    def test_exact_adjoint_telescope_and_unchanged_baseline(self):
        eta=np.tile([.9978,.9978,.996],(6,1))
        for s in [0,1]:
            row=run(solver,6,4,2,4,eta**s,4,16)
            original=baseline(solver,6,4,2,4,eta**s,4)[0]
            exact=baseline(solver,6,4,2,4,eta**s,64)[0]
            self.assertAlmostEqual(row['base'],original,places=12)
            self.assertGreater(abs(original-exact),1e-5)
            self.assertAlmostEqual(row['corrected'],exact,places=11)
    def test_no_truncation(self):
        row=run(solver,6,3,2,4,np.ones((6,3)),64,16)
        self.assertLess(abs(row['delta']),1e-12)
    def test_four_block_exact_adjoint(self):
        eta=np.tile([.9978,.9978,.996],(8,1))
        row=run(solver,8,4,3,5,eta,8,256)
        exact=baseline(solver,8,4,3,5,eta,256)[0]
        self.assertGreater(abs(row['base']-exact),1e-5)
        self.assertAlmostEqual(row['corrected'],exact,places=10)
    def test_residual_scale_and_direct_dense_contraction(self):
        rng=np.random.default_rng(91)
        state=solver.PauliMPS([rng.normal(size=(1,16,2)),rng.normal(size=(2,16,3)),rng.normal(size=(3,16,1))])
        guide=solver.PauliMPS([rng.normal(size=(1,16,4)),rng.normal(size=(4,16,2)),rng.normal(size=(2,16,1))])
        m=rng.normal(size=(32,16));l=rng.normal(size=(32,3));r=rng.normal(size=(3,16))
        dense=np.einsum('apg,gqh,hrb->pqr',*guide.tensors,optimize=True)
        grad=np.einsum('pl,pqr->lqr',state.tensors[0][0],dense,optimize=True).reshape(m.shape)
        expected=np.sum(grad*(m-l@r))
        self.assertAlmostEqual(residual_value(state,guide,1,m,l,r),expected,places=9)
        state.tensors[0]/=7;state.log_scale+=np.log(7)
        guide.tensors[1]/=11;guide.log_scale+=np.log(11)
        self.assertAlmostEqual(residual_value(state,guide,1,m,l,r),expected,places=9)

if __name__=='__main__':unittest.main()
