"""GPU integration tests are opt-in; CPU tests check API guardrails."""
import os
import numpy as np
import pytest
from lattice_circuit import square
from pauli_peps import evaluate_peps


@pytest.mark.parametrize('symmetry,sweeps',[('none',12),('z2z2',0)])
def test_gpu_boundary_rejects_invalid_configuration(symmetry,sweeps):
    with pytest.raises(ValueError,match='Metal Jacobi requires'):
        evaluate_peps('ZZII',square(2,2),nx=2,ny=2,pairs=[(0,1),(2,3)],
            n_steps=1,phi=.19,symmetry=symmetry,boundary_backend='metal-jacobi',jacobi_sweeps=sweeps)


@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires explicit Metal access')
def test_gpu_global_charge_truncation_and_restore():
    import mlx.core as mx
    import quimb.tensor as qtn
    from peps_gpu_boundary import gpu_boundary
    from pauli_peps_symmetry import symmray
    sr=symmray()
    from symmray.sparse.sparse_array_common import SparseArrayCommon
    import symmray.sparse.sparse_array_common as sparse_module
    original=SparseArrayCommon._split_abelian
    original_truncate=sparse_module.truncate_svd_result_blocksparse
    array=sr.Z2Z2Array.from_blocks({
        ((0,0),(0,0)):mx.diag(mx.array([10.,1.])),
        ((1,1),(1,1)):mx.diag(mx.array([9.,.5])),
        ((0,1),(0,1)):mx.diag(mx.array([8.,.25])),
    },duals=(False,True),charge=(0,0))
    with gpu_boundary(12):
        u,s,v=qtn.Tensor(array,inds=['a','b']).split(left_inds=['a'],get='tensors',
            max_bond=2,cutoff=0.,absorb=None)
        actual=sorted(float(x) for block in s.data.blocks.values() for x in np.asarray(block))
        np.testing.assert_allclose(actual,[9.,10.],rtol=2e-6)
        assert u.shape[-1]==v.shape[0]==2
    assert SparseArrayCommon._split_abelian is original
    assert sparse_module.truncate_svd_result_blocksparse is original_truncate
    with pytest.raises(RuntimeError):
        with gpu_boundary(12):raise RuntimeError('interrupted')
    assert SparseArrayCommon._split_abelian is original
    assert sparse_module.truncate_svd_result_blocksparse is original_truncate


@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires explicit Metal access')
@pytest.mark.parametrize('mode,cutoff,cap',[
    ('abs',.5,3),('rel',.1,3),('sum2',.3,3),('rsum2',.1,3),
    ('sum1',.5,3),('rsum1',.1,3),('rsum2',0.,2),('rsum2',2.,2)])
def test_combined_rank_readback_matches_reference_including_ties(mode,cutoff,cap):
    import mlx.core as mx
    import quimb.tensor as qtn
    from peps_gpu_boundary import gpu_boundary
    from pauli_peps_symmetry import symmray
    sr=symmray()
    outputs=[]
    for optimized in (False,True):
        a=sr.Z2Z2Array.from_blocks({
            ((0,0),(0,0)):mx.diag(mx.array([4.,2.,.1])),
            ((1,1),(1,1)):mx.diag(mx.array([2.,1.,.01])),
        },duals=(False,True),charge=(0,0))
        with gpu_boundary(16,optimized=optimized):
            u,s,v=qtn.Tensor(a,inds=['a','b']).split(left_inds=['a'],get='tensors',
                max_bond=cap,cutoff=cutoff,cutoff_mode=mode,absorb=None)
            outputs.append({key:np.asarray(block) for key,block in s.data.blocks.items()})
    assert outputs[0].keys()==outputs[1].keys()
    for key in outputs[0]:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])


@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY')!='1',reason='requires explicit Metal access')
def test_large_qr_materialization_preserves_factorization():
    import mlx.core as mx
    from peps_gpu_decomp import GPUDecompositions
    engine=GPUDecompositions()
    engine._qr_flush_bytes=0  # Exercise the large-matrix path on a small fixture.
    a=np.random.default_rng(19).normal(size=(2,192,160)).astype('float32')
    a[1,:,1]=a[1,:,0]
    q,r=engine.qr(mx.array(a))
    mx.eval(q,r)
    q,r=np.asarray(q),np.asarray(r)
    assert engine.stats['large_qr_flushes']>0
    # Avoid this environment's BLAS floating-point warning artefacts, as in
    # the independent decomposition smoke checks.
    residual=np.einsum('bik,bkj->bij',q,r,optimize=False)-a
    assert np.sqrt(np.sum(residual.astype('float64')**2)/np.sum(a.astype('float64')**2))<3e-6
    np.testing.assert_allclose(np.einsum('bki,bkj->bij',q,q,optimize=False),
                               np.broadcast_to(np.eye(160),(2,160,160)),atol=3e-6)
