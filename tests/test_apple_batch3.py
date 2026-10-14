"""Experimental split plans and ragged shader against independent references."""
import os
import numpy as np
import pytest
from quimb.tensor import Tensor
from pauli_peps_symmetry import symmray, CHARGES
from peps_split_layout import split_tensor


@pytest.mark.parametrize('absorb', ['left', 'right', None])
@pytest.mark.parametrize('cap', [None, 5])
def test_split_matches_quimb_for_fresh_and_reused_layouts(absorb, cap):
    rng=np.random.default_rng(61)
    for _ in range(2):
        dense=rng.normal(size=(8,4,4,8))
        data=symmray().Z2Z2Array.from_dense(dense,[CHARGES*2,CHARGES,CHARGES,CHARGES*2],
                  [False,False,True,True],charge=(0,0),invalid_sectors='ignore')
        t=Tensor(data,inds=('a','b','c','d'))
        kw=dict(left_inds=['b','a'],right_inds=['d','c'],bond_ind='bond',
                absorb=absorb,cutoff=1e-8,max_bond=cap,get='tensors')
        expected=t.split(**kw);actual=split_tensor(t,**kw)
        for x,y in zip(actual,expected):
            assert x.inds==y.inds
            np.testing.assert_array_equal(x.data.to_dense(),y.data.to_dense())
        np.testing.assert_array_equal(t.data.to_dense(),data.to_dense())


@pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires Metal')
def test_ragged_shader_shapes_reductions_and_reuse():
    from apple_metal_ragged import RaggedProducts
    from peps_metal_layer import contract_block_products
    rng=np.random.default_rng(15);engine=RaggedProducts()
    for _ in range(3):
        lb=[(None,rng.normal(size=(6,9))[:,::2]),(None,rng.normal(size=(6,5))),
            (None,rng.normal(size=(3,7)))]
        rb=[(None,rng.normal(size=(5,4))),(None,rng.normal(size=(5,4))),
            (None,rng.normal(size=(7,2)))]
        plan=((0,0,'a',(6,4),(6,5,4)),(1,1,'a',(6,4),(6,5,4)),
              (2,2,'b',(3,2),(3,7,2)))
        expected=contract_block_products(lb,rb,plan,(0,1),(0,1))
        actual=engine(lb,rb,plan,(0,1),(0,1))
        for key in expected:np.testing.assert_allclose(actual[key],expected[key],atol=3e-6,rtol=2e-5)
    assert engine([],[],(),(),())=={}


@pytest.mark.parametrize('shape', [(64,64,64),(80,64,96),(63,64,80)])
@pytest.mark.parametrize('dtype', [np.float64,np.float32,np.complex128])
def test_bond_product_dispatch_accuracy_and_ownership(monkeypatch,shape,dtype):
    import mps_contractions as module
    monkeypatch.setattr(module,'_ACCELERATE',True)
    m,k,n=shape;rng=np.random.default_rng(56)
    a=rng.normal(size=(m,k*2)).astype(dtype)[:,::2]
    b=np.asfortranarray(rng.normal(size=(k,n)).astype(dtype))
    if dtype==np.complex128:a+=1j*rng.normal(size=a.shape)
    saved_a=a.copy();saved_b=b.copy();calls=[];original=module.dgemm
    def record(*args,**kw):calls.append(1);return original(*args,**kw)
    monkeypatch.setattr(module,'dgemm',record)
    result=module.bond_product(a,b)
    np.testing.assert_allclose(result,a@b,rtol=3e-13,atol=3e-13)
    assert bool(calls)==(dtype==np.float64 and min(shape)>=64)
    np.testing.assert_array_equal(a,saved_a);np.testing.assert_array_equal(b,saved_b)
    assert result.dtype==dtype
    monkeypatch.setattr(module,'_ACCELERATE',False)
    np.testing.assert_array_equal(module.bond_product(a,b),a@b)
