"""Strict indexing/order contracts and complete residual workflow comparisons."""
import numpy as np
import pytest
import cpu_numba_kernels as kernels


@pytest.mark.parametrize('shape,axes', [((3,4),(1,0)),((2,3,4,2),(3,1,0,2)),((2,3,2,4,3),(4,2,0,3,1))])
@pytest.mark.parametrize('strided',[False,True])
def test_scale_pack_exact_and_input_immutable(shape,axes,strided):
    rng=np.random.default_rng(419)
    block=rng.normal(size=shape)
    if strided:block=block[...,::-1]
    before=block.copy();expected=before.copy();weights=[None]*5
    for axis in range(0,len(shape),2):
        weights[axis]=rng.normal(size=shape[axis])
        view=[1]*len(shape);view[axis]=-1
        expected*=weights[axis].reshape(view)
    result=kernels.scale_pack(block,tuple(weights),axes)
    np.testing.assert_array_equal(result,expected.transpose(axes))
    np.testing.assert_array_equal(block,before)
    assert result.flags.c_contiguous and not np.shares_memory(result,block)


def test_merge_ties_zeros_cutoff_and_empty_sectors():
    spectra=[np.array([5.,2.,2.,1e-14,0.]),np.array([]),np.array([5.,2.,1.,0.]),np.array([2.,2.,0.])]
    values=np.concatenate(spectra);offsets=np.array([0,*np.cumsum([len(s) for s in spectra])])
    ref=sorted([(float(s),p,j) for p,a in enumerate(spectra) for j,s in enumerate(a) if s>0],key=lambda x:x[0],reverse=True)
    for cap in (1,3,6,20):
        got,positive=kernels.select_sorted(values,offsets,cap)
        expected=np.array([(p,j) for v,p,j in ref[:cap] if v>1e-13*ref[0][0]])
        np.testing.assert_array_equal(got,expected)
        assert positive==len(ref)
    got,positive=kernels.select_sorted(np.zeros(3),np.array([0,1,3]),5)
    assert got.shape==(0,2) and positive==0


def test_strided_sector_gather_and_transfer():
    rng=np.random.default_rng(10)
    a=rng.normal(size=(20,24)).T[::2,::-1]
    ri=np.array([1,0,5]);ci=np.array([2,4,1,7])
    np.testing.assert_array_equal(kernels.gather(a,ri,ci),a[ri[:,None],ci])
    t=rng.normal(size=(3,16,5))[:,:,::-1];before=t.copy()
    cols=rng.permutation(16);weights=rng.normal(size=16)
    np.testing.assert_array_equal(kernels.transfer(t,cols,weights),t[:,cols,:]*weights[None,:,None])
    np.testing.assert_array_equal(t,before)


@pytest.mark.parametrize('feature',['gather','select','scatter','transfer','all'])
def test_mps_ratio_q_residual_workflow(monkeypatch,feature):
    from tensor_zne import curves,evaluate_mps
    kw=dict(n=6,steps=2,sites=(1,4),cap=4,guide_cap=8,noise_placement='after_cnot',pair_rates=.001,
            noise_edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(0,4)])
    monkeypatch.setattr(kernels,'ENABLED',dict.fromkeys(kernels.ENABLED,False))
    reference=curves(evaluate_mps,[0.,1.],**kw)
    for name in kernels.ENABLED:kernels.ENABLED[name]=feature=='all' or name==feature
    actual=curves(evaluate_mps,[0.,1.],**kw)
    for ref,got in zip(reference['rows'],actual['rows']):
        for key in ('observable_corrected','unsigned_corrected','ratio_corrected','q_plus_corrected','q_minus_corrected'):
            assert got[key]==pytest.approx(ref[key],abs=2e-12,rel=2e-12)
    for ref,got in zip(reference['raw'],actual['raw']):
        for kind in ('signed','unsigned'):
            assert got[kind]['details']['charge_leakage']==0
            assert got[kind]['details']['contribution_count']==ref[kind]['details']['contribution_count']
            np.testing.assert_allclose([c['value'] for c in got[kind]['details']['contributions']],
                                       [c['value'] for c in ref[kind]['details']['contributions']],atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize('residual',[False,True])
def test_peps_cpu_scaling_workflow(monkeypatch,residual):
    from tensor_zne import curves,evaluate_peps
    kw=dict(nx=2,ny=3,steps=2,sites=(1,4),cap=4,boundary_cap=4,environment_cap=8,
            noise_placement='after_cnot',pair_rates=.001,gate_backend='batched-cpu',residual=residual)
    monkeypatch.setitem(kernels.ENABLED,'scale',False)
    reference=curves(evaluate_peps,[0.,1.],**kw)
    monkeypatch.setitem(kernels.ENABLED,'scale',True)
    actual=curves(evaluate_peps,[0.,1.],**kw)
    for ref,got in zip(reference['rows'],actual['rows']):
        for key in ('observable_corrected','unsigned_corrected','ratio_corrected','q_plus_corrected','q_minus_corrected'):
            assert got[key]==pytest.approx(ref[key],abs=3e-11,rel=3e-11)


def test_scatter_respects_global_order_and_leaves_forbidden_entries_zero():
    rng=np.random.default_rng(58)
    selected=np.array([[1,0],[0,1],[1,2],[0,0]])
    left=np.zeros((9,4));right=np.zeros((4,11));charges=np.empty(4,dtype=np.int64)
    expected_l=left.copy();expected_r=right.copy();expected_q=charges.copy()
    for p,charge,ri,ci in [(0,2,np.array([0,4,8]),np.array([1,3,5,7])),
                            (1,3,np.array([1,3,5,7]),np.array([0,2,4,6,8]))]:
        u=np.asfortranarray(rng.normal(size=(len(ri),3)))
        s=np.array([3.,2.,1.]);v=rng.normal(size=(3,len(ci)))[:,::-1]
        for k,(piece,j) in enumerate(selected):
            if piece==p:
                expected_l[ri,k]=u[:,j];expected_r[k,ci]=s[j]*v[j];expected_q[k]=charge
        kernels.scatter_svd(left,right,charges,selected,p,charge,ri,ci,u,s,v)
    np.testing.assert_array_equal(left,expected_l)
    np.testing.assert_array_equal(right,expected_r)
    np.testing.assert_array_equal(charges,expected_q)


@pytest.mark.parametrize('shape,dtype', [((17,17,17),np.float64),((2,3,4),np.float32),((2,3,4),np.complex128)])
def test_peps_large_or_other_dtype_uses_reference_scaling(monkeypatch,shape,dtype):
    from quimb.tensor import Tensor
    from pauli_peps_symmetry import symmray
    from peps_metal_layer import LayerEngine
    sr=symmray();neutral=(0,0)
    block=np.arange(np.prod(shape),dtype=dtype).reshape(shape)
    data=sr.Z2Z2Array.from_blocks({(neutral,)*3:block},duals=(False,False,True),charge=neutral)
    tensor=Tensor(data,inds=('a','b','c'))
    weights=np.linspace(.5,1.5,shape[0]).astype(dtype)
    scales={'a':sr.BlockVector({neutral:weights})}
    monkeypatch.setitem(kernels.ENABLED,'scale',True)
    def forbidden(*args):raise AssertionError('unsupported block must use reference scaling')
    monkeypatch.setattr(kernels,'_scale_pack',forbidden)
    out,=LayerEngine('cpu').scale_many([(tensor,scales,('c','a','b'))])
    expected=(block*weights[:,None,None]).transpose(2,0,1)
    np.testing.assert_array_equal(out.data.to_dense(),expected)
