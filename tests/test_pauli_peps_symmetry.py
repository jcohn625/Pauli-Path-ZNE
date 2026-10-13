import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/pauli_zne_numba_cache')
import numpy as np
import pytest
from lattice_circuit import square, heisenberg_transfer
from causal_pauli_network import compile_network
from pauli_peps import evaluate_peps
from pauli_peps_symmetry import CHARGES, total_charge, gate, diagonal


def test_charge_means_equal_parities_not_all_even():
    assert total_charge([1, 2, 3]) == (0, 0)
    assert total_charge([3, 3]) == (0, 0)
    assert total_charge([1]) != (0, 0)
    transfer = heisenberg_transfer(.23)
    # XI -> YZ is nonzero and flips all three count parities.
    assert abs(transfer[4*2+3,4*1]) > .01
    q = np.array([total_charge([a,b]) for a in range(4) for b in range(4)])
    forbidden = np.any(q[:,None,:] != q[None,:,:], axis=2)
    assert np.max(abs(transfer[forbidden])) == 0
    sparse = gate(transfer)
    np.testing.assert_allclose(sparse.to_dense(index_maps=[CHARGES]*4), transfer.reshape((4,)*4))


@pytest.mark.parametrize('unsigned', [False, True])
@pytest.mark.parametrize('timing', ['before', 'after'])
@pytest.mark.parametrize('placement', ['gate', 'layer'])
def test_exact_four_sites_anisotropic_noise(unsigned,timing,placement):
    lat = square(2,2)
    kw = dict(pairs=[(0,1),(2,3)],n_steps=2,phi=.19,
              lam_xyz=np.random.default_rng(3).uniform(.001,.02,(2,4,4,3)),
              scale=1.7,unsigned=unsigned,noise_timing=timing,noise_placement=placement)
    exact = compile_network('ZZII', lat, **kw).contract()['value']
    result = evaluate_peps('ZZII',lat,nx=2,ny=2,bond_cap=256,boundary_cap=1024,
                           cutoff=0,symmetry='z2z2',**kw)
    assert result['value'] == pytest.approx(exact,abs=3e-11)
    assert result['symmetry_stats']['tensor_charge_checks_passed']


@pytest.mark.parametrize('pauli', ['IIIIII','XYZIII','ZZIIII','XIIIII','XYIIII'])
@pytest.mark.parametrize('unsigned', [False,True])
def test_six_sites_and_selection_rule(pauli,unsigned):
    lat = square(2,3)
    kw = dict(pairs=[(0,1),(2,3),(4,5)],n_steps=1,phi=.23,
              lam_xyz=(.002,.005,.013),unsigned=unsigned)
    exact = compile_network(pauli,lat,**kw).contract()['value']
    result = evaluate_peps(pauli,lat,nx=2,ny=3,bond_cap=64,boundary_cap=256,
                          symmetry='z2z2',**kw)
    assert result['value'] == pytest.approx(exact,abs=3e-11)
    assert result['symmetry_exact_zero'] == (total_charge(['IXYZ'.index(c) for c in pauli]) != (0,0))


def test_boundary_compression_and_total_spatial_cap():
    lat = square(3,4)
    pairs=list(zip(lat.order[::2],lat.order[1::2]))
    result=evaluate_peps('IIIIIZZIIIII',lat,nx=3,ny=4,pairs=pairs,n_steps=2,phi=.2,
                         lam_xyz=(.001,.004,.009),bond_cap=5,boundary_cap=5,symmetry='z2z2')
    assert np.isfinite(result['value'])
    assert result['symmetry_stats']['max_bond'] <= 5
    assert result['symmetry_stats']['storage_fraction'] < .4
    assert result['boundary_symmetry_stats']['tensor_charge_checks_passed']


def test_switching_dense_and_symmetric_does_not_poison_expression_cache():
    lat=square(2,2)
    kw=dict(nx=2,ny=2,pairs=[(0,1),(2,3)],n_steps=1,phi=.21,
            lam_xyz=(.001,.004,.009),bond_cap=16,boundary_cap=64)
    values=[evaluate_peps('ZZII',lat,symmetry=sym,**kw)['value']
            for sym in ('none','z2z2','none','z2z2')]
    np.testing.assert_allclose(values,values[0],atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize('nx,ny',[(2,2),(2,3),(3,4)])
def test_exactly_two_independent_additive_count_parities(nx,ny):
    # A general site-dependent binary parity has 3N coefficients, one per X/Y/Z.
    # Require equality across every nonzero local gate transition and find nullity.
    lat=square(nx,ny)
    transitions=np.argwhere(abs(heisenberg_transfer(.23)) > 1e-13)
    equations=[]
    for layer in lat.layers:
        for a,b in layer.bonds:
            for out,inp in transitions:
                row=np.zeros(3*lat.n_sites,dtype=np.uint8)
                for site,code in [(a,out//4),(b,out%4),(a,inp//4),(b,inp%4)]:
                    if code:row[3*site+code-1]^=1
                equations.append(row)
    m=np.array(equations)
    rank=0
    for col in range(m.shape[1]):
        candidates=np.flatnonzero(m[rank:,col])
        if not len(candidates):continue
        k=rank+candidates[0]
        m[[rank,k]]=m[[k,rank]]
        for j in range(rank+1,len(m)):
            if m[j,col]:m[j]^=m[rank]
        rank+=1
    assert m.shape[1]-rank == 2
