"""Independent dense elementary-circuit references for transported parity."""
import itertools
import os
import numpy as np
import pytest
from cnot_zne.pauli_zne.circuit import transfers,DIRECTIONS
from cnot_parity import compile_operations,route_operations,check_operation,normalize_noise,Q
from lattice_circuit import chain,square
from tensor_zne import evaluate_mps,evaluate_peps,curves


def dense_reference(lattice,steps,phi,sites,singles,pairs,edges,scale=1.,unsigned=False):
    """No parity-frame/compiler/MPS calls: apply original elementary maps densely."""
    n=lattice.n_sites
    singles,pairs,edges=normalize_noise(lattice,steps,singles,pairs,edges)
    rotations,cnots=transfers(phi)
    if unsigned:
        rotations=tuple(tuple(abs(x) for x in row) for row in rotations)
        cnots=tuple(abs(x) for x in cnots)
    labels=np.array(list(itertools.product(range(4),repeat=n)))
    codes=[0]*n
    for q in sites:codes[q]=3
    value=np.zeros((4,)*n);value[tuple(codes)]=1.
    def apply(value,matrix,axes):
        order=list(axes)+[i for i in range(n) if i not in axes]
        out=matrix.reshape(4**len(axes),-1)@value.transpose(order).reshape(4**len(axes),-1)
        return out.reshape((4,)*n).transpose(np.argsort(order))
    for t in reversed(range(steps)):
        for li in reversed(range(len(lattice.layers))):
            bonds=lattice.layers[li].bonds
            if not bonds:continue
            for i,j in bonds:
                value=apply(value,rotations[3][0],(i,));value=apply(value,rotations[3][1],(j,))
            for k in reversed(range(3)):
                exponent=np.zeros(len(labels))
                for q in range(n):
                    for a in range(1,4):
                        anti=(labels[:,q]!=0)&(labels[:,q]!=a)
                        exponent+=singles[t,li,k,q,a-1]*anti
                for e,(i,j) in enumerate(edges):
                    for a,b in itertools.product(range(1,4),repeat=2):
                        anti=((labels[:,i]!=0)&(labels[:,i]!=a))^((labels[:,j]!=0)&(labels[:,j]!=b))
                        exponent+=pairs[t,li,k,e,a-1,b-1]*anti
                value*=np.exp(-2*scale*exponent).reshape(value.shape)
                for i,j in bonds:value=apply(value,cnots[DIRECTIONS[k]],(i,j))
                for i,j in bonds:
                    value=apply(value,rotations[k][0],(i,));value=apply(value,rotations[k][1],(j,))
    terminal=np.ones(len(labels))
    for i,j in zip(lattice.order[::2],lattice.order[1::2]):
        terminal*=labels[:,i]==labels[:,j]
        if not unsigned:terminal*=np.where(labels[:,i]==0,1.,-1.)
    return float(value.ravel()@terminal)


def test_all_elementary_frames_and_restoration():
    for phi in (0.,.23,np.pi/4,np.pi/2):
        for unsigned in (False,True):
            ops=compile_operations(chain(4),1,phi,pair_rates=.001,unsigned=unsigned)
            frames=[Q]*4
            for op in ops:
                assert tuple(frames[q] for q in op.sites)==op.before
                assert check_operation(op)<1e-13
                for q,f in zip(op.sites,op.after):frames[q]=f
            assert frames==[Q]*4
            # Three graph-wide insertions for EACH nonempty matching, not per gate.
            for k in range(3):
                assert sum(op.kind=='pair_noise' and op.location==(0,0,k) for op in ops)==3
                assert sum(op.kind=='single_noise' and op.location==(0,0,k) for op in ops)==4


@pytest.mark.parametrize('unsigned',[False,True])
@pytest.mark.parametrize('scale',[0.,1.3])
def test_mps_all_weight_two_and_scheduled_noise(unsigned,scale):
    n=4;steps=2;lat=chain(n);edges=list(itertools.combinations(range(n),2));rng=np.random.default_rng(19)
    singles=rng.uniform(.001,.013,(steps,2,3,n,3))
    pair=rng.uniform(.0001,.003,(steps,2,3,len(edges),3,3))
    exact=dense_reference(lat,steps,.23,(0,2),singles,pair,edges,scale,unsigned)
    r=evaluate_mps(n=n,steps=steps,sites=(0,2),phi=.23,scale=scale,
        cap=32,guide_cap=32,noise_placement='after_cnot',single_rates=singles,
        pair_rates=pair,noise_edges=edges,unsigned=unsigned)
    assert r['corrected']==pytest.approx(exact,rel=2e-10,abs=2e-11)
    assert r['approximate']==pytest.approx(exact,rel=2e-10,abs=2e-11)
    assert r['details']['charge_leakage']==0
    assert r['details']['final_standard_frame']
    if scale:assert r['details']['routing_swaps']>0


@pytest.mark.parametrize('unsigned',[False,True])
def test_mps_truncated_residual_includes_routing(unsigned):
    edges=[(0,1),(1,2),(2,3),(0,3)]
    kw=dict(n=4,steps=2,sites=(0,2),phi=.23,scale=1.,cap=2,guide_cap=32,
        noise_placement='after_cnot',lam=(.004,.002,.008),pair_rates=.001,
        noise_edges=edges,unsigned=unsigned)
    exact=dense_reference(chain(4),2,.23,(0,2),kw['lam'],.001,edges,unsigned=unsigned)
    r=evaluate_mps(**kw);uncorrected=evaluate_mps(**kw,residual=False)
    assert r['approximate']==pytest.approx(uncorrected['approximate'],abs=1e-12)
    assert abs(r['approximate']-exact)>1e-6
    assert r['corrected']==pytest.approx(exact,rel=2e-9,abs=2e-10)
    assert any(c['kind']=='routing_swap' for c in r['details']['contributions'])
    assert r['details']['guide_snapshot_bound']<len(r['details']['contributions'])


@pytest.mark.parametrize('unsigned',[False,True])
def test_peps_against_direct_dense(unsigned):
    lat=square(2,2);edges=[(0,1),(0,2),(1,3),(2,3)];single=(.001,.003,.007)
    exact=dense_reference(lat,1,.19,(0,1),single,.0002,edges,unsigned=unsigned)
    r=evaluate_peps(nx=2,ny=2,steps=1,sites=(0,1),phi=.19,cap=32,boundary_cap=128,
        lam=single,noise_placement='after_cnot',pair_rates=.0002,noise_edges=edges,unsigned=unsigned)
    assert r['corrected']==pytest.approx(exact,rel=2e-8,abs=2e-9)
    assert r['details']['cnot_evolution_stats']['final_standard_frame']
    assert r['details']['cnot_evolution_stats']['tensor_charge_checks_passed']


def test_peps_long_range_routing():
    lat=square(2,2);edges=[(0,3)]
    exact=dense_reference(lat,1,.13,(0,1),(0,0,0),.0002,edges)
    r=evaluate_peps(nx=2,ny=2,steps=1,sites=(0,1),phi=.13,cap=32,boundary_cap=128,
        lam=(0,0,0),noise_placement='after_cnot',pair_rates=.0002,noise_edges=edges)
    assert r['corrected']==pytest.approx(exact,rel=2e-8,abs=2e-9)
    assert r['details']['cnot_evolution_stats']['routing_swaps']>0


@pytest.mark.parametrize('bad',[-.1,np.nan,np.inf])
def test_reject_invalid_rates(bad):
    with pytest.raises(ValueError):compile_operations(chain(4),1,.2,singles=bad)
    with pytest.raises(ValueError):compile_operations(chain(4),1,.2,pair_rates=bad)


def test_reject_duplicate_and_invalid_edges():
    for edges in [[(0,1),(0,1)],[(1,0)],[(0,4)]]:
        with pytest.raises(ValueError):compile_operations(chain(4),1,.2,edges=edges)


@pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires Metal')
def test_peps_metal_dynamic_charge_maps():
    kw=dict(nx=2,ny=2,steps=1,sites=(0,1),phi=.19,cap=8,boundary_cap=16,
            noise_placement='after_cnot',pair_rates=.0001)
    cpu=evaluate_peps(**kw)
    gpu=evaluate_peps(**kw,gate_backend='metal',boundary_backend='metal-jacobi')
    assert gpu['corrected']==pytest.approx(cpu['corrected'],rel=2e-4,abs=1e-5)


@pytest.mark.parametrize('unsigned',[False,True])
def test_six_qubit_correlated_residual_and_idle_qubits(unsigned):
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(0,5)]
    single=(.002,.003,.006)
    exact=dense_reference(chain(6),1,.21,(1,4),single,.0004,edges,unsigned=unsigned)
    r=evaluate_mps(n=6,steps=1,sites=(1,4),phi=.21,lam=single,cap=4,guide_cap=64,
        noise_placement='after_cnot',pair_rates=.0004,noise_edges=edges,unsigned=unsigned)
    assert r['corrected']==pytest.approx(exact,rel=1e-8,abs=1e-10)
    assert r['details']['charge_leakage']==0


def test_matches_existing_cnot_resolved_solver():
    from cnot_zne.pauli_zne.exact import density_value
    rng=np.random.default_rng(92);n=4;steps=2
    singles=rng.uniform(.001,.01,(steps,4,n,3));pairs=rng.uniform(.0001,.001,(steps,4,n-1,3,3))
    config=dict(circuit=dict(N=n,L=steps,phi=.22,O='ZIZI'),noise=dict(placement='after_cnot'))
    reference=density_value(config,singles,pairs,1.4)
    mapped_single=np.array([[singles[t,[2*parity+d for d in DIRECTIONS]] for parity in range(2)] for t in range(steps)])
    mapped_pair=np.array([[pairs[t,[2*parity+d for d in DIRECTIONS]] for parity in range(2)] for t in range(steps)])
    r=evaluate_mps(n=n,steps=steps,sites=(0,2),phi=.22,scale=1.4,cap=16,guide_cap=16,
        noise_placement='after_cnot',single_rates=mapped_single,pair_rates=mapped_pair)
    assert r['corrected']==pytest.approx(reference,rel=1e-10,abs=1e-11)


def test_elementary_q_curves_at_singular_absolute_rotation_angle():
    # No similarity inversion of absolute transfers is required, including phi=pi/4.
    result=curves(evaluate_mps,[0,1],n=4,steps=2,sites=(0,2),phi=np.pi/4,
                  cap=16,guide_cap=16,noise_placement='after_cnot',pair_rates=.001)
    for row in result['rows']:
        signed=dense_reference(chain(4),2,np.pi/4,(0,2),(.001,.002,.003),.001,None,row['s'])
        unsigned=dense_reference(chain(4),2,np.pi/4,(0,2),(.001,.002,.003),.001,None,row['s'],True)
        assert row['observable_corrected']==pytest.approx(signed,abs=1e-11)
        assert row['unsigned_corrected']==pytest.approx(unsigned,abs=1e-11)


def test_peps_streamed_residual_same_evolved_network():
    from tensor_zne import peps_contractions
    from pauli_peps import evaluate_peps as evolve
    from peps_boundary_residual import scalar
    saved=[]
    class Captured(Exception):pass
    def capture(tn,scale):saved.append((tn.copy(),scale));raise Captured
    lat=square(2,3);pairs=list(zip(lat.order[::2],lat.order[1::2]))
    with pytest.raises(Captured):
        evolve('ZZIIII',lat,nx=2,ny=3,pairs=pairs,n_steps=1,phi=.19,bond_cap=4,
            symmetry='z2z2',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.0001),capture_boundary=capture)
    with peps_contractions():reference=scalar(saved[0][0])*np.exp(saved[0][1])
    r=evaluate_peps(nx=2,ny=3,steps=1,sites=(0,1),phi=.19,cap=4,boundary_cap=1,
                   environment_cap=None,noise_placement='after_cnot',pair_rates=.0001)
    assert r['corrected']==pytest.approx(reference,rel=1e-9,abs=1e-10)


def test_noise_file_cli_and_q_export(tmp_path):
    import csv
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    noise=tmp_path/'noise.npz';prefix=tmp_path/'run'
    np.savez(noise,singles=np.tile([.001,.002,.003],(4,1)),
             pairs=np.full((1,3,3),.0001),edges=np.array([[0,3]]))
    subprocess.run([sys.executable,str(root/'tensor_zne.py'),'mps','--noise-placement','after_cnot',
        '--noise-file',str(noise),'--n','4','--steps','2','--sites','0','1','--cap','16',
        '--guide-cap','16','--scales','0','1','--out',str(prefix)],cwd=root,check=True,capture_output=True,text=True)
    output=json.loads(Path(str(prefix)+'.json').read_text())
    assert output['config']['noise_file_sha256']==hashlib.sha256(noise.read_bytes()).hexdigest()
    assert output['raw'][1]['signed']['details']['routing_swaps']>0
    with Path(str(prefix)+'_q_groups.csv').open() as f:rows=list(csv.DictReader(f))
    assert rows and all(r['method']=='cnot' for r in rows)


@pytest.mark.skipif(os.environ.get('TEST_MPS_METAL')!='1',reason='requires Metal')
def test_cnot_mps_metal_residual_smoke():
    kw=dict(n=6,steps=1,sites=(1,4),cap=8,guide_cap=16,
            noise_placement='after_cnot',pair_rates=.0001)
    cpu=evaluate_mps(**kw);gpu=evaluate_mps(**kw,backend='metal')
    assert gpu['corrected']==pytest.approx(cpu['corrected'],rel=1e-5,abs=1e-8)
