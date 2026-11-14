"""Numerical equivalence and bounded-memory contracts for CNOT optimizations."""
import numpy as np
import pytest
from cnot_parity import Q,Operation,compact_operation,compile_operations
from cnot_parity_mps import FrameMPS
from cnot_noise_mpo import build_noise_mpo
from lattice_circuit import chain
from tensor_zne import evaluate_mps,evaluate_peps
from test_cnot_parity import dense_reference


@pytest.mark.parametrize('unsigned',[False,True])
@pytest.mark.parametrize('budget',[0.,.001,.02])
def test_budgeted_guides_and_streamed_diagnostics(unsigned,budget):
    kw=dict(n=4,steps=2,sites=(0,2),cap=2,guide_cap=32,unsigned=unsigned,
            noise_placement='after_cnot',pair_rates=.001)
    reference=evaluate_mps(**kw);items=[]
    actual=evaluate_mps(**kw,guide_cache_mb=budget,store_contributions=False,contribution_callback=items.append)
    assert actual['corrected']==pytest.approx(reference['corrected'],abs=2e-12)
    assert actual['approximate']==reference['approximate']
    assert actual['details']['contributions']==[]
    assert len(items)==actual['details']['contribution_count']
    assert actual['details']['guide_cache_peak_bytes']<=int(budget*1024**2)
    assert actual['details']['stats']['residual_skipped']>0


def test_skip_only_full_splits_without_positive_discarded_values():
    state=FrameMPS.observable([3,0]);frame=(Q,Q);calls=[]
    state.apply(Operation((0,1),np.eye(16),frame,frame,'identity'),16,lambda *args:calls.append(1))
    assert calls==[] and state.stats['residual_skipped']==1
    # Tiny but positive singular value is discarded by the numerical cutoff.
    matrix=np.eye(16);matrix[:,0]=0;matrix[0,0]=1;matrix[5,0]=1e-15
    state=FrameMPS.observable([0,0])
    state.apply(Operation((0,1),matrix,frame,frame,'test'),16,lambda *args:calls.append(1))
    assert calls==[1]


@pytest.mark.parametrize('unsigned',[False,True])
@pytest.mark.parametrize('budget',[None,0.])
def test_mpo_correlated_noise_residual_matches_dense(unsigned,budget):
    edges=[(0,1),(1,2),(2,3),(0,3),(0,2)]
    exact=dense_reference(chain(4),2,.23,(0,2),(.002,.004,.006),.001,edges,unsigned=unsigned)
    r=evaluate_mps(n=4,steps=2,sites=(0,2),phi=.23,lam=(.002,.004,.006),
        cap=2,guide_cap=32,pair_rates=.001,noise_edges=edges,unsigned=unsigned,
        noise_placement='after_cnot',noise_strategy='mpo',guide_cache_mb=budget)
    assert r['corrected']==pytest.approx(exact,abs=3e-11)
    assert r['details']['charge_leakage']==0
    assert r['details']['routing_swaps']==0
    assert r['details']['noise_mpo_max_bond']<=16


def test_noise_mpo_full_diagonal_and_frontier_guard():
    edges=[(0,4),(1,5),(2,5)]
    ops=compile_operations(chain(6),1,.23,pair_rates=.001,edges=edges)
    group=[x for x in ops if x.kind in ('single_noise','pair_noise') and x.location==(0,0,0)]
    mpo=build_noise_mpo(group,6,[Q]*6,max_bond=64)
    from itertools import product
    for codes in product(range(4),repeat=6):
        value=np.ones(1);exact=1.
        for q,t in enumerate(mpo.tensors):value=value@t[:,codes[q],:]
        for op in group:
            index=0
            for q in op.sites:index=4*index+codes[q]
            exact*=np.diag(op.matrix)[index]
        assert value[0]==pytest.approx(exact,abs=2e-15)
    with pytest.raises(ValueError,match='exceeding'):build_noise_mpo(group,6,[Q]*6,max_bond=4)


def test_compact_permutation_and_diagonal_adjoint():
    from cnot_parity import SWAP,CompactOperation
    for matrix in (SWAP,np.diag(np.arange(16)),np.eye(16)[np.roll(np.arange(16),1)]):
        op=compact_operation(Operation((0,1),matrix,(Q,Q),(Q,Q),'test'))
        assert isinstance(op,CompactOperation)
        np.testing.assert_array_equal(op.matrix,matrix)
        np.testing.assert_array_equal(op.adjoint().matrix,matrix.T)
        assert op.values.nbytes<matrix.nbytes


def test_direct_diagonal_peps_matches_generic(monkeypatch):
    import cnot_parity_peps as module
    # Disable specialization by replacing only the gate constructor with its
    # exact dense/sparse equivalent; route through the same CPU layer engine.
    kw=dict(nx=2,ny=3,steps=1,sites=(0,1),cap=8,boundary_cap=16,
            noise_placement='after_cnot',pair_rates=.001,gate_backend='batched-cpu')
    specialized=evaluate_peps(**kw)
    import peps_metal_layer
    def generic(t,gate,left,right,output):
        from quimb.tensor import Tensor
        op=Operation((0,1),np.diag(gate.values),gate.maps,gate.maps,'pair_noise')
        sparse=module.sparse_gate(op)
        return t@Tensor(sparse,inds=[*output,left,right])
    monkeypatch.setattr(peps_metal_layer,'diagonal_center',generic)
    reference=evaluate_peps(**kw)
    assert specialized['corrected']==pytest.approx(reference['corrected'],abs=2e-11)
    assert specialized['details']['cnot_evolution_stats']['diagonal_updates']>0


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf')])
def test_invalid_guide_budget(bad):
    with pytest.raises(ValueError,match='guide_cache_mb'):
        evaluate_mps(n=4,sites=(0,2),noise_placement='after_cnot',guide_cache_mb=bad)


def test_mpo_memory_cli(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];prefix=tmp_path/'mpo'
    subprocess.run([sys.executable,str(root/'tensor_zne.py'),'mps','--noise-placement','after_cnot',
        '--noise-strategy','mpo','--noise-mpo-max-bond','16','--guide-cache-mb','0.01',
        '--summary-only','--pair-rate','0.001','--n','4','--steps','2','--sites','0','2',
        '--cap','2','--guide-cap','32','--scales','0','1','--out',str(prefix)],
        cwd=root,check=True,capture_output=True,text=True)
    data=json.loads(Path(str(prefix)+'.json').read_text())
    details=data['raw'][1]['signed']['details']
    assert details['contributions']==[] and details['contribution_count']>0
    assert details['guide_cache_peak_bytes']<=int(.01*1024**2)
    assert details['noise_strategy']=='mpo' and details['noise_mpo_max_bond']==4
