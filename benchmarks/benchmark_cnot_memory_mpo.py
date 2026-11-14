"""Benchmark retained-guide budgets and the experimental diagonal-noise MPO."""
import importlib.util,json,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_mps
from lattice_circuit import chain
spec=importlib.util.spec_from_file_location('dense_checks',root/'tests/test_cnot_parity.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
rows=[];output=root/'results/cnot_memory_mpo.json'
def run(label,settings,exact=None):
    evaluate_mps(**settings);times=[]
    for _ in range(3):
        start=time.perf_counter();r=evaluate_mps(**settings);times.append(time.perf_counter()-start)
    d=r['details'];row=dict(label=label,settings=settings,seconds=times,median_s=statistics.median(times),
        approximate=r['approximate'],corrected=r['corrected'],exact=exact,
        error=None if exact is None else abs(r['corrected']-exact),
        guide_cache_peak_bytes=d['guide_cache_peak_bytes'],noise_mpo_max_bond=d['noise_mpo_max_bond'],
        noise_mpo_bytes=d['noise_mpo_bytes'],routing_swaps=d['routing_swaps'],
        residual_skipped=d['stats']['residual_skipped'],contribution_count=d['contribution_count'])
    rows.append(row);output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
base=dict(n=8,steps=1,sites=(0,1),phi=.23,cap=8,guide_cap=16,
          noise_placement='after_cnot',pair_rates=.0005,store_contributions=False)
for budget in (None,.0625,0.):run('guide budget',dict(base,guide_cache_mb=budget))
base.update(steps=2,guide_cap=32)
for label,edges in [('nearest neighbor',None),('long range star',[(0,5),(0,6),(0,7)])]:
    for unsigned in (False,True):
        exact=checks.dense_reference(chain(8),2,.23,(0,1),(.001,.002,.003),.0005,edges,unsigned=unsigned)
        for strategy in ('local','mpo'):
            run(label,dict(base,noise_edges=edges,unsigned=unsigned,noise_strategy=strategy),exact)
