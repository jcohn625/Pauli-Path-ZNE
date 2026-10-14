"""Compare contraction estimators on exactly the same evolved CNOT-noise PEPS."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_boundary_residual import streaming_residuals
from peps_bethe_experiment import bethe_contract
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,nargs='+',default=[2,4]);p.add_argument('--spatial-cap',type=int,default=4);p.add_argument('--scales',nargs='+',type=float,default=[0.,1.,3.]);p.add_argument('--caps',nargs='+',type=int,default=[2,4,8,16]);p.add_argument('--output',type=Path,default=root/'results/exploration_20261013/estimators.json');a=p.parse_args()
rows=[]
for steps in a.steps:
 for unsigned in (False,True):
  for scale in a.scales:
   lat=square(4,4);obs=list('I'*16);obs[5]=obs[10]='Z';saved=[]
   class Captured(Exception):pass
   def capture(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
   start=time.perf_counter()
   try:evolve(obs,lat,nx=4,ny=4,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=steps,phi=np.pi/16,scale=scale,unsigned=unsigned,bond_cap=a.spatial_cap,boundary_cap=8,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=capture)
   except Captured:pass
   tn,log_scale=saved[0];evolve_seconds=time.perf_counter()-start;factor=np.exp(log_scale)
   with peps_contractions():
    start=time.perf_counter();exact=float(tn.contract(all,optimize='greedy'))*factor;exact_seconds=time.perf_counter()-start
    for cap in a.caps:
     b=bethe_contract(tn,4,4,cap);r=streaming_residuals(tn,4,4,cap,cap)
     values=dict(raw=r['approximate']*factor,residual=r['corrected']*factor,bethe=None if b['value'] is None else b['value']*factor)
     row=dict(steps=steps,unsigned=unsigned,scale=scale,cap=cap,spatial_cap=a.spatial_cap,exact=exact,values=values,
              absolute_errors={k:None if v is None else abs(v-exact) for k,v in values.items()},
              relative_errors={k:None if v is None or exact==0 else abs(v/exact-1) for k,v in values.items()},
              evolve_seconds=evolve_seconds,exact_seconds=exact_seconds,bethe_seconds=b['seconds'],residual_seconds=r['seconds'],bethe_status=b['status'])
     rows.append(row);print(json.dumps(row),flush=True)
     a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
