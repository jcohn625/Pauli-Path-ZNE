"""Compare row telescoping with bond-by-bond residual evaluation."""
import json, statistics, sys, time, resource
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_boundary_residual import streaming_residuals
from peps_bethe_experiment import row_residual_contract
output=root/'results/exploration_20261013/row_residual.json';rows=[]
for side in (4,6,8):
 lat=square(side,side);obs=list('I'*(side*side));obs[side+1]=obs[2*side+2]='Z';saved=[]
 class Captured(Exception):pass
 def capture(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
 start=time.perf_counter()
 try:evolve(obs,lat,nx=side,ny=side,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=2,phi=np.pi/16,scale=1.,unsigned=False,bond_cap=8,boundary_cap=8,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=capture)
 except Captured:pass
 tn,log_scale=saved[0];evolution=time.perf_counter()-start
 with peps_contractions():
  for cap in (8,16):
   records={}
   for mode,fn in (('bond',streaming_residuals),('row',row_residual_contract)):
    fn(tn,side,side,cap,cap);samples=[]
    for rep in range(3):
     start=time.perf_counter();r=fn(tn,side,side,cap,cap);samples.append(time.perf_counter()-start)
    records[mode]=dict(seconds=statistics.median(samples),raw=r['approximate'],corrected=r['corrected'])
   row=dict(side=side,spatial_cap=8,cap=cap,evolution_seconds=evolution,log_scale=log_scale,methods=records,speedup=records['bond']['seconds']/records['row']['seconds'],corrected_delta=abs(records['bond']['corrected']-records['row']['corrected'])*np.exp(log_scale),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
   rows.append(row);print(json.dumps(row),flush=True);output.write_text(json.dumps(rows,indent=2)+'\n')
