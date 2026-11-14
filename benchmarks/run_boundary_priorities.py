"""Production row telescoping and finite signed environment pilot, fixed PEPS."""
import argparse,json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_boundary_residual import streaming_residuals,scalar
from peps_environment_experiment import guided_residuals
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['row','guided'],required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);rows=[]
settings=([(side,2,False,1.,8,cap) for side in (4,6,8) for cap in (8,16)] if a.mode=='row' else
 [(4,steps,unsigned,scale,8,cap) for steps in (2,6) for unsigned in (False,True) for scale in (0.,1.) for cap in (4,8,16,32)])
cache={}
for side,steps,unsigned,scale,spatial_cap,cap in settings:
 key=(side,steps,unsigned,scale,spatial_cap)
 if key not in cache:
  lat=square(side,side);obs=list('I'*(side*side));obs[side+1]=obs[2*side+2]='Z';saved=[]
  class Captured(Exception):pass
  def capture(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
  tick=time.perf_counter()
  try:evolve(obs,lat,nx=side,ny=side,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=steps,phi=np.pi/16,scale=scale,unsigned=unsigned,bond_cap=spatial_cap,boundary_cap=8,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=capture)
  except Captured:pass
  tn,log_scale=saved[0];evolution=time.perf_counter()-tick
  with peps_contractions():exact=scalar(tn)*np.exp(log_scale) if a.mode=='guided' else None
  cache[key]=(tn,log_scale,evolution,exact)
 tn,log_scale,evolution,exact=cache[key];factor=np.exp(log_scale);methods={}
 with peps_contractions():
  for name in (('bond','row') if a.mode=='row' else ('row','guided','rtm')):
   def run():
    if name in ('guided','rtm'):return guided_residuals(tn,side,side,cap,cap,method='rtm' if name=='rtm' else 'projector')
    return streaming_residuals(tn,side,side,cap,cap,granularity=name)
   run();samples=[]
   for rep in range(a.repeats):
    tick=time.perf_counter();result=run();samples.append(time.perf_counter()-tick)
   methods[name]=dict(seconds=statistics.median(samples),samples=samples,
    approximate=result['approximate']*factor,corrected=result['corrected']*factor,
    overlap_count=result.get('overlap_count'),compression_seconds=result.get('compression_seconds'))
 row=dict(side=side,steps=steps,unsigned=unsigned,scale=scale,spatial_cap=spatial_cap,cap=cap,
          evolution_seconds=evolution,exact=exact,methods=methods)
 rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
