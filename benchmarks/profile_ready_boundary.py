"""Observe actual same-shape batches within sequential PEPS boundary splits."""
import argparse,copy,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_gpu_boundary import gpu_boundary,to_gpu
from peps_boundary_residual import streaming_residuals
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
a.output.parent.mkdir(parents=True,exist_ok=True)
for side,steps,spatial_cap,boundary_cap in ((4,6,8,32),(6,2,16,32)):
 lat=square(side,side);obs=list('I'*(side*side));obs[side+1]=obs[2*side+2]='Z';saved=[]
 class Captured(Exception):pass
 def capture(tn,log_scale):saved.append(tn.copy());raise Captured
 try:evolve(obs,lat,nx=side,ny=side,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=steps,phi=np.pi/16,scale=1.,bond_cap=spatial_cap,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=capture)
 except Captured:pass
 with peps_contractions(),gpu_boundary(32,profile_decompositions=True) as engine:
  tn=to_gpu(saved[0],engine.mx)
  def materialize(net):engine.mx.eval([b for t in net for b in t.data.blocks.values()])
  empty_stats=copy.deepcopy(engine.stats)
  streaming_residuals(tn,side,side,boundary_cap,boundary_cap,granularity='row',materialize=materialize)
  engine.stats=empty_stats
  result=streaming_residuals(tn,side,side,boundary_cap,boundary_cap,granularity='row',materialize=materialize)
  row=dict(side=side,steps=steps,spatial_cap=spatial_cap,boundary_cap=boundary_cap,
           result=result,stats=engine.stats)
 rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ('result','stats')}),flush=True)
