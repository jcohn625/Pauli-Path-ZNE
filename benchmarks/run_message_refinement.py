"""Bidirectional messages vs ordinary residuals at multiple costs and caps."""
import argparse,json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_boundary_residual import scalar,streaming_residuals
from peps_message_refinement import refined_residuals
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);rows=[]
for steps in (2,6):
 for unsigned in (False,True):
  for scale in (0.,1.):
   lat=square(4,4);obs=list('I'*16);obs[5]=obs[10]='Z';saved=[]
   class Captured(Exception):pass
   def capture(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
   try:evolve(obs,lat,nx=4,ny=4,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=steps,phi=np.pi/16,scale=scale,unsigned=unsigned,bond_cap=8,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=capture)
   except Captured:pass
   tn,log_scale=saved[0];factor=np.exp(log_scale)
   with peps_contractions():
    exact=scalar(tn)*factor
    for cap in (4,8,16,32):
     methods={}
     for name in ('ordinary','larger_guide','larger_both','refined_1','refined_3','refined_6'):
      def run():
       if name.startswith('refined'):
        return refined_residuals(tn,4,4,cap,cap,sweeps=int(name.split('_')[1]))
       return streaming_residuals(tn,4,4,cap*2 if name=='larger_both' else cap,
                                  cap if name=='ordinary' else cap*2,granularity='row')
      run();samples=[]
      for rep in range(3):
       tick=time.perf_counter();result=run();samples.append(time.perf_counter()-tick)
      history=result.get('message_history')
      if history:
       for h in history:
        h['cut_values']=[v*factor for v in h['cut_values']];h['mean']*=factor
      methods[name]=dict(seconds=statistics.median(samples),samples=samples,
                         corrected=result['corrected']*factor,approximate=result['approximate']*factor,
                         history=history)
     row=dict(steps=steps,unsigned=unsigned,scale=scale,spatial_cap=8,cap=cap,exact=exact,methods=methods)
     rows.append(row);a.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
