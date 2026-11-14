"""Full warmed boundary contraction: CPU FP64/FP32, Metal, and selective LAPACK."""
import argparse,copy,json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lattice_circuit import square
from pauli_peps import evaluate_peps as evolve
from tensor_zne import peps_contractions
from peps_gpu_boundary import gpu_boundary,to_gpu
from peps_boundary_residual import streaming_residuals
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--large-only',action='store_true');a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);rows=[];capture={}
settings=[(4,6,8,32,False,0.),(4,6,8,32,False,1.),(4,6,8,32,True,1.),(6,2,16,32,False,1.)]
if a.large_only:settings=[s for s in settings if s[0]==6]
for side,steps,spatial_cap,cap,unsigned,scale in settings:
 lat=square(side,side);obs=list('I'*(side*side));obs[side+1]=obs[2*side+2]='Z';saved=[]
 class Captured(Exception):pass
 def save(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
 tick=time.perf_counter()
 try:evolve(obs,lat,nx=side,ny=side,pairs=list(zip(lat.order[::2],lat.order[1::2])),n_steps=steps,phi=np.pi/16,scale=scale,unsigned=unsigned,bond_cap=spatial_cap,symmetry='z2z2',gate_backend='batched-cpu',cnot_noise=dict(singles=(.001,.002,.003),pair_rates=.001,edges=None),capture_boundary=save)
 except Captured:pass
 tn,log_scale=saved[0];evolution=time.perf_counter()-tick;methods={}
 with peps_contractions():
  for precision in ('float64','float32'):
   net=tn.copy()
   for t in net:t.modify(data=t.data.copy_with(blocks={c:b.astype(precision) for c,b in t.data.blocks.items()}))
   def run_cpu():return streaming_residuals(net,side,side,cap,cap,granularity='row')
   run_cpu();samples=[]
   for rep in range(3):
    tick=time.perf_counter();result=run_cpu();samples.append(time.perf_counter()-tick)
   methods['cpu_'+precision]=dict(seconds=statistics.median(samples),samples=samples,corrected=result['corrected']*np.exp(log_scale),approximate=result['approximate']*np.exp(log_scale))
  for name,qmin,smin in (('metal',None,None),('metal_panel16',None,None),('hybrid_large',256,64),('hybrid_medium',64,16),('hybrid_all',1,1)):
   with gpu_boundary(32,decomposition_backend='metal' if name.startswith('metal') else 'hybrid',cpu_qr_min=qmin,cpu_svd_min=smin,qr_panel_width=16 if name=='metal_panel16' else 32) as engine:
    net=to_gpu(tn,engine.mx)
    def materialize(net):engine.mx.eval([b for t in net for b in t.data.blocks.values()])
    if side==6 and name=='metal':
     originals={m:getattr(engine,m) for m in ('qr','svd')}
     def wrap(method):
      def wrapped(array):
       shape=tuple(array.shape)
       key=f'{method}_{shape[0]}_{shape[1]}_{shape[2]}'
       if min(shape[-2:])>=128 and key not in capture and len(capture)<30:
        engine.mx.eval(array);capture[key]=np.array(array)
       return originals[method](array)
      return wrapped
     for method in originals:setattr(engine,method,wrap(method))
    def run_gpu():
     return streaming_residuals(net,side,side,cap,cap,granularity='row',materialize=materialize)
    run_gpu()
    if side==6 and name=='metal':
     for method in originals:setattr(engine,method,originals[method])
     np.savez(a.output.with_suffix('.npz'),**capture)
    empty=copy.deepcopy(engine.stats);samples=[]
    for rep in range(3):
     tick=time.perf_counter();result=run_gpu();samples.append(time.perf_counter()-tick)
    methods[name]=dict(seconds=statistics.median(samples),samples=samples,corrected=result['corrected']*np.exp(log_scale),approximate=result['approximate']*np.exp(log_scale),stats=engine.stats)
 row=dict(side=side,steps=steps,spatial_cap=spatial_cap,cap=cap,unsigned=unsigned,scale=scale,evolution_seconds=evolution,methods=methods)
 rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
