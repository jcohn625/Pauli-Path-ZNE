"""Local Apple Jacobi experiments; timing includes packing and host return."""
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from apple_jacobi_experiment import AdaptiveJacobi
from tensor_zne import evaluate_mps
p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',type=int,default=[16,32,64,128,256]);p.add_argument('--batches',nargs='+',type=int,default=[1,4,16,32]);p.add_argument('--deflation-tol',type=float,default=0.);p.add_argument('--output',type=Path,default=root/'results/exploration_20261013/adaptive_jacobi.json');args=p.parse_args()
original=la.svd;captured={n:[] for n in args.sizes}
def capture(mat,*a,**kw):
 n=mat.shape[0]
 if mat.shape==(n,n) and n in captured and len(captured[n])<8:captured[n].append(mat.copy())
 return original(mat,*a,**kw)
la.svd=capture
try:
 evaluate_mps(n=12,steps=3,sites=(3,8),cap=max(args.sizes),guide_cap=max(args.sizes),noise_placement='after_cnot',pair_rates=.0001)
 for limit in args.sizes:
  if not captured[limit]:
   evaluate_mps(n=12,steps=3,sites=(3,8),cap=limit,guide_cap=limit,noise_placement='after_cnot',pair_rates=.0001)
finally:la.svd=original
args.output.parent.mkdir(parents=True,exist_ok=True)
np.savez(args.output.with_suffix('.npz'),**{str(n):np.stack(v) for n,v in captured.items() if v})
rows=[];rng=np.random.default_rng(19)
for n in args.sizes:
 matrices=captured[n] or [rng.normal(size=(n,n))]
 for batch in args.batches:
  x=np.stack([matrices[j%len(matrices)] for j in range(batch)])
  ref=[la.svd(z,full_matrices=False) for z in x]
  for precision in ('float64','float32'):
   inp=x.astype(precision);samples=[]
   for rep in range(5):
    start=time.perf_counter();out=[la.svd(z,full_matrices=False) for z in inp]
    if rep:samples.append(time.perf_counter()-start)
   rows.append(dict(n=n,batch=batch,mode='cpu_'+precision,seconds=statistics.median(samples),captured=bool(captured[n])))
  for shared,threads in ((False,256),(True,256),(True,128)):
   engine=AdaptiveJacobi(sweeps=24,threads=threads,shared=shared,deflation_tol=args.deflation_tol);mx=engine.mx
   def run():
    dev=mx.array(x,dtype=mx.float32);scale=mx.maximum(mx.max(mx.abs(dev),axis=(-2,-1),keepdims=True),1e-30)
    u,s,v,info=engine.jacobi_info(dev/scale);out=(u,s*scale[...,0],v,info);mx.eval(out)
    return [np.array(z).astype(np.float64) for z in out]
   out=run();samples=[]
   for rep in range(4):
    start=time.perf_counter();out=run();samples.append(time.perf_counter()-start)
   u,s,v,info=out;metrics={k:[] for k in ('reconstruction','singular_values','active_u_orthogonality','v_orthogonality','full_u_orthogonality')}
   for j in range(batch):
    metrics['reconstruction'].append(la.norm((u[j]*s[j])@v[j]-x[j])/max(la.norm(x[j]),1e-30))
    metrics['singular_values'].append(la.norm(s[j]-ref[j][1])/max(la.norm(ref[j][1]),1e-30))
    keep=s[j]>1e-5*max(s[j,0],1e-30)
    metrics['active_u_orthogonality'].append(la.norm(u[j][:,keep].T@u[j][:,keep]-np.eye(sum(keep))))
    metrics['full_u_orthogonality'].append(la.norm(u[j].T@u[j]-np.eye(n)))
    metrics['v_orthogonality'].append(la.norm(v[j]@v[j].T-np.eye(n)))
   row=dict(n=n,batch=batch,mode=f'adaptive_shared{shared}_t{threads}',seconds=statistics.median(samples),captured=bool(captured[n]),deflation_tol=args.deflation_tol,sweeps_max=int(info[:,0].max()),converged_all=bool(info[:,1].all()),**{k:float(max(v)) for k,v in metrics.items()})
   rows.append(row);print(json.dumps(row),flush=True)
   args.output.write_text(json.dumps(rows,indent=2)+'\n')
