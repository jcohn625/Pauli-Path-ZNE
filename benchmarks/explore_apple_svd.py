"""Warmed CPU and custom Metal SVD comparison on captured MPS sector matrices."""
import argparse,json,sys,time,statistics
from pathlib import Path
import numpy as np
import scipy.linalg as la
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_mps
from peps_gpu_decomp import GPUDecompositions
p=argparse.ArgumentParser();p.add_argument('--sizes',type=int,nargs='+',default=[16,32,64,128,256]);p.add_argument('--batches',type=int,nargs='+',default=[1,4,16]);p.add_argument('--sweeps',type=int,nargs='+',default=[8,16,32]);p.add_argument('--output',type=Path,default=root/'results/exploration_20261013/jacobi.json');a=p.parse_args()
original=la.svd;captured={n:[] for n in a.sizes}
def capture(mat,*args,**kw):
 n=mat.shape[0]
 if mat.shape==(n,n) and n in captured and len(captured[n])<8:captured[n].append(mat.copy())
 return original(mat,*args,**kw)
la.svd=capture
try:evaluate_mps(n=12,steps=3,sites=(3,8),cap=max(a.sizes),guide_cap=max(a.sizes),noise_placement='after_cnot',pair_rates=.0001)
finally:la.svd=original
rng=np.random.default_rng(19);rows=[]
for n in a.sizes:
 matrices=captured[n]
 if not matrices:matrices=[rng.normal(size=(n,n))]
 for batch in a.batches:
  x=np.stack([matrices[i%len(matrices)] for i in range(batch)])
  ref=[la.svd(z,full_matrices=False) for z in x]
  for precision in ('float64','float32'):
   inp=x.astype(precision);samples=[]
   for rep in range(4):
    start=time.perf_counter();out=[la.svd(z,full_matrices=False) for z in inp]
    if rep:samples.append(time.perf_counter()-start)
   rows.append(dict(n=n,batch=batch,mode='cpu_'+precision,seconds=statistics.median(samples),captured=bool(captured[n])))
  for sweeps in a.sweeps:
   engine=GPUDecompositions(sweeps=sweeps);mx=engine.mx
   for mode in ('qr_jacobi','direct_jacobi'):
    def run(inp):
     if mode=='qr_jacobi':return engine.svd(inp)
     scale=mx.maximum(mx.max(mx.abs(inp),axis=(-2,-1),keepdims=True),1e-30)
     u,s,v=engine.jacobi(inp/scale)
     return u,s*scale[...,0],v
    dev=mx.array(x,dtype=mx.float32);out=run(dev);mx.eval(out)
    samples=[]
    for rep in range(3):
     start=time.perf_counter();out=run(mx.array(x,dtype=mx.float32));mx.eval(out)
     host=[np.array(z).astype(np.float64) for z in out];samples.append(time.perf_counter()-start)
    u,s,v=host;reconstruction=[];orthogonal=[];active=[];serr=[]
    for j in range(batch):
     reconstruction.append(float(la.norm((u[j]*s[j])@v[j]-x[j])/max(la.norm(x[j]),1e-30)))
     orthogonal.append(float(la.norm(u[j].T@u[j]-np.eye(n))))
     k=s[j]>1e-5*max(s[j,0],1e-30)
     active.append(float(la.norm(u[j][:,k].T@u[j][:,k]-np.eye(sum(k)))))
     serr.append(float(la.norm(s[j]-ref[j][1])/max(la.norm(ref[j][1]),1e-30)))
    row=dict(n=n,batch=batch,mode=mode,sweeps=sweeps,seconds=statistics.median(samples),captured=bool(captured[n]),relative_reconstruction=max(reconstruction),singular_value_relative_error=max(serr),full_u_orthogonality=max(orthogonal),active_u_orthogonality=max(active))
    rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
