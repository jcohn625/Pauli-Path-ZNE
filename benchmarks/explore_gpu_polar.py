"""GEMM-only Newton-Schulz polar experiment for canonicalization, not SVD.

Simple fixed iteration baseline, not the stabilized TPU polar implementation.
Never selected by the solvers. Rank-deficient matrices need separate treatment.
"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import mlx.core as mx
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from mps_qr import economic_qr
output=root/'results/exploration_20261013/polar.json';rows=[];rng=np.random.default_rng(32)
for n in (64,128,256):
 for condition in (10.,1e6):
  matrices=[]
  for j in range(4):
   q,_=la.qr(rng.normal(size=(2*n,n)),mode='economic');v,_=la.qr(rng.normal(size=(n,n)))
   matrices.append((q*np.geomspace(1,1/condition,n))@v.T)
  x=np.stack(matrices)
  samples=[]
  for rep in range(4):
   start=time.perf_counter();qr=[economic_qr(z) for z in x]
   if rep:samples.append(time.perf_counter()-start)
  cpu_seconds=statistics.median(samples)
  for iterations in (12,24,40):
   @mx.compile
   def polar(a):
    q=a/mx.sqrt(mx.sum(a*a,axis=(-2,-1),keepdims=True))
    ident=mx.eye(n,dtype=mx.float32)
    for _ in range(iterations):q=.5*q@(3*ident-mx.swapaxes(q,-1,-2)@q)
    return q,mx.swapaxes(q,-1,-2)@a
   def run():
    out=polar(mx.array(x,dtype=mx.float32));mx.eval(out)
    return [np.asarray(a).astype(np.float64) for a in out]
   run();samples=[]
   for rep in range(3):
    start=time.perf_counter();q,h=run();samples.append(time.perf_counter()-start)
   row=dict(n=n,m=2*n,batch=4,condition=condition,iterations=iterations,cpu_qr_seconds=cpu_seconds,gpu_seconds=statistics.median(samples),orthogonality=max(la.norm(z.T@z-np.eye(n)) for z in q),relative_reconstruction=max(la.norm(qi@hi-xi)/la.norm(xi) for qi,hi,xi in zip(q,h,x)))
   rows.append(row);print(json.dumps(row),flush=True);output.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'device':mx.device_info()},default=str),flush=True)
