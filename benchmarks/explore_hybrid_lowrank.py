"""FP32 GPU range finding with FP64 CPU QR/core SVD, versus CPU algorithms.

This is a low-rank algorithm experiment, not the production SVD dispatcher.
Includes every host/device synchronization and final reconstruction factors.
"""
import json, statistics, sys, time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import mlx.core as mx
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from scipy.linalg.blas import dgemm
rng=np.random.default_rng(27);rows=[]
output=root/'results/exploration_20261013/hybrid_lowrank.json'
captured=np.load(output.with_name('adaptive_real.npz'))
def mm(a,b):return dgemm(1.,b.T,a.T).T
# Same sketch in each mode, batched across independent inputs.
def lowrank(x,omega,k,mode,iterations):
 gpu=mode!='cpu_randomized';batch=x.shape[0]
 dev=mx.array(x,dtype=mx.float32) if gpu else None
 def multiply(a,b):return np.stack([mm(aa,bb) for aa,bb in zip(a,b)])
 def product_gpu(a,b):
  out=a@b;mx.eval(out);return np.asarray(out).astype(np.float64)
 y=product_gpu(dev,mx.array(omega,dtype=mx.float32)) if gpu else multiply(x,omega)
 q=np.stack([la.qr(z,mode='economic',check_finite=False)[0] for z in y])
 for _ in range(iterations):
  z=product_gpu(mx.swapaxes(dev,-1,-2),mx.array(q,dtype=mx.float32)) if gpu else multiply(x.transpose(0,2,1),q)
  z=np.stack([la.qr(a,mode='economic',check_finite=False)[0] for a in z])
  y=product_gpu(dev,mx.array(z,dtype=mx.float32)) if gpu else multiply(x,z)
  q=np.stack([la.qr(a,mode='economic',check_finite=False)[0] for a in y])
 if mode=='gpu_range_and_core':b=product_gpu(mx.array(q.transpose(0,2,1),dtype=mx.float32),dev)
 else:b=multiply(q.transpose(0,2,1),x)
 factors=[]
 for qi,bi in zip(q,b):
  u,s,v=la.svd(bi,full_matrices=False,check_finite=False)
  factors.append((mm(qi,u[:,:k]),s[:k],v[:k]))
 return factors
cases=[('captured256',np.stack([captured['256'][i%len(captured['256'])] for i in range(4)]))]
for n,decay in ((512,12.),(512,1.),(1024,12.)):
 x=[]
 for _ in range(4):
  q,_=la.qr(rng.normal(size=(n,n)),check_finite=False);v,_=la.qr(rng.normal(size=(n,n)),check_finite=False)
  x.append(mm(q*np.geomspace(1,10**-decay,n),v.T))
 cases.append((f'synthetic{n}_decay{decay}',np.stack(x)))
for name,x in cases:
 n=x.shape[-1];ref=[la.svd(z,full_matrices=False,check_finite=False) for z in x]
 for k in (32,64):
  omega=rng.normal(size=(len(x),n,k+8));optimal=[la.norm(s[k:]) for u,s,v in ref]
  for mode in ('cpu_full','cpu_randomized','gpu_range_cpu_core','gpu_range_and_core'):
   for iterations in ((0,) if mode=='cpu_full' else (0,1,2)):
    def run():
     if mode=='cpu_full':return [(u[:,:k],s[:k],v[:k]) for u,s,v in [la.svd(z,full_matrices=False,check_finite=False) for z in x]]
     return lowrank(x,omega,k,mode,iterations)
    run();samples=[]
    for rep in range(3):
     start=time.perf_counter();f=run();samples.append(time.perf_counter()-start)
    err=[la.norm((u*s)@v-a) for (u,s,v),a in zip(f,x)]
    row=dict(case=name,n=n,batch=len(x),k=k,mode=mode,iterations=iterations,seconds=statistics.median(samples),relative_residual=max(e/la.norm(a) for e,a in zip(err,x)),optimal_relative_residual=max(e/la.norm(a) for e,a in zip(optimal,x)),residual_over_optimal=max(e/max(o,1e-30) for e,o in zip(err,optimal)),u_orthogonality=max(la.norm(u.T@u-np.eye(k)) for u,s,v in f))
    rows.append(row);print(json.dumps(row),flush=True);output.write_text(json.dumps(rows,indent=2)+'\n')
