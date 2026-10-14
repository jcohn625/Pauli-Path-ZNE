"""Persistent CPU worker pools and overlapping independent CPU/Metal batches."""
import json,statistics,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import scipy.linalg as la
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from apple_jacobi_experiment import AdaptiveJacobi
engine=AdaptiveJacobi(deflation_tol=1e-7);mx=engine.mx
output=root/'results/exploration_20261013/scheduling.json'
captured=np.load(output.with_name('adaptive_real.npz'));rows=[]
def gpu(x):
 if not len(x):return []
 dev=mx.array(x,dtype=mx.float32);scale=mx.maximum(mx.max(mx.abs(dev),axis=(-2,-1),keepdims=True),1e-30)
 u,s,v=engine.jacobi(dev/scale);out=(u,s*scale[...,0],v);mx.eval(out)
 return [np.asarray(a).astype(np.float64) for a in out]
def cpu(x):return la.svd(x,full_matrices=False,check_finite=False)
rng=np.random.default_rng(19)
for n in (32,64,128,256):
 matrices=captured[str(n)] if str(n) in captured.files else [rng.normal(size=(n,n))]
 x=np.stack([matrices[i%len(matrices)] for i in range(32)])
 for precision in ('float64','float32'):
  inp=x.astype(precision)
  for workers in (1,2,4,8):
   with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(cpu,inp));samples=[]
    for rep in range(5):
     start=time.perf_counter();out=list(pool.map(cpu,inp));samples.append(time.perf_counter()-start)
   rows.append(dict(n=n,batch=32,captured=str(n) in captured.files,mode='cpu',precision=precision,workers=workers,gpu_count=0,seconds=statistics.median(samples)))
 for gpu_count in (8,16,24,32):
  with ThreadPoolExecutor(max_workers=4) as pool:
   def run():
    jobs=[pool.submit(cpu,a) for a in x[gpu_count:]]
    out=gpu(x[:gpu_count]);other=[j.result() for j in jobs];return out,other
   run();samples=[]
   for rep in range(5):
    start=time.perf_counter();out=run();samples.append(time.perf_counter()-start)
  rows.append(dict(n=n,batch=32,captured=str(n) in captured.files,mode='gpu' if gpu_count==32 else 'hybrid_overlap',precision='mixed' if gpu_count<32 else 'float32',workers=4 if gpu_count<32 else 0,gpu_count=gpu_count,seconds=statistics.median(samples)))
 output.write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps({'n':n,'measurements':rows[-12:]}),flush=True)
