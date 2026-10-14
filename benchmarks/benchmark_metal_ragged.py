"""Replay real PEPS contractions through a custom ragged Metal shader."""
import json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_peps
import peps_metal_layer as layer
from apple_metal_ragged import RaggedProducts
original=layer.contract_block_products;kernel=RaggedProducts();rows=[]
for cap in (8,16):
 captured=[];seen=0;used=0
 def capture(lb,rb,plan,lo,ro):
  global seen,used
  seen+=1;size=sum(x.nbytes for _,x in lb)+sum(x.nbytes for _,x in rb)
  if plan and seen%97==0 and used+size<=32*1024**2:
   captured.append(([(q,x.copy()) for q,x in lb],[(q,x.copy()) for q,x in rb],plan,lo,ro));used+=size
  return original(lb,rb,plan,lo,ro)
 layer.contract_block_products=capture
 try:evaluate_peps(nx=4,ny=4,steps=2,sites=(5,6),cap=cap,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu',noise_placement='after_cnot',pair_rates=.0001)
 finally:layer.contract_block_products=original
 prepared=[];maxerr=0.
 for args in captured:
  ref=original(*args);out=kernel(*args)
  for key in ref:
   np.testing.assert_allclose(out[key],ref[key],atol=5e-6,rtol=2e-4)
   maxerr=max(maxerr,float(np.max(abs(out[key]-ref[key]),initial=0)))
  p=kernel.prepare(*args);kernel.mx.eval(p[0]);prepared.append(p)
 samples={k:[] for k in ('cpu_fp64','shader_end_to_end','shader_resident_dispatch')}
 for rep in range(5):
  for mode in (list(samples) if rep%2==0 else list(samples)[::-1]):
   start=time.perf_counter()
   if mode=='cpu_fp64':
    for args in captured:original(*args)
   elif mode=='shader_end_to_end':
    for args in captured:kernel(*args)
   else:
    for p in prepared:kernel.mx.eval(kernel.dispatch(p))
   samples[mode].append(time.perf_counter()-start)
 row=dict(cap=cap,requests=len(captured),captured_bytes=used,max_absolute_error=maxerr,seconds={k:statistics.median(v) for k,v in samples.items()},samples=samples)
 rows.append(row);print(json.dumps(row),flush=True)
(root/'results/apple_batch3_shader.json').write_text(json.dumps(rows,indent=2)+'\n')
