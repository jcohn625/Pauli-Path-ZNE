"""Calibrate representative real PEPS batch shapes with synthetic values."""
import json, sys, time, statistics
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from apple_silicon_hybrid import SharedBatch
cases=json.loads((root/'results/apple_block_shapes.json').read_text())
selected={}
for case in cases:
 groups=case['gemm_groups']
 def work(row):
  b,m,k,n=row['shape'];return b*m*k*n*row['count']
 for row in sorted(groups,key=lambda r:r['count'],reverse=True)[:3]+sorted(groups,key=work,reverse=True)[:5]:
  selected[tuple(row['shape'])]=True
rng=np.random.default_rng(18);rows=[]
for shape in selected:
 b,m,k,n=shape;pairs=[(rng.normal(size=(m,k)),rng.normal(size=(k,n))) for _ in range(b)]
 shared=SharedBatch(*shape)
 fns={'cpu_matmul':lambda:np.array([a@b for a,b in pairs]),'cpu_einsum':lambda:np.array([np.einsum('ij,jk->ik',a,b,optimize=False) for a,b in pairs]),'metal_shared':lambda:shared(pairs)}
 for fn in fns.values():fn()
 samples={key:[] for key in fns};outputs={}
 for rep in range(15):
  keys=list(fns);keys=keys[rep%3:]+keys[:rep%3]
  for key in keys:
   start=time.perf_counter();outputs[key]=fns[key]();samples[key].append(time.perf_counter()-start)
 ref=outputs['cpu_matmul'];diff=outputs['metal_shared']-ref
 row=dict(shape=shape,seconds={k:statistics.median(v) for k,v in samples.items()},relative_error=float(np.linalg.norm(diff)/np.linalg.norm(ref)))
 rows.append(row);print(json.dumps(row),flush=True)
(root/'results/apple_rectangular_calibration.json').write_text(json.dumps(rows,indent=2)+'\n')
