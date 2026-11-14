"""End-to-end CPU/Metal/shared-buffer crossover, including FP64->FP32->FP64."""
import importlib.metadata,json,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from apple_silicon_hybrid import SharedBatch
import mlx.core as mx
out=root/'results/apple_shared_calibration.json';out.parent.mkdir(exist_ok=True)
info=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
 memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
 mlx=importlib.metadata.version('mlx'))
probes={}
for name,fn in [('gpu_qr',lambda:mx.linalg.qr(mx.ones((8,4)),stream=mx.gpu)),
                ('gpu_svd',lambda:mx.linalg.svd(mx.ones((8,4)),stream=mx.gpu)),
                ('gpu_float64',lambda:mx.ones((4,4),dtype=mx.float64)@mx.ones((4,4),dtype=mx.float64))]:
 try:mx.eval(fn());probes[name]='supported'
 except Exception as e:probes[name]=str(e)
rows=[];rng=np.random.default_rng(913)
for batch,n in [(1,16),(32,16),(1,32),(32,32),(1,64),(8,64),(32,64),(1,128),(8,128),(32,128),(1,256),(8,256),(32,256),(1,512),(8,512)]:
 pairs=[(rng.normal(size=(n,n)),rng.normal(size=(n,n))) for _ in range(batch)]
 shared=SharedBatch(batch,n,n,n)
 def copied():
  a=mx.array(np.stack([x for x,y in pairs]).astype(np.float32))
  b=mx.array(np.stack([y for x,y in pairs]).astype(np.float32));c=a@b;mx.eval(c)
  return np.asarray(c,dtype=np.float64)
 fns={'cpu64':lambda:np.array([a@b for a,b in pairs]),'metal_copied':copied,'metal_shared':lambda:shared(pairs)}
 timing={};outputs={}
 for key,fn in fns.items():fn()
 samples={k:[] for k in fns}
 for repeat in range(9):
  for key in list(fns)[repeat%3:]+list(fns)[:repeat%3]:
   start=time.perf_counter();outputs[key]=fns[key]();samples[key].append(time.perf_counter()-start)
 timing={k:statistics.median(v) for k,v in samples.items()}
 reference=outputs['cpu64'];difference=outputs['metal_shared']-reference
 row=dict(batch=batch,m=n,k=n,n=n,seconds=timing,buffer_bytes=shared.buffer_bytes,
  relative_frobenius_error=float(np.linalg.norm(difference)/np.linalg.norm(reference)),
  max_absolute_error=float(np.max(abs(difference))))
 rows.append(row);out.write_text(json.dumps(dict(machine=info,probes=probes,rows=rows),indent=2)+'\n')
 print(json.dumps(row),flush=True)
print(json.dumps(dict(machine=info,probes=probes)),flush=True)
