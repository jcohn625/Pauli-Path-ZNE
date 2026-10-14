"""Replay actual PEPS sector batches; compare QR panels and selective LAPACK."""
import argparse,json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from peps_gpu_decomp import GPUDecompositions
from peps_hybrid_decomp import HybridDecompositions
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
capture=np.load(a.input);rows=[]
# Largest cores and two mid-sized SVDs from the real boundary trajectory.
qrkeys=sorted([k for k in capture.files if k.startswith('qr_')],key=lambda k:np.prod(capture[k].shape),reverse=True)[:3]
svdkeys=sorted([k for k in capture.files if k.startswith('svd_')],key=lambda k:np.prod(capture[k].shape),reverse=True)[:3]
for key in qrkeys+svdkeys:
 method=key.split('_')[0];x=capture[key].astype('float32');batch,m,n=x.shape;k=min(m,n)
 configs=[('metal_panel'+str(w),GPUDecompositions(sweeps=32,qr_panel_width=w)) for w in (16,32,64,128)] if method=='qr' else [('metal',GPUDecompositions(sweeps=32))]
 configs+=[('cpu_lapack',HybridDecompositions(cpu_qr_min=1,cpu_svd_min=1))]
 for name,engine in configs:
  mx=engine.mx
  # Inputs start on-device. Materialization and returned GPU factors are timed.
  dev=mx.array(x);mx.eval(dev)
  def run():
   result=getattr(engine,method)(dev);mx.eval(result);return result
  run();samples=[]
  for rep in range(3):
   tick=time.perf_counter();result=run();samples.append(time.perf_counter()-tick)
  factors=[np.array(z).astype(float) for z in result]
  if method=='qr':u,r=factors;reconstructed=u@r;vorth=None
  else:u,s,v=factors;reconstructed=(u*s[:,None,:])@v;vorth=float(np.linalg.norm(v@v.transpose(0,2,1)-np.eye(k)))
  row=dict(key=key,method=method,mode=name,shape=list(x.shape),seconds=statistics.median(samples),samples=samples,
    reconstruction=float(np.linalg.norm(reconstructed-x)/max(np.linalg.norm(x),1e-30)),
    full_left_orthogonality=float(np.linalg.norm(u.transpose(0,2,1)@u-np.eye(k))),right_orthogonality=vorth)
  rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
