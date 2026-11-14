"""Blocked Gram-core rotations vs CPU and adaptive one-sided Metal Jacobi."""
import argparse,json,statistics,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from apple_blocked_jacobi_experiment import BlockedGramJacobi
from apple_jacobi_experiment import AdaptiveJacobi
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--refine',action='store_true')
p.add_argument('--sizes',nargs='+',type=int,default=[64,128,256]);p.add_argument('--batches',nargs='+',type=int,default=[1,16]);a=p.parse_args()
source=root/'results/exploration_20261013/adaptive_real.npz'
captured=np.load(source);rows=[];rng=np.random.default_rng(52)
a.output.parent.mkdir(parents=True,exist_ok=True)
for n in a.sizes:
 for batch in a.batches:
  for case in ('captured','condition100'):
   if case=='captured':x=np.stack([captured[str(n)][i%len(captured[str(n)])] for i in range(batch)])
   else:
    q,_=la.qr(rng.normal(size=(n,n)));r,_=la.qr(rng.normal(size=(n,n)))
    x=np.stack([q@np.diag(np.geomspace(1,.01,n))@r.T]*batch)
   reference=[la.svd(z,full_matrices=False,check_finite=False)[1] for z in x]
   configs=[('cpu64',None),('cpu32',None),('adaptive',AdaptiveJacobi(sweeps=24,threads=256,deflation_tol=1e-7))]
   configs += [(f'blocked_b{b}_s{s}',BlockedGramJacobi(block=b,sweeps=s)) for b in (16,32) for s in (2,4)]
   if a.refine:
    configs=configs[:3]+[(f'blocked_b32_s{s}',BlockedGramJacobi(block=32,sweeps=s)) for s in (8,16)]
    configs += [('blocked_b32_s2_polish',BlockedGramJacobi(block=32,sweeps=2,polish=True))]
   for name,engine in configs:
    def run():
     if engine is None:
      out=[la.svd(z.astype(np.float64 if name=='cpu64' else np.float32),full_matrices=False,check_finite=False) for z in x]
      return [np.stack([o[k] for o in out]).astype(np.float64) for k in range(3)]
     mx=engine.mx;dev=mx.array(x,dtype=mx.float32)
     scale=mx.maximum(mx.max(mx.abs(dev),axis=(-2,-1),keepdims=True),1e-30)
     u,s,v=engine.jacobi(dev/scale);out=(u,s*scale[...,0],v);mx.eval(out)
     return [np.asarray(o).astype(np.float64) for o in out]
    run();samples=[]
    for rep in range(3):
     tick=time.perf_counter();u,s,v=run();samples.append(time.perf_counter()-tick)
    errors=[];spectra=[];orth=[];vorth=[]
    for j in range(batch):
     errors.append(la.norm((u[j]*s[j])@v[j]-x[j])/max(la.norm(x[j]),1e-30))
     spectra.append(la.norm(s[j]-reference[j])/max(la.norm(reference[j]),1e-30))
     active=s[j]>1e-5*max(s[j,0],1e-30)
     orth.append(la.norm(u[j][:,active].T@u[j][:,active]-np.eye(sum(active))))
     vorth.append(la.norm(v[j]@v[j].T-np.eye(n)))
    row=dict(n=n,batch=batch,case=case,mode=name,seconds=statistics.median(samples),samples=samples,
             reconstruction=max(errors),singular_values=max(spectra),active_u_orthogonality=max(orth),v_orthogonality=max(vorth))
    rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
