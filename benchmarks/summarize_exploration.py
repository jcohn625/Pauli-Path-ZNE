"""Archive measurements and render the research comparison figure."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
root=Path(__file__).resolve().parents[1]
source=root/'results/exploration_20261013'
keys=('jacobi','adaptive_real','hybrid_lowrank','polar','estimators','estimators_cap8','row_residual','scheduling')
data={k:json.loads((source/f'{k}.json').read_text()) for k in keys}
files=['apple_jacobi_experiment.py','peps_bethe_experiment.py']+[str(p.relative_to(root)) for p in (root/'benchmarks').glob('explore_*.py')]
# Use the production denominator/mass policy when forming derived quantities.
curves=[]
for dataset in ('estimators','estimators_cap8'):
 rr=data[dataset]
 for steps in sorted({x['steps'] for x in rr}):
  for cap in sorted({x['cap'] for x in rr}):
   group={(x['scale'],x['unsigned']):x for x in rr if x['steps']==steps and x['cap']==cap}
   for scale in sorted({x[0] for x in group}):
    record=dict(dataset=dataset,steps=steps,cap=cap,scale=scale,methods={})
    for mode in ('exact','raw','residual','bethe'):
     def value(scale,unsigned):
      z=group[scale,unsigned];return z['exact'] if mode=='exact' else z['values'][mode]
     o,u,o0,u0=value(scale,False),value(scale,True),value(0.,False),value(0.,True)
     if any(x is None for x in (o,u,o0,u0)):
      record['methods'][mode]=None;continue
     derived={'ratio':o/o0 if abs(o0)>1e-12 else None}
     for name,sign in (('q_plus',1),('q_minus',-1)):
      num=(u+sign*o)/2;den=(u0+sign*o0)/2
      threshold=1e-12*max(abs(u0),abs(o0),np.finfo(float).tiny)
      derived[name]=num/den if den>threshold and num>=0 else None
     record['methods'][mode]=derived
    curves.append(record)
data['derived_curves']=curves
data['provenance']=dict(date='2026-10-13',device='Apple M4 Pro',memory_bytes=25769803776,python='3.12',numpy='1.26.4/OpenBLAS',scipy='1.17.1/Accelerate',mlx='0.31.2',blas_threads=1,numba_threads=1,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},notes=['Warmed medians; GPU times include input conversion, execution and host return.','Final adaptive_real SVD batches use actual captured MPS matrices at every tested dimension, filling absent sizes with additional smaller-cap runs; batches repeat up to eight matrices. Original jacobi cases may use labelled synthetic fallbacks. No claim these matrices are simultaneously available in production.','The final adaptive kernel uses explicit 1e-7 relative Frobenius deflation; null-space orthogonality is not guaranteed.','Exact PEPS references contract the fixed spatially truncated PEPS, not the exact circuit.','The intermediate pre-deflation kernel experiment is retained only in local results/adaptive_jacobi.json.'])
archive=root/'docs/apple-silicon/exploration-20261013.json';archive.write_text(json.dumps(data,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
r=data['adaptive_real'];ns=(16,32,64,128,256);bs=(1,4,16,32);speed=[]
for n in ns:
 vals=[]
 for b in bs:
  z=[x for x in r if x['n']==n and x['batch']==b];cpu=next(x for x in z if x['mode']=='cpu_float64');gpu=min((x for x in z if 'adaptive' in x['mode']),key=lambda x:x['seconds']);vals.append(cpu['seconds']/gpu['seconds'])
 speed.append(vals)
speed=np.array(speed);im=axes[0].imshow(np.log2(speed),cmap='RdBu',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=3),aspect='auto')
for i in range(len(ns)):
 for j in range(len(bs)):axes[0].text(j,i,f'{speed[i,j]:.2f}×',ha='center',va='center',color='white' if abs(np.log2(speed[i,j]))>2 else 'black')
axes[0].set(xticks=range(len(bs)),xticklabels=bs,yticks=range(len(ns)),yticklabels=ns,xlabel='Independent matrices in batch',ylabel='Square sector matrix dimension',title='FP64 CPU time / FP32 GPU time\nJacobi prototype; >1 favors GPU')
z=[x for x in data['estimators_cap8'] if x['steps']==6 and not x['unsigned'] and x['scale']==0]
for method,color in zip(('raw','residual','bethe'),('#808080','#0072B2','#D55E00')):
 axes[1].plot([x['cap'] for x in z],[x['relative_errors'][method] for x in z],'-o',label=method,color=color)
axes[1].set(yscale='log',xlabel='Boundary bond cap',ylabel='Relative scalar error',title='Hard signed PEPS example\n4×4, spatial cap 8, six steps')
axes[1].legend(frameon=False)
z=[x for x in data['row_residual'] if x['cap']==16];ix=np.arange(len(z))
for dx,method,label,color in ((-.18,'bond','Bond overlaps','#808080'),(.18,'row','Row telescoping','#0072B2')):
 axes[2].bar(ix+dx,[1000*x['methods'][method]['seconds'] for x in z],.36,label=label,color=color)
axes[2].set(xticks=ix,xticklabels=[f"{x['side']}×{x['side']}" for x in z],ylabel='Residual contraction time (ms)',title='Same corrected values\nSpatial cap 8, boundary cap 16')
axes[2].legend(frameon=False)
fig.savefig(root/'docs/apple-silicon/exploration-20261013.png',dpi=170)
