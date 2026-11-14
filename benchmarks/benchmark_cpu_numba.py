"""Alternating, warmed CPU kernel/workflow comparisons (no concurrent workloads)."""
import argparse,contextlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import cpu_numba_kernels as kernels
from tensor_zne import evaluate_mps,evaluate_peps
p=argparse.ArgumentParser();p.add_argument('--kind',choices=['micro','mps','peps'],required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
p.add_argument('--caps',type=int,nargs='+',default=[16,64,128]);p.add_argument('--combined-only',action='store_true')
p.add_argument('--baseline',type=Path,default=None)
p.add_argument('--gate-backend',choices=['quimb','batched-cpu'],default='batched-cpu')
p.add_argument('--variants',nargs='+',choices=['reference','disabled','gather','select','scatter','transfer','all','policy'])
a=p.parse_args();rows=[];policy=kernels.ENABLED.copy()
def save(row):
 rows.append(row);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
def load(name):
 spec=importlib.util.spec_from_file_location('reference_'+name,a.baseline/(name+'.py'))
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def paired(label,functions,number=1,config=None):
 samples={k:[] for k in functions};values={};cold={}
 for k,fn in functions.items():
  start=time.perf_counter();values[k]=fn();cold[k]=time.perf_counter()-start
 for rep in range(a.repeats):
  order=list(functions) if rep%2==0 else list(reversed(functions))
  for k in order:
   start=time.perf_counter()
   for _ in range(number):value=functions[k]()
   samples[k].append((time.perf_counter()-start)/number);values[k]=value
 med={k:statistics.median(v) for k,v in samples.items()}
 save(dict(label=label,config=config,seconds=med,samples=samples,first_call_seconds=cold,
           speedup={k:med['reference']/v for k,v in med.items()},values=values if number==1 else None))

if a.kind=='micro':
 rng=np.random.default_rng(172)
 for size in [4,16,64,256]:
  m=rng.normal(size=(size*4,size*4));ri=np.arange(size)*4;ci=np.arange(size)*4
  paired('gather_'+str(size),{'reference':lambda:m[ri[:,None],ci], 'numba':lambda:kernels.gather(m,ri,ci)},number=100)
  t=rng.normal(size=(size,16,size));cols=rng.permutation(16);weights=rng.normal(size=16)
  paired('transfer_'+str(size),{'reference':lambda:t[:,cols,:]*weights[None,:,None], 'numba':lambda:kernels.transfer(t,cols,weights)},number=100)
  spectra=[np.sort(rng.uniform(0,1,size))[::-1] for _ in range(4)]
  offsets=np.array([0,*np.cumsum([len(s) for s in spectra])]);values=np.concatenate(spectra)
  def select():
   c=sorted([(float(v),p,j) for p,s in enumerate(spectra) for j,v in enumerate(s) if v>0],key=lambda x:x[0],reverse=True)
   return [x for x in c[:size] if x[0]>1e-13*c[0][0]]
  paired('select_'+str(size),{'reference':select,'numba':lambda:kernels.select_sorted(np.concatenate(spectra),np.array([0,*np.cumsum([len(s) for s in spectra])]),size)},number=100)
  selected=np.array([[k%4,k//4] for k in range(size)],dtype=np.int64)
  pieces=[(np.arange(size)*4+p,np.arange(size)*4+p,
           np.asfortranarray(rng.normal(size=(size,size))),np.sort(rng.uniform(size=size))[::-1],
           np.asfortranarray(rng.normal(size=(size,size)))) for p in range(4)]
  def scatter(compiled):
   left=np.zeros((4*size,size));right=np.zeros((size,4*size));charges=np.empty(size,dtype=np.int64)
   if compiled:
    for p,(ri,ci,u,s,v) in enumerate(pieces):kernels.scatter_svd(left,right,charges,selected,p,p,ri,ci,u,s,v)
   else:
    for k,(p,j) in enumerate(selected):
     ri,ci,u,s,v=pieces[p];left[ri,k]=u[:,j];right[k,ci]=s[j]*v[j];charges[k]=p
   return left,right,charges
  paired('scatter_'+str(size),{'reference':lambda:scatter(False),'numba':lambda:scatter(True)},number=100)
 for shape in [(2,2,2,2,1),(4,4,4,4,1),(8,8,8,8,1),(16,16,16,16,1)]:
  block=rng.normal(size=shape);weights=tuple(rng.normal(size=d) for d in shape);axes=(3,2,0,1,4)
  def scale():
   out=block.copy()
   for i,w in enumerate(weights):
    shape=[1]*5;shape[i]=-1;out*=w.reshape(shape)
   return np.ascontiguousarray(out.transpose(axes))
  paired('scale_'+str(block.size),{'reference':scale,'numba':lambda:kernels.scale_pack(block,weights,axes)},number=100)
else:
 import cnot_parity_mps as mps
 from peps_metal_layer import LayerEngine
 original_class=mps.FrameMPS;original_scale=LayerEngine.scale_many
 baseline=load('cnot_parity_mps' if a.kind=='mps' else 'peps_metal_layer') if a.baseline else None
 configurations=([dict(n=16,steps=4,sites=(3,10),cap=cap,guide_cap=cap) for cap in a.caps] if a.kind=='mps'
                 else [dict(nx=4,ny=4,steps=2,sites=(5,10),cap=8,boundary_cap=16,environment_cap=16),
                       dict(nx=6,ny=6,steps=2,sites=(7,14),cap=16,boundary_cap=32,environment_cap=32)])
 for config in configurations:
  kw=dict(**config,noise_placement='after_cnot',pair_rates=.001,residual=True)
  if a.kind=='peps':kw['gate_backend']=a.gate_backend
  variants=a.variants or (['reference','policy'] if a.combined_only else (['reference','all'] if a.kind=='peps' else ['reference','disabled','gather','select','scatter','transfer','all']))
  def run(variant):
   for name in kernels.ENABLED:kernels.ENABLED[name]=policy[name] if variant=='policy' else (variant=='all' or variant==name)
   mps.FrameMPS=baseline.FrameMPS if baseline and a.kind=='mps' and variant=='reference' else original_class
   LayerEngine.scale_many=baseline.LayerEngine.scale_many if baseline and a.kind=='peps' and variant=='reference' else original_scale
   r=(evaluate_mps if a.kind=='mps' else evaluate_peps)(**kw)
   return dict(corrected=r['corrected'],approximate=r['approximate'],correction=r['correction'],
               leakage=r['details'].get('charge_leakage'),stage_stats=r['details'].get('cnot_evolution_stats'))
  try:paired(a.kind+'_'+str(config['cap']),{v:lambda v=v:run(v) for v in variants},config=kw)
  finally:mps.FrameMPS=original_class;LayerEngine.scale_many=original_scale
