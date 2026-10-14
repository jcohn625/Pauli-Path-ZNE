"""Split-layout A/B and larger-bond MPS profiles for Apple optimization batch 3."""
import argparse, cProfile, importlib.util, json, statistics, sys, time
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['peps','mps'],required=True)
p.add_argument('--baseline',type=Path,default=Path('/tmp/pauli-apple-batch3-baseline'))
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import cnot_parity_mps as mps
import peps_metal_layer as layer
from tensor_zne import evaluate_mps,evaluate_peps
rows=[]
if a.kind=='peps':
 spec=importlib.util.spec_from_file_location('batch3_old_layer',a.baseline/'peps_metal_layer.py')
 old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
 old.DiagonalGate=layer.DiagonalGate
 from quimb.tensor import Tensor
 from peps_split_layout import split_tensor
 reference_split=Tensor.split
 def candidate_split(self,left_inds=None,**kw):
  if kw.get("get")=="tensors" and "bond_ind" in kw and "absorb" in kw and set(kw)<=set(("right_inds","bond_ind","absorb","cutoff","max_bond","get")):
   return split_tensor(self,left_inds=left_inds,**kw)
  return reference_split(self,left_inds,**kw)
 current=layer.LayerEngine.apply
 def candidate_apply(self,*args,**kw):
  Tensor.split=candidate_split
  try:return current(self,*args,**kw)
  finally:Tensor.split=reference_split
 try:
  for cap in (8,16):
   kw=dict(nx=4,ny=4,steps=2,sites=(5,6),cap=cap,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu',noise_placement='after_cnot',pair_rates=.0001,residual=True)
   samples={False:[],True:[]};values={}
   for rep in range(6):
    for before in ((True,False) if rep%2==0 else (False,True)):
     layer.LayerEngine.apply=old.LayerEngine.apply if before else candidate_apply
     start=time.perf_counter();r=evaluate_peps(**kw);elapsed=time.perf_counter()-start
     if rep:samples[before].append(elapsed)
     values[before]=r['corrected']
   b,n=statistics.median(samples[True]),statistics.median(samples[False])
   row=dict(cap=cap,settings=kw,before_s=b,after_s=n,reduction_percent=100*(1-n/b),corrected_difference=abs(values[True]-values[False]),before_samples=samples[True],after_samples=samples[False])
   rows.append(row);print(json.dumps(row),flush=True)
 finally:Tensor.split=reference_split;layer.LayerEngine.apply=current
else:
 import scipy.linalg as la
 from optimized_pauli_svd import RandomizedSVD
 original_qr=mps.economic_qr;original_svd=la.svd;original_rsvd=RandomizedSVD.__call__
 for cap in (64,128,256):
  counts={'qr':Counter(),'svd':Counter(),'rsvd':Counter()};durations=Counter()
  def trace(kind,fn):
   def call(mat,*args,**kw):
    start=time.perf_counter();result=fn(mat,*args,**kw)
    durations[kind]+=time.perf_counter()-start;counts[kind][mat.shape]+=1;return result
   return call
  def rsvd(self,mat,*args,**kw):
   counts['rsvd'][mat.shape]+=1;return original_rsvd(self,mat,*args,**kw)
  kw=dict(n=16,steps=4,sites=(3,10),cap=cap,guide_cap=cap,noise_placement='after_cnot',pair_rates=.0001,residual=True)
  evaluate_mps(**kw)
  start=time.perf_counter();r=evaluate_mps(**kw);elapsed=time.perf_counter()-start
  mps.economic_qr=trace('qr',original_qr);la.svd=trace('svd',original_svd);RandomizedSVD.__call__=rsvd
  profile=cProfile.Profile()
  try:profile.runcall(evaluate_mps,**kw)
  finally:mps.economic_qr=original_qr;la.svd=original_svd;RandomizedSVD.__call__=original_rsvd
  profile.create_stats()
  row=dict(cap=cap,settings=kw,seconds=elapsed,corrected=r['corrected'],charge_leakage=r['details']['charge_leakage'],stats=r['details']['stats'],blocks={k:[dict(shape=s,count=c) for s,c in v.most_common()] for k,v in counts.items()},instrumented_seconds=dict(durations),profile=[dict(file=Path(k[0]).name,function=k[2],calls=v[1],self_s=v[2],cumulative_s=v[3]) for k,v in sorted(profile.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:30]])
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('blocks','profile')}),flush=True)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
