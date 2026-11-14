"""Alternating whole-MPS benchmark of NumPy versus Accelerate bond products."""
import argparse,importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg.blas import dgemm
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import cnot_parity_mps as mps
import mps_contractions as products
saved_accelerate=products._ACCELERATE
from tensor_zne import evaluate_mps
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,default=Path('/tmp/pauli-apple-batch3-baseline'));p.add_argument('--policy',action='store_true');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
def product(a,b):return dgemm(1.,b.T,a.T).T

def left(link,tensor):
 return product(link,tensor.reshape(tensor.shape[0],-1)).reshape(link.shape[0],tensor.shape[1],tensor.shape[2])
def right(tensor,link):
 return product(tensor.reshape(-1,tensor.shape[2]),link).reshape(tensor.shape[0],tensor.shape[1],link.shape[1])
def join(left,right):
 return product(left.reshape(-1,left.shape[2]),right.reshape(right.shape[0],-1)).reshape(left.shape[0],left.shape[1],right.shape[1],right.shape[2])
names=['absorb_left','absorb_right','join_sites'];current=[getattr(mps,n) for n in names]
spec=importlib.util.spec_from_file_location('batch3_old_contractions',a.baseline/'mps_contractions.py');baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
old=[getattr(baseline,n) for n in names];new=current if a.policy else [left,right,join];rows=[]
try:
 for cap in (16,64,128,256):
  kw=dict(n=16,steps=4,sites=(3,10),cap=cap,guide_cap=cap,noise_placement='after_cnot',pair_rates=.0001,residual=True)
  samples={False:[],True:[]};values={}
  for rep in range(a.repeats+1):
   for accel in ((False,True) if rep%2==0 else (True,False)):
    if a.policy:products._ACCELERATE=saved_accelerate and accel
    for name,fn in zip(names,new if accel else old):setattr(mps,name,fn)
    start=time.perf_counter();r=evaluate_mps(**kw);elapsed=time.perf_counter()-start
    if rep:samples[accel].append(elapsed)
    values[accel]=r['corrected']
  b,n=statistics.median(samples[False]),statistics.median(samples[True])
  row=dict(cap=cap,settings=kw,policy=a.policy,repeats=a.repeats,before_s=b,after_s=n,reduction_percent=100*(1-n/b),corrected_difference=abs(values[True]-values[False]),samples={str(k):v for k,v in samples.items()})
  rows.append(row);print(json.dumps(row),flush=True)
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
finally:
 products._ACCELERATE=saved_accelerate
 for name,fn in zip(names,current):setattr(mps,name,fn)
