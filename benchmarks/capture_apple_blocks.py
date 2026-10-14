"""Record actual charge-block sizes; no tensor data is persisted."""
from collections import Counter
import json,sys
from pathlib import Path
import scipy.linalg as la
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_mps,evaluate_peps
import peps_metal_layer as layer
qr=Counter();svd=Counter();groups=Counter();active=[]
original_qr=la.qr;original_svd=la.svd;original_layout=layer.contraction_layout;original_direct=layer.LayerEngine.contract_direct
def trace_qr(a,*args,**kw):qr[a.shape]+=1;return original_qr(a,*args,**kw)
def trace_svd(a,*args,**kw):svd[a.shape]+=1;return original_svd(a,*args,**kw)
def trace_layout(*args):
 plan=original_layout(*args)
 if active:
  for job in plan:active[-1][job[-1]]+=1
 return plan
def trace_direct(self,requests):
 active.append(Counter())
 try:return original_direct(self,requests)
 finally:
  for dims,count in active.pop().items():groups[(count,*dims)]+=1
la.qr=trace_qr;la.svd=trace_svd;layer.contraction_layout=trace_layout;layer.LayerEngine.contract_direct=trace_direct
rows=[]
for name,fn,kw in [('MPS12',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16)),
                  ('PEPS4x4-cap8',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=8,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu')),
                  ('PEPS4x4-cap16',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=16,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu'))]:
 qr.clear();svd.clear();groups.clear();fn(**kw,noise_placement='after_cnot',pair_rates=.0001)
 row=dict(name=name,settings=kw,qr=[dict(shape=k,count=v) for k,v in qr.most_common()],
          svd=[dict(shape=k,count=v) for k,v in svd.most_common()],
          gemm_groups=[dict(shape=k,count=v) for k,v in groups.most_common()])
 rows.append(row);print(name,'top QR',qr.most_common(4),'top GEMM',groups.most_common(6),flush=True)
(root/'results/apple_block_shapes.json').write_text(json.dumps(rows,indent=2)+'\n')
