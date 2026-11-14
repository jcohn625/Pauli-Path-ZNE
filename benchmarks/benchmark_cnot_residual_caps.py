"""Separate residual-guide convergence from evolution/boundary bond caps."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_mps,evaluate_peps
refs=json.loads((root/'results/cnot_convergence_after.json').read_text())['references'];rows=[]
common=dict(sites=(0,1),phi=.23,lam=(.003,.006,.009),scale=3.,pair_rates=.0005,noise_placement='after_cnot')
for unsigned in (False,True):
 for engine in ('MPS','PEPS'):
  exact=next(r['value'] for r in refs if r['engine']==engine and r['pair_rate']==.0005 and r['scale']==3. and r['unsigned']==unsigned)
  if engine=='MPS':
   configs=[dict(n=6,steps=2,cap=2,guide_cap=c) for c in (4,8,16,32,64)];fn=evaluate_mps
  else:
   configs=[dict(nx=2,ny=3,steps=1,cap=32,boundary_cap=b,environment_cap=e) for b in (1,2,4,8) for e in (2,8,32)];fn=evaluate_peps
  for config in configs:
   r=fn(**common,**config,unsigned=unsigned)
   rows.append(dict(engine=engine,unsigned=unsigned,settings=config,exact=exact,
     approximate=r['approximate'],corrected=r['corrected'],base_error=abs(r['approximate']-exact),
     corrected_error=abs(r['corrected']-exact)))
(root/'results/cnot_residual_caps.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 if r['engine']=='MPS' or r['settings']['boundary_cap']==1:print(r,flush=True)
