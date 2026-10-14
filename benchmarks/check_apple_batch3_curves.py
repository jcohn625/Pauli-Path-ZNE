"""Check promoted BLAS dispatch on signed/unsigned, ratios, Q and residuals."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import mps_contractions as products
from tensor_zne import curves,evaluate_mps
original=products._ACCELERATE
if not original:raise RuntimeError('this validation needs the measured Accelerate/OpenBLAS configuration')
kw=dict(n=12,steps=3,sites=(3,8),cap=64,guide_cap=64,noise_placement='after_cnot',pair_rates=.001,residual=True)
values=[]
try:
 for enabled in (False,True):
  products._ACCELERATE=enabled
  values.append(curves(evaluate_mps,[0,1,3],**kw))
finally:products._ACCELERATE=original
errors={};residual=0.
for old,new in zip(values[0]['rows'],values[1]['rows']):
 for key in old:
  if old[key] is None or new[key] is None:assert old[key] is new[key]
  else:errors[key]=max(errors.get(key,0.),abs(old[key]-new[key]))
for old,new in zip(values[0]['raw'],values[1]['raw']):
 for kind in ('signed','unsigned'):
  residual=max(residual,abs(old[kind]['correction']-new[kind]['correction']))
  assert new[kind]['details']['charge_leakage']==0.
assert max(errors.values())<1e-10
assert residual<1e-10
result=dict(settings=kw,max_differences=errors,max_residual_difference=residual,before=values[0]['rows'],after=values[1]['rows'])
(root/'results/apple_batch3_curves.json').write_text(json.dumps(result,indent=2)+'\n')
print('Maximum derived quantity difference:',max(errors.values()),'residual:',residual)
