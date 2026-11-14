"""Compare larger CNOT-noise PEPS runs with CPU and Metal gate/boundary paths."""
import json, statistics, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_peps
rows=[];output=root/'results/cnot_metal_comparison.json'
for nx,ny,steps in [(4,4,2),(6,6,1)]:
 reference=None
 for gate,boundary in [('batched-cpu','blas'),('metal','blas'),('metal','metal-jacobi')]:
  kw=dict(nx=nx,ny=ny,steps=steps,sites=(nx+1,nx+2),cap=8,boundary_cap=8,
    environment_cap=16,noise_placement='after_cnot',pair_rates=.0001,
    gate_backend=gate,boundary_backend=boundary)
  evaluate_peps(**kw);times=[]
  for _ in range(3):
   start=time.perf_counter();r=evaluate_peps(**kw);times.append(time.perf_counter()-start)
  if reference is None:reference=r['corrected']
  row=dict(settings=kw,seconds=times,median_s=statistics.median(times),value=r['corrected'],
    absolute_difference_from_cpu=abs(r['corrected']-reference))
  rows.append(row);output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
