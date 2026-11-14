"""Time the complete high-level API, including evolution and GPU conversion."""
import argparse,json,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_peps
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
kw=dict(nx=6,ny=6,steps=2,sites=(7,14),cap=16,boundary_cap=32,environment_cap=32,
        noise_placement='after_cnot',pair_rates=.001,scale=1.,gate_backend='batched-cpu')
for backend in ('blas','metal-hybrid'):
 evaluate_peps(**kw,boundary_backend=backend);samples=[];values=[]
 for rep in range(3):
  start=time.perf_counter();r=evaluate_peps(**kw,boundary_backend=backend)
  samples.append(time.perf_counter()-start);values.append(r['corrected'])
 row=dict(backend=backend,config=kw,seconds=statistics.median(samples),samples=samples,corrected=values,
          boundary_gpu_stats=r['details'].get('boundary_gpu_stats'))
 rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
