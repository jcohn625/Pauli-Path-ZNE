"""Whole-solver timing/value/profile comparison against a saved module tree."""
import argparse, cProfile, hashlib, json, resource, statistics, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--module-dir',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--repeats',type=int,default=5)
a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
source=(a.module_dir or root).resolve();sys.path.insert(0,str(source))
from tensor_zne import evaluate_mps,evaluate_peps
cases=[
 ('MPS16-single',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16,pair_rates=0.)),
 ('MPS16-pair',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16)),
 ('MPS64-pair',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=64,guide_cap=64)),
 ('MPS16-unsigned',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16,unsigned=True)),
 ('PEPS4',evaluate_peps,dict(nx=4,ny=4,steps=1,sites=(5,10),cap=4,boundary_cap=4,environment_cap=8)),
 ('PEPS8',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=8,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu')),
 ('PEPS16',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=16,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu'))]
rows=[]
for name,fn,kw in cases:
 settings=dict(noise_placement='after_cnot',pair_rates=.0001,residual=True);settings.update(kw)
 fn(**settings);times=[]
 for _ in range(a.repeats):
  start=time.perf_counter();r=fn(**settings);times.append(time.perf_counter()-start)
 row=dict(name=name,settings=settings,seconds=times,median_s=statistics.median(times),
          approximate=r['approximate'],corrected=r['corrected'],
          process_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 if name in ('MPS16-pair','PEPS8'):
  profile=cProfile.Profile();profile.runcall(fn,**settings)
  profile.create_stats()
  row['profile']=[dict(file=Path(k[0]).name,line=k[1],function=k[2],calls=v[1],self_s=v[2],cumulative_s=v[3])
                  for k,v in sorted(profile.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:40]]
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='profile'}),flush=True)
files=['cnot_parity_mps.py','mps_residual.py','mps_contractions.py','mps_qr.py','peps_metal_layer.py']
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(source=str(source),hashes={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files if (source/f).exists()},rows=rows),indent=2)+'\n')
