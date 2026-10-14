"""Compare signed, ratio and elementary-Q convergence to independent dense evolution."""
import argparse, importlib.util, json, statistics, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--resume',action='store_true');p.add_argument('--module-dir',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
if a.module_dir:sys.path.insert(0,str(a.module_dir.resolve()))
from tensor_zne import evaluate_mps,evaluate_peps
from lattice_circuit import chain,square
spec=importlib.util.spec_from_file_location('dense_checks',root/'tests/test_cnot_parity.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
previous=json.loads(a.output.read_text()) if a.resume and a.output.exists() else {}
rows=previous.get('rows',[]);references=previous.get('references',[])
completed={(r['engine'],r['pair_rate'],r['scale'],r['unsigned'],r['cap']) for r in rows}
def save():
 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(dict(rows=rows,references=references),indent=2)+'\n')
for engine,fn,lat,settings,caps in [
 ('MPS',evaluate_mps,chain(6),dict(n=6,steps=2,sites=(0,1),guide_cap=32),[2,4,8,16,32]),
 ('PEPS',evaluate_peps,square(2,3),dict(nx=2,ny=3,steps=1,sites=(0,1),boundary_cap=8,environment_cap=32),[2,4,8,16,32])]:
 for pair_rate in (0.,.0005):
  common=dict(settings,phi=.23,lam=(.003,.006,.009),pair_rates=pair_rate,noise_placement='after_cnot')
  fn(**common,cap=2)
  for scale in (0.,1.,3.):
   for unsigned in (False,True):
    exact=checks.dense_reference(lat,settings['steps'],.23,(0,1),common['lam'],pair_rate,None,scale,unsigned)
    if not any(r['engine']==engine and r['pair_rate']==pair_rate and r['scale']==scale and r['unsigned']==unsigned for r in references):
     references.append(dict(engine=engine,pair_rate=pair_rate,scale=scale,unsigned=unsigned,value=exact))
    for cap in caps:
     if (engine,pair_rate,scale,unsigned,cap) in completed:continue
     times=[]
     for repeat in range(2):
      start=time.perf_counter();r=fn(**common,cap=cap,scale=scale,unsigned=unsigned);times.append(time.perf_counter()-start)
     rows.append(dict(engine=engine,pair_rate=pair_rate,scale=scale,unsigned=unsigned,cap=cap,
       seconds=times,median_s=statistics.median(times),exact=exact,approximate=r['approximate'],corrected=r['corrected'],
       base_error=abs(r['approximate']-exact),corrected_error=abs(r['corrected']-exact)))
     save()
   print(engine,pair_rate,scale,'done',flush=True)
