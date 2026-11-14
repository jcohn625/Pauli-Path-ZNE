"""Alternate pre-pass and current contraction methods in one warmed process.

The baseline must differ only in the methods switched below: scheduling,
compression, guide policies and public evaluation APIs must be identical.
"""
import argparse, importlib.util, json, statistics, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
import cnot_parity_mps as mps
import peps_metal_layer as peps
from tensor_zne import evaluate_mps,evaluate_peps


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


old_mps=load('apple_before_mps',args.baseline/'cnot_parity_mps.py')
old_residual=load('apple_before_residual',args.baseline/'mps_residual.py')
old_peps=load('apple_before_peps',args.baseline/'peps_metal_layer.py')
targets=[(mps.FrameMPS,old_mps.FrameMPS,name) for name in ('apply','_left_qr','_right_qr')]
targets += [(mps,old_residual,name) for name in ('overlap','residual_value')]
targets += [(peps.LayerEngine,old_peps.LayerEngine,name) for name in ('scale_many','contract_direct','contract_many')]
current=[getattr(obj,name) for obj,_,name in targets]


def select(before):
    for (obj,old,name),new in zip(targets,current):setattr(obj,name,getattr(old,name) if before else new)


rows=[]
try:
    for name,fn,kw in [
        ('MPS16',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16)),
        ('MPS64',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=64,guide_cap=64)),
        ('PEPS8',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=8,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu')),
        ('PEPS16',evaluate_peps,dict(nx=4,ny=4,steps=2,sites=(5,6),cap=16,boundary_cap=8,environment_cap=16,gate_backend='batched-cpu'))]:
        settings=dict(kw,noise_placement='after_cnot',pair_rates=.0001,residual=True)
        samples={False:[],True:[]};values={}
        for rep in range(6):
            for before in ((True,False) if rep%2==0 else (False,True)):
                select(before);start=time.perf_counter();r=fn(**settings);elapsed=time.perf_counter()-start
                if rep:samples[before].append(elapsed)
                values[before]=r['corrected']
        b,a=statistics.median(samples[True]),statistics.median(samples[False])
        row=dict(name=name,settings=settings,before_s=b,after_s=a,speedup=b/a,
                 reduction_percent=100*(1-a/b),corrected_difference=abs(values[True]-values[False]),
                 before_samples=samples[True],after_samples=samples[False])
        rows.append(row);print(json.dumps(row),flush=True)
finally:select(False)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(rows,indent=2)+'\n')
