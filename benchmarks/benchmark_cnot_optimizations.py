"""Warmed CPU medians; optional --module-dir selects an older implementation.

Example: OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
  .venv312/bin/python benchmarks/benchmark_cnot_optimizations.py --output results/cnot_cost.json
Bond caps match, but changes to truncation locations can change finite-cap values.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--module-dir',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if args.module_dir:sys.path.insert(0,str(args.module_dir.resolve()))
from tensor_zne import evaluate_mps,evaluate_peps
rows=[]
for engine,fn,settings in [
    ('MPS',evaluate_mps,dict(n=12,steps=3,sites=(3,8),cap=16,guide_cap=16)),
    ('PEPS',evaluate_peps,dict(nx=4,ny=4,steps=1,sites=(5,10),cap=4,boundary_cap=4,environment_cap=8)),
    ('PEPS-batched',evaluate_peps,dict(nx=4,ny=4,steps=1,sites=(5,10),cap=4,boundary_cap=4,environment_cap=8,gate_backend='batched-cpu'))]:
    for pairs in (0.,.0001):
        kw=dict(settings,noise_placement='after_cnot',pair_rates=pairs,residual=True)
        fn(**kw)
        times=[]
        for _ in range(5):
            start=time.perf_counter();result=fn(**kw);times.append(time.perf_counter()-start)
        row=dict(engine=engine,settings=kw,median_s=statistics.median(times),seconds=times,
                 approximate=result['approximate'],corrected=result['corrected'])
        rows.append(row);print(json.dumps(row),flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(rows,indent=2)+'\n')
