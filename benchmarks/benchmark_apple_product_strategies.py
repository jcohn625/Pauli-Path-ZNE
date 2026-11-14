"""Replay actual PEPS requests to compare CPU libraries, grouping and packing.

Run with one BLAS thread. Values are captured in RAM for a bounded sample of
requests; result JSON contains timings and errors only. No solver API changes.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import json, statistics, sys, time
import numpy as np
from scipy.linalg.blas import dgemm
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import peps_metal_layer as layer
from tensor_zne import evaluate_peps
original=layer.contract_block_products


@lru_cache(maxsize=128)
def groups(plan):
    result=defaultdict(list)
    for job in plan:result[job[2]].append(job)
    return tuple(tuple(v) for v in result.values())


def execute(lb,rb,plan,lo,ro,mode):
    if mode=='matmul':return original(lb,rb,plan,lo,ro)
    if mode in ('einsum','accelerate'):
        out={}
        for i,j,key,shape,(m,k,n) in plan:
            a=lb[i][1].transpose(lo).reshape(m,k)
            b=rb[j][1].transpose(ro).reshape(k,n)
            c=(np.einsum('ij,jk->ik',a,b,optimize=False) if mode=='einsum'
               else dgemm(1.,b.T,a.T).T).reshape(shape)
            if key in out:out[key]+=c
            else:out[key]=c
        return out
    out={};left={};right={}
    for jobs in groups(plan):
        pairs=[]
        for i,j,key,shape,(m,k,n) in jobs:
            if mode in ('reuse','grouped'):
                if i not in left:left[i]=lb[i][1].transpose(lo).reshape(m,k)
                if j not in right:right[j]=rb[j][1].transpose(ro).reshape(k,n)
                a,b=left[i],right[j]
            else:
                a=lb[i][1].transpose(lo).reshape(m,k)
                b=rb[j][1].transpose(ro).reshape(k,n)
            pairs.append((a,b))
        if mode=='grouped' and len(pairs)>1:
            c=np.concatenate([a for a,b in pairs],axis=1) @ np.concatenate([b for a,b in pairs],axis=0)
        else:
            c=None
            for a,b in pairs:
                if mode=='einsum':v=np.einsum('ij,jk->ik',a,b,optimize=False)
                elif mode=='accelerate':v=dgemm(1.,b.T,a.T).T
                else:v=a @ b
                if c is None:c=v
                else:c+=v
        out[key]=c.reshape(shape)
    return out


if __name__=='__main__':
    original=layer.contract_block_products
    rows=[]
    for cap in (4,8,16):
        captured=[];seen=0;used=0
        def capture(lb,rb,plan,lo,ro):
            global seen,used
            seen+=1;size=sum(a.nbytes for _,a in lb)+sum(a.nbytes for _,a in rb)
            # Sample throughout evolution rather than only early small bonds.
            if seen%7==0 and used+size<=64*1024**2:
                captured.append(([(q,a.copy()) for q,a in lb],[(q,a.copy()) for q,a in rb],plan,lo,ro));used+=size
            return original(lb,rb,plan,lo,ro)
        layer.contract_block_products=capture
        evaluate_peps(nx=4,ny=4,steps=2,sites=(5,6),cap=cap,boundary_cap=8,
                      environment_cap=16,gate_backend='batched-cpu',noise_placement='after_cnot',pair_rates=.0001)
        layer.contract_block_products=original
        modes=['matmul','einsum','accelerate','reuse','grouped'];samples={m:[] for m in modes};errors={}
        for mode in modes:
            errors[mode]=0.
            for args in captured:
                ref=original(*args);actual=execute(*args,mode)
                for key in ref:errors[mode]=max(errors[mode],float(np.max(abs(ref[key]-actual[key]),initial=0)))
        for rep in range(7):
            for mode in modes[rep%len(modes):]+modes[:rep%len(modes)]:
                start=time.perf_counter()
                for args in captured:execute(*args,mode)
                samples[mode].append(time.perf_counter()-start)
        row=dict(cap=cap,requests=len(captured),captured_bytes=used,
                 products=sum(len(a[2]) for a in captured),
                 output_groups=sum(len(groups(a[2])) for a in captured),
                 repeated_output_groups=sum(len(g)>1 for a in captured for g in groups(a[2])),
                 seconds={m:statistics.median(v) for m,v in samples.items()},max_absolute_error=errors)
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'results/apple_product_strategies.json').write_text(json.dumps(rows,indent=2)+'\n')
