"""Experimental exact diagonal noise MPO, with an explicit frontier-size guard.

A bond carries Pauli labels of earlier vertices with edges to later vertices.
Its size is 4**frontier_width. MPO auxiliary charges are neutral because the
physical Pauli label is unchanged. Never form a full many-qubit PTM.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from cnot_parity import Q,Operation


@dataclass(frozen=True)
class NoiseMPO:
    sites: tuple
    tensors: tuple
    before: tuple
    after: tuple
    location: tuple
    kind: str = 'noise_mpo'
    def adjoint(self):return self
    @property
    def max_bond(self):return max(max(x.shape[0],x.shape[2]) for x in self.tensors)
    @property
    def nbytes(self):return sum(x.nbytes for x in self.tensors)


def build_noise_mpo(group,n,frames,max_bond=64):
    if not isinstance(max_bond,int) or max_bond<1:raise ValueError('noise_mpo_max_bond must be a positive integer')
    singles=np.ones((n,4));pairs={}
    for op in group:
        if op.before!=op.after or op.kind not in ('single_noise','pair_noise'):raise ValueError('MPO needs one diagonal noise insertion')
        diagonal=getattr(op,'diagonal',None)
        if diagonal is None:diagonal=np.diag(op.matrix)
        if len(op.sites)==1:singles[op.sites[0]]*=diagonal
        else:
            i,j=op.sites;table=diagonal.reshape(4,4)
            if i>j:i,j=j,i;table=table.T
            pairs[i,j]=pairs.get((i,j),np.ones((4,4)))*table
    last={i:max(j for a,j in pairs if a==i) for i in {a for a,j in pairs}}
    frontiers=[tuple(i for i in sorted(last) if i<q and last[i]>=q) for q in range(n+1)]
    width=max(map(len,frontiers))
    if 4**width>max_bond:
        raise ValueError(f'noise MPO needs bond {4**width}, exceeding noise_mpo_max_bond={max_bond}; use local routing or raise the limit')
    tensors=[]
    for q in range(n):
        left,right=frontiers[q:q+2]
        tensor=np.zeros((4**len(left),4,4**len(right)))
        for a,labels in enumerate(product(range(4),repeat=len(left))):
            values=dict(zip(left,labels))
            for p in range(4):
                values[q]=p;weight=singles[q,p]
                for i in left:
                    if (i,q) in pairs:weight*=pairs[i,q][values[i],p]
                b=0
                for i in right:b=4*b+values[i]
                tensor[a,p,b]=weight
        tensor.flags.writeable=False;tensors.append(tensor)
    return NoiseMPO(tuple(range(n)),tuple(tensors),tuple(frames),tuple(frames),group[0].location)


def group_noise_mpos(operations,n,max_bond=64):
    result=[];frames=[Q]*n;j=0
    while j<len(operations):
        op=operations[j]
        if op.kind in ('single_noise','pair_noise'):
            end=j+1
            while end<len(operations) and operations[end].kind in ('single_noise','pair_noise') and operations[end].location==op.location:end+=1
            group=operations[j:end]
            if any(x.kind=='pair_noise' for x in group):result.append(build_noise_mpo(group,n,frames,max_bond))
            else:result.extend(group)
            j=end;continue
        result.append(op)
        for q,frame in zip(op.sites,op.after):frames[q]=frame
        j+=1
    return result


def apply_noise_mpo(state,op,cap,callback):
    if tuple(state.maps)!=op.before:raise ValueError('noise MPO parity frame mismatch')
    for q,(a,w) in enumerate(zip(state.tensors,op.tensors)):
        value=np.einsum('lpr,xpy->lxpry',a,w,optimize=True)
        state.tensors[q]=value.reshape(a.shape[0]*w.shape[0],4,a.shape[2]*w.shape[2])
    for q in range(len(state.charges)):
        dim=op.tensors[q].shape[0] if q<len(op.tensors) else 1
        state.charges[q]=np.repeat(state.charges[q],dim)
    state.center=None
    # Each compression is an identity transfer after the exact MPO. The same
    # downstream guide contracts every discarded residual in this sweep.
    for q in range(len(state.tensors)-1):
        frame=tuple(state.maps[q:q+2])
        state.apply(Operation((q,q+1),np.eye(16),frame,frame,'mpo_compress',op.location),cap,callback)
