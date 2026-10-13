"""Pauli MPS with site-dependent parity maps, canonical updates and residuals."""
import copy
import math
import time
import numpy as np
import scipy.linalg as la
from cnot_parity import Q, Operation, compile_operations, route_operations
from optimized_pauli_svd import RandomizedSVD
from mps_residual import overlap, residual_value


class FrameMPS:
    def __init__(self,tensors,charges,maps=None,backend='cpu'):
        self.tensors=tensors;self.charges=charges
        self.maps=[Q]*len(tensors) if maps is None else list(maps)
        self.log_scale=0.;self.center=None;self.rsvd=RandomizedSVD('lu',backend=backend)
        self.stats=dict(svd_blocks=0,qr_blocks=0,max_bond=max(map(len,charges)),truncations=0)

    def copy(self):
        # Share only deterministic sketch cache; all state data is independent.
        new=copy.copy(self);new.tensors=[x.copy() for x in self.tensors]
        new.charges=[x.copy() for x in self.charges];new.maps=list(self.maps);new.stats=dict(self.stats)
        return new

    @classmethod
    def observable(cls,codes,backend='cpu'):
        charges=[np.array([0])]
        for c in codes:charges.append(charges[-1]^Q[c])
        return cls([np.eye(4)[c].reshape(1,4,1) for c in codes],charges,backend=backend)

    @classmethod
    def singlets(cls,n,unsigned=False,backend='cpu'):
        tensors=[];charges=[np.array([0])]
        for _ in range(n//2):
            tensors.extend([np.eye(4).reshape(1,4,4),np.diag([1,1,1,1] if unsigned else [1,-1,-1,-1]).reshape(4,4,1).astype(float)])
            charges.extend([np.array(Q),np.array([0])])
        return cls(tensors,charges,backend=backend)

    def _right_qr(self,i):
        a=self.tensors[i];ql,qr=self.charges[i:i+2]
        matrix=a.reshape(len(ql),-1);colq=(np.array(self.maps[i])[:,None]^qr[None,:]).ravel()
        pieces=[];newq=[]
        for q in np.unique(ql):
            ri=np.flatnonzero(ql==q);ci=np.flatnonzero(colq==q)
            if not len(ci):continue
            block=matrix[np.ix_(ri,ci)]
            if not np.any(block):continue
            u,r=la.qr(block.T,mode='economic',check_finite=True)
            pieces.append((ri,ci,u.T,r.T,len(newq)));newq.extend([q]*u.shape[1])
            self.stats['qr_blocks']+=1
        new=np.zeros((len(newq),matrix.shape[1]));link=np.zeros((len(ql),len(newq)))
        for ri,ci,u,r,start in pieces:
            ix=np.arange(start,start+len(u));new[np.ix_(ix,ci)]=u;link[np.ix_(ri,ix)]=r
        self.tensors[i]=new.reshape(len(newq),4,len(qr))
        self.tensors[i-1]=np.einsum('lpa,ab->lpb',self.tensors[i-1],link,optimize=True)
        self.charges[i]=np.array(newq,dtype=int)

    def _left_qr(self,i):
        a=self.tensors[i];ql,qr=self.charges[i:i+2]
        matrix=a.reshape(-1,len(qr));rowq=(ql[:,None]^np.array(self.maps[i])[None,:]).ravel()
        pieces=[];newq=[]
        for q in np.unique(qr):
            ri=np.flatnonzero(rowq==q);ci=np.flatnonzero(qr==q)
            if not len(ri):continue
            block=matrix[np.ix_(ri,ci)]
            if not np.any(block):continue
            u,r=la.qr(block,mode='economic',check_finite=True)
            pieces.append((ri,ci,u,r,len(newq)));newq.extend([q]*u.shape[1]);self.stats['qr_blocks']+=1
        new=np.zeros((matrix.shape[0],len(newq)));link=np.zeros((len(newq),len(qr)))
        for ri,ci,u,r,start in pieces:
            ix=np.arange(start,start+u.shape[1]);new[np.ix_(ri,ix)]=u;link[np.ix_(ix,ci)]=r
        self.tensors[i]=new.reshape(len(ql),4,len(newq))
        self.tensors[i+1]=np.einsum('ab,bpr->apr',link,self.tensors[i+1],optimize=True)
        self.charges[i+1]=np.array(newq,dtype=int)

    def move_center(self,i):
        if self.center is None:
            for j in range(len(self.tensors)-1,0,-1):self._right_qr(j)
            self.center=0
        while self.center<i:self._left_qr(self.center);self.center+=1
        while self.center>i:self._right_qr(self.center);self.center-=1

    def rescale(self):
        a=self.tensors[self.center];s=float(np.max(abs(a)))
        if not np.isfinite(s) or s<=0:raise FloatingPointError('zero or nonfinite MPS center')
        self.tensors[self.center]=a/s;self.log_scale+=np.log(s)

    def apply(self,op,cap,callback=None):
        sites=op.sites;matrix=op.matrix;before=op.before;after=op.after
        if len(sites)==2 and sites[0]>sites[1]:
            sites=sites[::-1];before=before[::-1];after=after[::-1]
            matrix=matrix.reshape(4,4,4,4).transpose(1,0,3,2).reshape(16,16)
        if tuple(self.maps[q] for q in sites)!=before:raise ValueError('MPS parity frame mismatch')
        i=sites[0];self.move_center(i)
        if len(sites)==1:
            if op.kind=='single_noise':
                self.tensors[i]=self.tensors[i]*np.diag(matrix)[None,:,None]
            else:
                self.tensors[i]=np.einsum('pq,lqr->lpr',matrix,self.tensors[i],optimize=True)
            self.maps[i]=after[0];self.rescale();return
        if sites[1]!=i+1:raise ValueError('MPS operation must be routed to adjacent sites')
        a,b=self.tensors[i:i+2];ql,qr=self.charges[i],self.charges[i+2]
        theta=np.einsum('lam,mbr->labr',a,b,optimize=True)
        if op.kind=='pair_noise':
            theta=theta*np.diag(matrix).reshape(1,4,4,1)
        elif op.kind in ('cx','routing_swap'):
            # Clifford Pauli transfer is a signed permutation, not a dense GEMM.
            columns=np.argmax(abs(matrix),axis=1)
            theta=(theta.reshape(len(ql),16,len(qr))[:,columns,:]
                   *matrix[np.arange(16),columns][None,:,None]).reshape(len(ql),4,4,len(qr))
        else:
            theta=np.einsum('uvab,labr->luvr',matrix.reshape(4,4,4,4),theta,optimize=True)
        self.maps[i:i+2]=after
        rowq=(ql[:,None]^np.array(after[0])[None,:]).ravel()
        colq=(np.array(after[1])[:,None]^qr[None,:]).ravel()
        m=theta.reshape(len(rowq),len(colq));pieces=[];candidates=[]
        for q in np.intersect1d(rowq,colq):
            ri=np.flatnonzero(rowq==q);ci=np.flatnonzero(colq==q);block=m[np.ix_(ri,ci)]
            if not np.any(block):continue
            rank=min(cap+16,*block.shape)
            if rank>=min(block.shape):u,s,v=la.svd(block,full_matrices=False,check_finite=True)
            else:u,s,v=self.rsvd(block,rank,3)
            idx=len(pieces);pieces.append((q,ri,ci,u,s,v))
            candidates.extend((float(x),idx,j) for j,x in enumerate(s) if x>0)
            self.stats['svd_blocks']+=1
        if not candidates:raise FloatingPointError('zero MPS after gate')
        candidates.sort(key=lambda x:x[0],reverse=True)
        chosen=[x for x in candidates[:cap] if x[0]>1e-13*candidates[0][0]]
        left=np.zeros((len(rowq),len(chosen)));right=np.zeros((len(chosen),len(colq)));newq=[]
        for k,(_,idx,j) in enumerate(chosen):
            q,ri,ci,u,s,v=pieces[idx];left[ri,k]=u[:,j];right[k,ci]=s[j]*v[j];newq.append(q)
        if callback is not None:callback(self,i,m,left,right)
        self.tensors[i]=left.reshape(len(ql),4,len(chosen))
        self.tensors[i+1]=right.reshape(len(chosen),4,len(qr));self.charges[i+1]=np.array(newq)
        self.center=i+1;self.stats['max_bond']=max(self.stats['max_bond'],len(chosen))
        self.stats['truncations']+=1;self.rescale()

    def leakage(self):
        return max(float(np.max(abs(a[(self.charges[i][:,None,None]^np.array(self.maps[i])[None,:,None])!=self.charges[i+1][None,None,:]]),initial=0))
                   for i,a in enumerate(self.tensors))


class Checkpoints:
    """O(sqrt(number of operations)) guide snapshots, not one per gate."""
    def __init__(self,operations,boundary,cap):
        self.ops=operations;self.cap=cap;self.span=max(1,math.isqrt(len(operations)))
        self.ends={};guide=boundary.copy()
        for j in range(len(operations)-1,-1,-1):
            if (j+1)%self.span==0 or j==len(operations)-1:self.ends[j+1]=guide.copy()
            guide.apply(operations[j].adjoint(),cap)
    def chunk(self,start):
        end=min(start+self.span,len(self.ops));guide=self.ends.pop(end)
        result={}
        for j in range(end-1,start-1,-1):
            if len(self.ops[j].sites)==2:result[j]=guide.copy()
            guide.apply(self.ops[j].adjoint(),self.cap)
        return result


def evaluate(*,n,steps,sites,phi,scale,singles,pair_rates=0.,edges=None,cap=16,
             guide_cap=32,residual=True,unsigned=False,backend='cpu'):
    from lattice_circuit import chain
    if not isinstance(n,int) or n<2 or n%2 or any(not isinstance(x,int) or x<1 for x in (cap,guide_cap)):
        raise ValueError('use even N >= 2 and positive bond caps')
    if len(sites)!=2 or len(set(sites))!=2 or any(q<0 or q>=n for q in sites):raise ValueError('two distinct in-range sites required')
    start=time.perf_counter();operations=compile_operations(chain(n),steps,phi,singles=singles,
        pair_rates=pair_rates,edges=edges,scale=scale,unsigned=unsigned)
    operations=route_operations(operations,n,lambda i,j:range(i,j+(1 if i<j else -1),1 if i<j else -1))
    codes=[0]*n
    for q in sites:codes[q]=3
    state=FrameMPS.observable(codes,backend);boundary=FrameMPS.singlets(n,unsigned,backend)
    checkpoints=Checkpoints(operations,boundary,guide_cap) if residual else None
    contributions=[];snapshot_bound=0
    for j,op in enumerate(operations):
        if residual and j%checkpoints.span==0:
            guides=checkpoints.chunk(j);snapshot_bound=max(snapshot_bound,len(guides)+len(checkpoints.ends))
        def record(st,i,m,left,right):
            value=residual_value(st,guides.pop(j),i,m,left,right)
            contributions.append(dict(operation=j,kind=op.kind,location=list(op.location),bond=i,value=value))
        state.apply(op,cap,record if residual and len(op.sites)==2 else None)
    base=overlap(state,boundary);delta=math.fsum(r['value'] for r in contributions)
    corrected=base+delta
    if not all(np.isfinite(x) for x in (base,delta,corrected)):raise FloatingPointError('nonfinite CNOT MPS result')
    return dict(approximate=base,correction=delta,corrected=corrected,details=dict(
        method='cnot-layer-transported-parity-mps',unsigned=unsigned,operations=len(operations),
        routing_swaps=sum(op.kind=='routing_swap' for op in operations),
        guide_snapshot_bound=snapshot_bound,guide_cap=guide_cap if residual else None,
        contributions=contributions,stats=state.stats,charge_leakage=state.leakage(),
        final_standard_frame=state.maps==[Q]*n,runtime_s=time.perf_counter()-start,
        noise_placement='after each nonempty CNOT layer',terminal_strategy='full evolution; no split tail'))
