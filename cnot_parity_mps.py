"""Pauli MPS with site-dependent parity maps, canonical updates and residuals."""
import copy
from functools import lru_cache
import math
import time
import numpy as np
import cpu_numba_kernels as cpu_kernels
import scipy.linalg as la
from cnot_parity import Q, Operation, compile_operations, route_operations, fuse_operations, sweep_operations
from optimized_pauli_svd import RandomizedSVD
from mps_residual import overlap, residual_value
from mps_contractions import absorb_left, absorb_right, join_sites
from mps_qr import economic_qr


@lru_cache(maxsize=256)
def sector_layout(row_charges,column_charges):
    rows=np.asarray(row_charges);cols=np.asarray(column_charges);result=[]
    for q in np.intersect1d(rows,cols):
        ri=np.flatnonzero(rows==q);ci=np.flatnonzero(cols==q)
        ri.flags.writeable=False;ci.flags.writeable=False
        result.append((q,ri,ci))
    return tuple(result)


class FrameMPS:
    def __init__(self,tensors,charges,maps=None,backend='cpu'):
        self.tensors=tensors;self.charges=charges
        self.maps=[Q]*len(tensors) if maps is None else list(maps)
        self.log_scale=0.;self.center=None;self.rsvd=RandomizedSVD('lu',backend=backend)
        self.stats=dict(svd_blocks=0,qr_blocks=0,max_bond=max(map(len,charges)),truncations=0,residual_skipped=0)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.tensors)+sum(x.nbytes for x in self.charges)

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
        for q,ri,ci in sector_layout(tuple(ql),tuple(colq)):
            block=(cpu_kernels.gather(matrix,ri,ci) if cpu_kernels.ENABLED["gather"]
                   else matrix[ri[:,None],ci])
            if not np.any(block):continue
            u,r=economic_qr(block.T)
            pieces.append((ri,ci,u.T,r.T,len(newq)));newq.extend([q]*u.shape[1])
            self.stats['qr_blocks']+=1
        new=np.zeros((len(newq),matrix.shape[1]));link=np.zeros((len(ql),len(newq)))
        for ri,ci,u,r,start in pieces:
            ix=np.arange(start,start+len(u));new[ix[:,None],ci]=u;link[ri[:,None],ix]=r
        self.tensors[i]=new.reshape(len(newq),4,len(qr))
        self.tensors[i-1]=absorb_right(self.tensors[i-1],link)
        self.charges[i]=np.array(newq,dtype=int)

    def _left_qr(self,i):
        a=self.tensors[i];ql,qr=self.charges[i:i+2]
        matrix=a.reshape(-1,len(qr));rowq=(ql[:,None]^np.array(self.maps[i])[None,:]).ravel()
        pieces=[];newq=[]
        for q,ri,ci in sector_layout(tuple(rowq),tuple(qr)):
            block=(cpu_kernels.gather(matrix,ri,ci) if cpu_kernels.ENABLED["gather"]
                   else matrix[ri[:,None],ci])
            if not np.any(block):continue
            u,r=economic_qr(block)
            pieces.append((ri,ci,u,r,len(newq)));newq.extend([q]*u.shape[1]);self.stats['qr_blocks']+=1
        new=np.zeros((matrix.shape[0],len(newq)));link=np.zeros((len(newq),len(qr)))
        for ri,ci,u,r,start in pieces:
            ix=np.arange(start,start+u.shape[1]);new[ri[:,None],ix]=u;link[ix[:,None],ci]=r
        self.tensors[i]=new.reshape(len(ql),4,len(newq))
        self.tensors[i+1]=absorb_left(link,self.tensors[i+1])
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
        if op.kind=='noise_mpo':
            from cnot_noise_mpo import apply_noise_mpo
            return apply_noise_mpo(self,op,cap,callback)
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
                self.tensors[i]=matrix @ self.tensors[i]
            self.maps[i]=after[0];self.rescale();return
        if sites[1]!=i+1:raise ValueError('MPS operation must be routed to adjacent sites')
        a,b=self.tensors[i:i+2];ql,qr=self.charges[i],self.charges[i+2]
        theta=join_sites(a,b)
        if op.kind=='pair_noise':
            if cpu_kernels.ENABLED['transfer']:
                theta=cpu_kernels.transfer(theta.reshape(len(ql),16,len(qr)),np.arange(16),np.diag(matrix)).reshape(theta.shape)
            else:theta=theta*np.diag(matrix).reshape(1,4,4,1)
        elif op.kind in ('cx','routing_swap'):
            # Clifford Pauli transfer is a signed permutation, not a dense GEMM.
            columns=np.argmax(abs(matrix),axis=1)
            if cpu_kernels.ENABLED['transfer']:
                theta=cpu_kernels.transfer(theta.reshape(len(ql),16,len(qr)),columns,matrix[np.arange(16),columns]).reshape(theta.shape)
            else:
                theta=(theta.reshape(len(ql),16,len(qr))[:,columns,:]
                       *matrix[np.arange(16),columns][None,:,None]).reshape(len(ql),4,4,len(qr))
        else:
            theta=(matrix @ theta.reshape(len(ql),16,len(qr))).reshape(len(ql),4,4,len(qr))
        self.maps[i:i+2]=after
        rowq=(ql[:,None]^np.array(after[0])[None,:]).ravel()
        colq=(np.array(after[1])[:,None]^qr[None,:]).ravel()
        m=theta.reshape(len(rowq),len(colq));pieces=[];candidates=[];complete=True
        for q,ri,ci in sector_layout(tuple(rowq),tuple(colq)):
            block=(cpu_kernels.gather(m,ri,ci) if cpu_kernels.ENABLED["gather"]
                   else m[ri[:,None],ci])
            if not np.any(block):continue
            rank=min(cap+16,*block.shape)
            if rank>=min(block.shape):u,s,v=la.svd(block,full_matrices=False,check_finite=True)
            else:
                u,s,v=self.rsvd(block,rank,3);complete=False
            idx=len(pieces);pieces.append((q,ri,ci,u,s,v))
            if not cpu_kernels.ENABLED['select']:
                candidates.extend((float(x),idx,j) for j,x in enumerate(s) if x>0)
            self.stats['svd_blocks']+=1
        if not pieces:raise FloatingPointError('zero MPS after gate')
        if cpu_kernels.ENABLED['select']:
            spectra=[p[4] for p in pieces]
            offsets=np.array([0,*np.cumsum([len(s) for s in spectra])])
            selected,positive=cpu_kernels.select_sorted(np.concatenate(spectra),offsets,cap)
        else:
            if not candidates:raise FloatingPointError('zero MPS after gate')
            candidates.sort(key=lambda x:x[0],reverse=True)
            chosen=[x for x in candidates[:cap] if x[0]>1e-13*candidates[0][0]]
            selected=np.array([(idx,j) for _,idx,j in chosen],dtype=np.int64).reshape(-1,2)
            positive=len(candidates)
        if not positive:raise FloatingPointError('zero MPS after gate')
        left=np.zeros((len(rowq),len(selected)));right=np.zeros((len(selected),len(colq)))
        newq=np.empty(len(selected),dtype=np.int64)
        if cpu_kernels.ENABLED['scatter']:
            for idx,(q,ri,ci,u,s,v) in enumerate(pieces):
                cpu_kernels.scatter_svd(left,right,newq,selected,idx,q,ri,ci,u,s,v)
        else:
            for k,(idx,j) in enumerate(selected):
                q,ri,ci,u,s,v=pieces[idx];left[ri,k]=u[:,j];right[k,ci]=s[j]*v[j];newq[k]=q
        if callback is not None:
            if complete and len(selected)==positive:self.stats['residual_skipped']+=1
            else:callback(self,i,m,left,right)
        self.tensors[i]=left.reshape(len(ql),4,len(selected))
        self.tensors[i+1]=right.reshape(len(selected),4,len(qr));self.charges[i+1]=newq
        self.center=i+1;self.stats['max_bond']=max(self.stats['max_bond'],len(selected))
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
            if len(self.ops[j].sites)==2 or self.ops[j].kind=='noise_mpo':result[j]=guide.copy()
            guide.apply(self.ops[j].adjoint(),self.cap)
        return result


class BudgetGuides:
    """Bound retained snapshot array bytes; regenerate requested guides on demand.

    The budget excludes the working guide, forward state, SVD workspace and
    small Python metadata. A zero budget retains no guide snapshots.
    """
    def __init__(self,operations,boundary,cap,budget_mb):
        if not np.isfinite(budget_mb) or budget_mb<0:raise ValueError('guide_cache_mb must be finite and nonnegative')
        self.ops=operations;self.boundary=boundary;self.cap=cap
        self.budget=int(budget_mb*1024**2);self.ends={};self.bytes=0;self.peak_bytes=0;self.peak_count=0
        self.regenerated_operations=0
        if not self.budget:return
        guide=boundary.copy();span=max(1,math.isqrt(len(operations)))
        for j in range(len(operations)-1,-1,-1):
            k=j+1;size=guide.nbytes
            if k%span==0 and k<len(operations) and size<=self.budget:
                # Remove the least useful (closely spaced, large) checkpoint.
                while self.ends and self.bytes+size>self.budget:
                    keys=sorted([k,*self.ends,len(operations)])
                    def utility(x):
                        i=keys.index(x)
                        return min(x-keys[i-1],keys[i+1]-x)/self.ends[x].nbytes
                    victim=min(self.ends,key=utility)
                    self.bytes-=self.ends.pop(victim).nbytes
                self.ends[k]=guide.copy();self.bytes+=size
                self.peak_bytes=max(self.peak_bytes,self.bytes);self.peak_count=max(self.peak_count,len(self.ends))
            guide.apply(operations[j].adjoint(),cap)

    def get(self,j):
        for k in [k for k in self.ends if k<j+1]:self.bytes-=self.ends.pop(k).nbytes
        end=min((k for k in self.ends if k>=j+1),default=len(self.ops))
        guide=(self.ends[end] if end in self.ends else self.boundary).copy()
        for k in range(end-1,j,-1):
            guide.apply(self.ops[k].adjoint(),self.cap);self.regenerated_operations+=1
        return guide


class ScalarSum:
    """Neumaier summation without retaining one scalar per residual."""
    def __init__(self):self.total=0.;self.correction=0.
    def add(self,value):
        t=self.total+value
        self.correction+=(self.total-t)+value if abs(self.total)>=abs(value) else (value-t)+self.total
        self.total=t
    def value(self):return self.total+self.correction


def evaluate(*,n,steps,sites,phi,scale,singles,pair_rates=0.,edges=None,cap=16,
             guide_cap=32,residual=True,unsigned=False,backend='cpu',
             guide_cache_mb=None,store_contributions=True,contribution_callback=None,
             noise_strategy='local',noise_mpo_max_bond=64):
    from lattice_circuit import chain
    if not isinstance(n,int) or n<2 or n%2 or any(not isinstance(x,int) or x<1 for x in (cap,guide_cap)):
        raise ValueError('use even N >= 2 and positive bond caps')
    if len(sites)!=2 or len(set(sites))!=2 or any(q<0 or q>=n for q in sites):raise ValueError('two distinct in-range sites required')
    start=time.perf_counter();operations=compile_operations(chain(n),steps,phi,singles=singles,
        pair_rates=pair_rates,edges=edges,scale=scale,unsigned=unsigned)
    if noise_strategy not in ('local','mpo'):raise ValueError('noise_strategy must be local or mpo')
    if noise_strategy=='mpo':
        if not isinstance(noise_mpo_max_bond,int) or noise_mpo_max_bond<1:
            raise ValueError('noise_mpo_max_bond must be a positive integer')
        from cnot_noise_mpo import group_noise_mpos
        operations=group_noise_mpos(operations,n,noise_mpo_max_bond)
    operations=route_operations(operations,n,lambda i,j:range(i,j+(1 if i<j else -1),1 if i<j else -1))
    elementary_count=len(operations)
    operations=sweep_operations(fuse_operations(operations))
    codes=[0]*n
    for q in sites:codes[q]=3
    state=FrameMPS.observable(codes,backend);boundary=FrameMPS.singlets(n,unsigned,backend)
    if guide_cache_mb is not None and (not np.isfinite(guide_cache_mb) or guide_cache_mb<0):
        raise ValueError('guide_cache_mb must be finite and nonnegative')
    budget_guides=BudgetGuides(operations,boundary,guide_cap,guide_cache_mb) if residual and guide_cache_mb is not None else None
    checkpoints=Checkpoints(operations,boundary,guide_cap) if residual and budget_guides is None else None
    contributions=[];snapshot_bound=0;count=0;total=ScalarSum();absolute=ScalarSum();peak_bytes=0
    for j,op in enumerate(operations):
        if checkpoints is not None and j%checkpoints.span==0:
            guides=checkpoints.chunk(j);snapshot_bound=max(snapshot_bound,len(guides)+len(checkpoints.ends))
            peak_bytes=max(peak_bytes,sum(g.nbytes for g in guides.values())+sum(g.nbytes for g in checkpoints.ends.values()))
        operation_guide=None
        def record(st,i,m,left,right):
            nonlocal count,operation_guide
            if operation_guide is None:operation_guide=budget_guides.get(j) if budget_guides is not None else guides[j]
            value=residual_value(st,operation_guide,i,m,left,right);count+=1;total.add(value);absolute.add(abs(value))
            if store_contributions or contribution_callback is not None:
                item=dict(operation=j,kind=op.kind,location=list(op.location),bond=i,value=value)
                if store_contributions:contributions.append(item)
                if contribution_callback is not None:contribution_callback(item)
        state.apply(op,cap,record if residual and (len(op.sites)==2 or op.kind=='noise_mpo') else None)
        if checkpoints is not None:guides.pop(j,None)
    base=overlap(state,boundary);delta=total.value()
    corrected=base+delta
    if not all(np.isfinite(x) for x in (base,delta,corrected)):raise FloatingPointError('nonfinite CNOT MPS result')
    return dict(approximate=base,correction=delta,corrected=corrected,details=dict(
        method='cnot-layer-transported-parity-mps',unsigned=unsigned,operations=len(operations),
        elementary_operations=elementary_count,noise_strategy=noise_strategy,
        noise_mpo_max_bond=max((op.max_bond for op in operations if op.kind=='noise_mpo'),default=0),
        noise_mpo_bytes=sum(op.nbytes for op in operations if op.kind=='noise_mpo'),
        routing_swaps=sum(op.kind=='routing_swap' for op in operations),
        guide_snapshot_bound=budget_guides.peak_count if budget_guides is not None else snapshot_bound,guide_cap=guide_cap if residual else None,
        guide_cache_peak_bytes=budget_guides.peak_bytes if budget_guides is not None else peak_bytes,
        guide_cache_budget_mb=guide_cache_mb,
        guide_regenerated_operations=budget_guides.regenerated_operations if budget_guides is not None else None,
        contribution_count=count,contributions_stored=store_contributions,absolute_correction_sum=absolute.value(),
        contributions=contributions,stats=state.stats,charge_leakage=state.leakage(),
        final_standard_frame=state.maps==[Q]*n,runtime_s=time.perf_counter()-start,
        noise_placement='after each nonempty CNOT layer',terminal_strategy='full evolution; no split tail'))
