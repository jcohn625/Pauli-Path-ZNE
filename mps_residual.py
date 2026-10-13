"""A posteriori scalar corrections from actual parity-MPS truncation residuals.

Exact adjoints make the residual telescope exact. Finite-bond guides are an
approximation; they do not change forward compression. An optional bounded
snapshot cache reuses guides; uncached layers are regenerated from checkpoints.
"""
import copy
import math
import time
import numpy as np
from parity_pauli_mps import ParityMPS,gates
from optimized_pauli_svd import RandomizedSVD
import scipy.linalg as la
from contiguous_pauli_kernel import odd_transfer




def overlap(a,b):
    env=np.ones((1,1));logs=a.log_scale+b.log_scale
    for x,y in zip(a.tensors,b.tensors):
        env=np.einsum('ab,apr,bps->rs',env,x,y,optimize=True)
        scale=np.max(abs(env))
        if not scale:return 0.
        env/=scale;logs+=np.log(scale)
    return float(env[0,0]*np.exp(logs))


def residual_value(state,guide,i,m,left_factor,right_factor):
    left=np.ones((1,1));right=np.ones((1,1))
    logs=state.log_scale+guide.log_scale
    for a,b in zip(state.tensors[:i],guide.tensors[:i]):
        left=np.einsum('ab,apr,bps->rs',left,a,b,optimize=True)
        scale=np.max(abs(left))
        if not scale:return 0.
        left/=scale;logs+=np.log(scale)
    for j in range(len(state.tensors)-1,i+1,-1):
        right=np.einsum('apr,bps,rs->ab',state.tensors[j],guide.tensors[j],right,optimize=True)
        scale=np.max(abs(right))
        if not scale:return 0.
        right/=scale;logs+=np.log(scale)
    a=np.einsum('lg,gam->lam',left,guide.tensors[i],optimize=True)
    b=np.einsum('mbr,hr->mbh',guide.tensors[i+1],right,optimize=True)
    a=a.reshape(-1,a.shape[-1]);b=b.reshape(b.shape[0],-1)
    # Contract the low-rank gradient with the actual residual. Avoid forming a
    # second full-size center or gradient; the guide bond controls this cost.
    projected=m@b.T-left_factor@(right_factor@b.T)
    return float(np.sum(a*projected)*np.exp(logs))


class Guides:
    def __init__(self,solver,n,layers,eta,cap,cache_mb=0,forward_layers=None,phi=np.pi/16,unsigned=False,backend="cpu"):
        self.solver=solver;self.cap=cap;self.rsvd=RandomizedSVD('lu',backend=backend)
        self.bt,self.bc,self.bn=solver.build_heisenberg_pauli_transfer(-phi)
        if unsigned:self.bc=np.abs(self.bc)
        self.tables=solver.build_odd_pair_transfer_tables(self.bt,self.bc,self.bn)
        self.diag=solver.build_block_noise_diag(eta)
        state=boundary_mps(solver,n,unsigned);self.checkpoints={};self.cached={}
        self.cache_bytes=0;self.cached_layers=0
        budget=max(0,int(cache_mb*1024**2))
        if forward_layers is None:forward_layers=layers
        # A conservative bound for one layer's independent snapshots. Deep
        # copies are intentional: the legacy noise map mutates tensor arrays.
        layer_bound=(n//2-1)*(n//2)*16*cap*cap*8
        for t in range(layers-1,-1,-1):
            cache=t<forward_layers and self.cache_bytes+layer_bound<=budget
            if t<forward_layers and not cache:self.checkpoints[t]=copy.deepcopy(state)
            snapshots=self.reverse_layer(state,cache)
            if cache:
                self.cached[t]=snapshots;self.cached_layers+=1
                self.cache_bytes+=sum(a.nbytes for st in snapshots.values() for a in st.tensors)
    def reverse_layer(self,state,snapshots):
        s=self.solver;result={}
        s.apply_block_noise_inplace(state,self.diag,parallel_kernels='serial')
        for i in range(len(state.tensors)-2,-1,-1):
            if snapshots:result[i]=copy.deepcopy(state)
            single_gate(s,state,i,self.tables,self.cap,self.rsvd)
        s.apply_block_noise_inplace(state,self.diag,parallel_kernels='serial')
        s.apply_even_layer_inplace(state,self.bt,self.bc,self.bn,parallel_kernels='serial')
        rescale_mps_inplace(state)
        return result
    def layer(self,t):
        if t in self.cached:return self.cached.pop(t)
        return self.reverse_layer(copy.deepcopy(self.checkpoints[t]),True)


def run(solver,n,layers,q1,q2,eta,cap,guide_cap,terminal=2,progress=False,guide_cache_mb=0,phi=np.pi/16,unsigned=False,backend="cpu"):
    if n<2 or n%2 or min(cap,guide_cap)<1 or q1==q2 or not (0<=q1<n and 0<=q2<n):
        raise ValueError('Use even n, positive bonds, and distinct in-range sites')
    if not 0<=terminal<=min(layers,2):
        raise ValueError('This benchmark supports zero to two terminal layers')
    start=time.perf_counter()
    dual=Guides(solver,n,layers,eta,guide_cap,guide_cache_mb,layers-terminal,phi,unsigned,backend)
    cache_bytes=dual.cache_bytes;cached_layers=dual.cached_layers
    guide_seconds=time.perf_counter()-start
    if progress:print(f'guide {guide_cap}: checkpoints {guide_seconds:.2f}s',flush=True)
    local=[]
    for i in range(n//2):
        v=np.zeros(16);v[4*(3 if 2*i in (q1,q2) else 0)+(3 if 2*i+1 in (q1,q2) else 0)]=1
        local.append(v)
    front=ParityMPS.product(local)
    fg,ft=gates(solver,phi,unsigned);bg,bt=gates(solver,-phi,unsigned)
    rsvd=RandomizedSVD('lu',backend=backend);contributions=[];regeneration=0.;correction_seconds=0.
    for t in range(layers-terminal):
        tick=time.perf_counter();guide_layer=dual.layer(t);regeneration+=time.perf_counter()-tick
        def record(state,i,m,left,right):
            nonlocal correction_seconds
            tick=time.perf_counter()
            value=residual_value(state,guide_layer[i],i,m,left,right)
            correction_seconds+=time.perf_counter()-tick
            contributions.append(dict(layer=t,bond=i,value=value))
        front.even(fg);front.noise(eta);front.odd(ft,cap,rsvd,3,residual_callback=record)
        front.noise(eta);front.rescale()
    del dual
    if layers>terminal:del guide_layer
    def suffix(k):
        b=np.zeros(16);b[[0,5,10,15]]=[1,1,1,1] if unsigned else [1,-1,-1,-1]
        back=ParityMPS.product([b for _ in local])
        for _ in range(terminal):
            back.noise(eta);back.odd(bt,k,rsvd,3);back.noise(eta);back.even(bg)
        return back
    base=overlap(front,suffix(cap))
    tick=time.perf_counter()
    # Each cross-cut adjoint gate increases Schmidt rank by at most 16.
    # Two suffix layers therefore have rank <=256 from a product boundary.
    tail=overlap(front,suffix(16**terminal))-base
    tail_seconds=time.perf_counter()-tick
    delta=math.fsum([tail]+[x['value'] for x in contributions])
    return dict(base=base,delta=delta,corrected=base+delta,tail_delta=tail,
        abs_sum=math.fsum([abs(tail)]+[abs(x['value']) for x in contributions]),
        contributions=contributions,runtime_s=time.perf_counter()-start,
        guide_build_s=guide_seconds,guide_regeneration_s=regeneration,
        guide_cache_bytes=cache_bytes,cached_layers=cached_layers,
        residual_contraction_s=correction_seconds,tail_correction_s=tail_seconds)



def single_gate(solver,state,site,tables,cap,rsvd):
    """Mixed-canonical single-bond compression for backward guide construction."""
    solver.right_canonicalize_inplace(state)
    for j in range(site):
        a=state.tensors[j];q,r=la.qr(a.reshape(-1,a.shape[-1]),mode='economic')
        state.tensors[j]=q.reshape(a.shape[0],16,q.shape[1])
        state.tensors[j+1]=np.einsum('ab,bpr->apr',r,state.tensors[j+1],optimize=True)
    a,b=state.tensors[site:site+2]
    theta=odd_transfer(np.einsum('lam,mbr->labr',a,b,optimize=True),*tables)
    m=theta.reshape(a.shape[0]*16,16*b.shape[-1])
    target=min(cap+24,*m.shape)
    if target>=min(m.shape):u,s,v=la.svd(m,full_matrices=False)
    else:u,s,v=rsvd(m,target,3)
    keep=min(cap,max(1,np.count_nonzero(s>1e-12*s[0])))
    state.tensors[site]=u[:,:keep].reshape(a.shape[0],16,keep)
    state.tensors[site+1]=(s[:keep,None]*v[:keep]).reshape(keep,16,b.shape[-1])

def boundary_mps(solver, n_qubits, absolute):
    local = np.zeros(16, dtype=float)
    for code in range(16):
        left, right = divmod(code, 4)
        if left == right:
            local[code] = 1.0 if absolute or left == 0 else -1.0
    return solver.PauliMPS([local.reshape(1, 16, 1).copy() for _ in range(n_qubits // 2)])

def rescale_mps_inplace(mps):
    """Move harmless per-tensor scales into log_scale to avoid overflow."""
    for site, tensor in enumerate(mps.tensors):
        scale = float(np.max(np.abs(tensor)))
        if scale > 0.0 and scale != 1.0:
            mps.tensors[site] = tensor / scale
            mps.log_scale += float(np.log(scale))
