"""Opt-in GPU-resident block arrays with batched QR/Jacobi splits.

Python/symmray retain topology and charge bookkeeping. Scalar norms and retained
sector sizes are read by that bookkeeping. The default keeps factors on the
GPU; decomposition_backend="hybrid" selectively synchronizes QR/SVD batches
with CPU LAPACK and returns their FP32 factors to Metal.
Scoped symmray patching is process-global, so do not run concurrent threads.
"""
from collections import defaultdict
from contextlib import contextmanager
import time
from peps_gpu_decomp import GPUDecompositions


@contextmanager
def gpu_boundary(sweeps=16,profile_decompositions=False,optimized=True,parallel_qr=True,
                 decomposition_backend="metal",cpu_qr_min=64,cpu_svd_min=16,qr_panel_width=32):
    import pauli_peps_symmetry as sym
    sym.symmray()
    from symmray.sparse.sparse_array_common import SparseArrayCommon
    import symmray.sparse.sparse_array_common as sparse_module
    from symmray.linalg_common import Absorb
    from autoray import infer_backend
    if decomposition_backend == "metal":
        engine=GPUDecompositions(sweeps=sweeps,optimized=optimized,parallel_qr=parallel_qr,
                                 qr_panel_width=qr_panel_width)
    elif decomposition_backend == "hybrid":
        from peps_hybrid_decomp import HybridDecompositions
        engine=HybridDecompositions(sweeps=sweeps,optimized=optimized,parallel_qr=parallel_qr,
                                    cpu_qr_min=cpu_qr_min,cpu_svd_min=cpu_svd_min,qr_panel_width=qr_panel_width)
    else:
        raise ValueError("decomposition_backend must be metal or hybrid")
    mx=engine.mx
    original=SparseArrayCommon._split_abelian
    original_truncate=sparse_module.truncate_svd_result_blocksparse

    def truncate(U,s,VH,cutoff,cutoff_mode,max_bond,absorb,renorm,backend=None,use_abs=False):
        if not optimized or infer_backend(s.get_any_array())!='mlx' or renorm:
            return original_truncate(U,s,VH,cutoff,cutoff_mode,max_bond,absorb,renorm,backend,use_abs)
        from symmray.linalg_common import absorb_svd_result
        blocks=list(s.get_all_blocks())
        if (cutoff>0) or (max_bond>0):
            all_s=mx.concatenate([mx.abs(b) if use_abs else b for b in blocks])
            all_s=mx.sort(all_s);threshold=None
            if cutoff>0:
                mode=sparse_module._CUTOFF_MODE_MAP[cutoff_mode]
                if mode==1:threshold=mx.array(cutoff)
                elif mode==2:threshold=all_s[-1]*cutoff
                else:
                    cumulative=mx.cumsum(all_s**2 if mode in (3,4) else all_s)
                    target=cutoff*cumulative[-1] if mode in (4,6) else cutoff
                    count=mx.sum(cumulative>=target).astype(mx.int32)
                    threshold=mx.take(all_s,-count)
            if 0<max_bond<all_s.size:
                bound=all_s[-max_bond]
                threshold=bound if threshold is None else mx.maximum(threshold,bound)
            if threshold is not None:
                # One readback of the small rank vector; spectra and factors
                # remain on-device. Preserve symmray's >= tie convention.
                counts=mx.stack([mx.sum((mx.abs(b) if use_abs else b)>=threshold) for b in blocks]).tolist()
                engine.stats['rank_vector_readbacks']=engine.stats.get('rank_vector_readbacks',0)+1
            else:counts=[b.size for b in blocks]
        else:counts=[b.size for b in blocks]
        charges={}
        for (c0,c1),count in zip(U.sectors,counts):
            if not count:
                U.del_block((c0,c1));s.del_block(c1);VH.del_block((c1,c1));continue
            U.set_block((c0,c1),U.get_block((c0,c1))[:,:count])
            s.set_block(c1,s.get_block(c1)[:count])
            VH.set_block((c1,c1),VH.get_block((c1,c1))[:count,:])
            charges[c1]=count
        charges=dict(sorted(charges.items()))
        U.modify(indices=(U.indices[0],U.indices[1].copy_with(chargemap=charges,subinfo=None)))
        VH.modify(indices=(VH.indices[0].copy_with(chargemap=charges,subinfo=None),VH.indices[1]))
        return absorb_svd_result(U,s,VH,absorb)

    def split(self,fn=None,**kw):
        if infer_backend(self.get_any_array())!='mlx' or fn is not None:
            return original(self,fn=fn,**kw)
        if (kw.get('max_bond') or -1)>0 or (kw.get('cutoff') or 0)>0 or kw.get('renorm'):
            return original(self,fn=fn,**kw)
        method=kw.get('method','auto');absorb=Absorb.parse(kw.get('absorb','right'))
        if method=='auto':method='qr' if absorb in (-1,1,10,-11,11,-10) else 'svd'
        if method not in ('qr','svd'):
            raise NotImplementedError(f'GPU boundary split method {method}')
        grouped=defaultdict(list);results={}
        for block in self.blocks.values():grouped[tuple(block.shape)].append(block)
        for blocks in grouped.values():
            arrays=mx.stack(blocks)
            if profile_decompositions:
                ready_start=time.perf_counter();mx.eval(arrays)
                ready_seconds=time.perf_counter()-ready_start
                factor_start=time.perf_counter()
            if method=='qr':
                left=absorb in (-1,-10,-11)
                q,r=engine.qr(mx.swapaxes(arrays,-1,-2) if left else arrays)
                l,ss,rr=(mx.swapaxes(r,-1,-2),None,mx.swapaxes(q,-1,-2)) if left else (q,None,r)
                if absorb in (10,-10):rr=None
                if absorb in (11,-11):l=None
            else:
                l,ss,rr=engine.svd(arrays)
                if absorb==1:rr=ss[:,:,None]*rr;ss=None
                elif absorb==-1:l=l*ss[:,None,:];ss=None
                elif absorb==0:
                    root=mx.sqrt(ss);l=l*root[:,None,:];rr=root[:,:,None]*rr;ss=None
                elif absorb is not None:raise NotImplementedError(f'GPU SVD absorb {absorb}')
            if profile_decompositions:
                mx.eval([x for x in (l,ss,rr) if x is not None])
                engine.stats.setdefault('decomposition_profile',[]).append(dict(
                    method=method,shape=list(arrays.shape),ready_seconds=ready_seconds,
                    factor_seconds=time.perf_counter()-factor_start))
            for i,b in enumerate(blocks):
                results[id(b)]=(None if l is None else l[i],None if ss is None else ss[i],None if rr is None else rr[i])
        return original(self,fn=lambda a,**kwargs:results[id(a)],**kw)

    SparseArrayCommon._split_abelian=split
    sparse_module.truncate_svd_result_blocksparse=truncate
    try:yield engine
    finally:
        SparseArrayCommon._split_abelian=original
        sparse_module.truncate_svd_result_blocksparse=original_truncate


def to_gpu(tn,mx):
    result=tn.copy()
    for t in result:
        t.modify(data=t.data.copy_with(blocks={c:mx.array(b,dtype=mx.float32) for c,b in t.data.blocks.items()}))
    return result
