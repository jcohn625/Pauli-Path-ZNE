"""Experimental packed Z2xZ2 sublayers: CPU decompositions, optional Metal products.

No GPU SVD and no dense expansion of the full PEPS. Metal arithmetic is float32;
results are converted to float64 before CPU decompositions. Packing, synchronization,
and host conversion are included in the recorded stage times.
"""
from collections import defaultdict
import time
import numpy as np


class LayerEngine:
    def __init__(self, backend='cpu'):
        if backend not in ('cpu','metal'):
            raise ValueError('backend must be cpu or metal')
        self.backend=backend
        self.stats=dict(layers=0,gates=0,scale_seconds=0.,contract_seconds=0.,
                        decompose_seconds=0.,total_seconds=0.,matmul_blocks=0,
                        matmul_batches=0,scale_blocks=0)
        if backend=='metal':
            import mlx.core as mx
            self.mx=mx
            # Trigger actual device access rather than trusting is_available().
            mx.eval(mx.ones(1))
            self.pack=mx.fast.metal_kernel(name='peps_pack_scale',
                input_names=['x','weights','meta','offsets','limits'],output_names=['y'],
                source='''
                uint i = thread_position_in_grid.x;
                if (i >= limits[1]) return;
                uint lo = 0, hi = limits[0];
                while (lo + 1 < hi) {
                    uint mid = (lo + hi) / 2;
                    if (offsets[mid] <= i) lo = mid; else hi = mid;
                }
                uint base = lo * 16;
                uint j = i - offsets[lo], loc = 0;
                float factor = 1.0f;
                for (int a = 4; a >= 0; --a) {
                    uint dim = meta[base + 1 + a];
                    uint coord = j % dim; j /= dim;
                    loc += coord * meta[base + 6 + a];
                    factor *= weights[meta[base + 11 + a] + coord];
                }
                y[i] = x[meta[base] + loc] * factor;
                ''')

    def scale_many(self, requests):
        """Requests: (Tensor, {index: BlockVector}, output index order)."""
        start=time.perf_counter()
        from quimb.tensor import Tensor
        outputs=[]
        tasks=[]
        for tensor,scales,order in requests:
            axes=tuple(tensor.inds.index(ix) for ix in order)
            blocks={}
            for sector,block in tensor.data.blocks.items():
                vectors=[scales[ix].blocks[sector[a]] if ix in scales else np.ones(block.shape[a])
                         for a,ix in enumerate(tensor.inds)]
                key=tuple(sector[a] for a in axes)
                tasks.append((block,axes,vectors,blocks,key))
            indices=tuple(tensor.data.indices[a] for a in axes)
            outputs.append((tensor,indices,blocks,order))
        self.stats['scale_blocks']+=len(tasks)
        if self.backend=='cpu':
            for block,axes,vectors,blocks,key in tasks:
                out=block.copy()
                for axis,v in enumerate(vectors):
                    shape=[1]*out.ndim;shape[axis]=len(v)
                    out*=v.reshape(shape)
                blocks[key]=out.transpose(axes)
        elif tasks:
            data=[];weights=[np.ones(1,dtype=np.float32)];meta=[];offsets=[0]
            input_offset=0;weight_offset=1
            for block,axes,vectors,_,_ in tasks:
                if block.ndim>5:raise ValueError('kernel supports at most five site indices')
                data.append(np.asarray(block,dtype=np.float32).ravel())
                strides=[];stride=1
                for d in reversed(block.shape):strides.append(stride);stride*=d
                strides=list(reversed(strides))
                dims=[block.shape[a] for a in axes]+[1]*(5-block.ndim)
                perm_strides=[strides[a] for a in axes]+[0]*(5-block.ndim)
                woffs=[]
                for a in axes:
                    woffs.append(weight_offset)
                    weights.append(np.asarray(vectors[a],dtype=np.float32))
                    weight_offset+=len(vectors[a])
                woffs += [0]*(5-block.ndim)
                meta.extend([input_offset,*dims,*perm_strides,*woffs])
                input_offset+=block.size;offsets.append(offsets[-1]+block.size)
            mx=self.mx
            out=self.pack(inputs=[mx.array(np.concatenate(data)),mx.array(np.concatenate(weights)),
                                  mx.array(meta,dtype=mx.uint32),mx.array(offsets,dtype=mx.uint32),
                                  mx.array([len(tasks),offsets[-1]],dtype=mx.uint32)],
                          grid=(offsets[-1],1,1),threadgroup=(256,1,1),
                          output_shapes=[(offsets[-1],)],output_dtypes=[mx.float32])[0]
            mx.eval(out)
            host=np.asarray(out,dtype=np.float64)
            for k,(block,axes,_,blocks,key) in enumerate(tasks):
                blocks[key]=host[offsets[k]:offsets[k+1]].reshape(tuple(block.shape[a] for a in axes))
        result=[Tensor(t.data.copy_with(indices=indices,blocks=blocks),inds=order,tags=t.tags)
                for t,indices,blocks,order in outputs]
        self.stats['scale_seconds']+=time.perf_counter()-start
        return result

    def contract_many(self, requests):
        """Binary contractions grouped by sector-matrix shape across all gates."""
        from quimb.tensor import Tensor
        start=time.perf_counter()
        products=defaultdict(list);descriptions=[]
        for left,right in requests:
            common=[ix for ix in left.inds if ix in right.inds]
            li=[ix for ix in left.inds if ix not in common]
            ri=[ix for ix in right.inds if ix not in common]
            if not common or not li or not ri:
                raise ValueError('layer contractions require open and contracted indices')
            # fuse() places groups by their first original axis, not argument
            # order. Transpose first so both fused contracted bases align.
            a,b=left.data.align_axes(right.data,axes=(
                tuple(left.inds.index(ix) for ix in common),
                tuple(right.inds.index(ix) for ix in common)))
            a=a.transpose(tuple(left.inds.index(ix) for ix in li+common))
            a=a.fuse(tuple(range(len(li))),tuple(range(len(li),a.ndim)))
            b=b.transpose(tuple(right.inds.index(ix) for ix in common+ri))
            b=b.fuse(tuple(range(len(common))),tuple(range(len(common),b.ndim)))
            blocks={}
            for (ql,qk),ab in a.blocks.items():
                for (qj,qr),bb in b.blocks.items():
                    if qj==qk:
                        if ab.shape[1]!=bb.shape[0]:raise ValueError('misaligned fused contracted basis')
                        products[(ab.shape[0],ab.shape[1],bb.shape[1])].append((ab,bb,blocks,(ql,qr)))
            charge=tuple(x^y for x,y in zip(a.charge,b.charge))
            descriptions.append((a,b,blocks,charge,li,ri))
        self.stats['matmul_batches']+=len(products)
        self.stats['matmul_blocks']+=sum(map(len,products.values()))
        if self.backend=='metal':
            mx=self.mx;pending=[]
            for tasks in products.values():
                a=mx.array(np.stack([x[0] for x in tasks]).astype(np.float32))
                b=mx.array(np.stack([x[1] for x in tasks]).astype(np.float32))
                pending.append((a@b,tasks))
            mx.eval([p[0] for p in pending])
            for product,tasks in pending:
                host=np.asarray(product,dtype=np.float64)
                for k,(_,_,blocks,key) in enumerate(tasks):
                    if key in blocks:raise ValueError('unexpected repeated fused charge product')
                    blocks[key]=host[k]
        else:
            for tasks in products.values():
                for a,b,blocks,key in tasks:
                    if key in blocks:raise ValueError('unexpected repeated fused charge product')
                    blocks[key]=np.einsum('ij,jk->ik',a,b,optimize=False)
        result=[]
        for a,b,blocks,charge,li,ri in descriptions:
            array=a.new_with(indices=(a.indices[0],b.indices[1]),charge=charge,blocks=blocks)
            if len(ri)>1:array=array.unfuse(-1)
            if len(li)>1:array=array.unfuse(0)
            result.append(Tensor(array,inds=li+ri))
        self.stats['contract_seconds']+=time.perf_counter()-start
        return result

    def apply(self,peps,gauges,gates,nx,cap,cutoff):
        """Apply one matching. Gates contain (site_a,site_b,block-sparse gate)."""
        from quimb.tensor import Tensor
        from quimb.tensor.tensor_core import rand_uuid
        start=time.perf_counter()
        sites=[q for a,b,_ in gates for q in (a,b)]
        if len(sites)!=len(set(sites)):raise ValueError('batched gates must be disjoint')
        jobs=[];scaling=[]
        for a,b,g in gates:
            ca,cb=divmod(a,nx),divmod(b,nx)
            tl,tr=peps[ca],peps[cb]
            il,ir=peps.site_ind(*ca),peps.site_ind(*cb)
            bond,=set(tl.inds)&set(tr.inds)
            orders=([ix for ix in tl.inds if ix not in (bond,il)]+[bond,il],
                    [bond,ir]+[ix for ix in tr.inds if ix not in (bond,ir)])
            outer=[]
            for t,order in zip((tl,tr),orders):
                scales={};undo={}
                for ix in t.inds:
                    if ix not in gauges:continue
                    if ix==bond:scales[ix]=gauges[ix]**.5
                    else:
                        scales[ix]=gauges[ix]+1e-12*gauges[ix].max()
                        undo[ix]=scales[ix]**-1
                scaling.append((t,scales,order));outer.append(undo)
            jobs.append(dict(tl=tl,tr=tr,il=il,ir=ir,bond=bond,gate=g,undo=outer,
                             ql=rand_uuid(),qr=rand_uuid(),pl=rand_uuid(),pr=rand_uuid()))
        scaled=self.scale_many(scaling)
        before=time.perf_counter()
        for k,j in enumerate(jobs):
            l,r=scaled[2*k:2*k+2]
            j['lq'],j['lr']=l.split(left_inds=None,right_inds=[j['bond'],j['il']],
                absorb='right',cutoff=0.,bond_ind=j['ql'],get='tensors')
            j['rl'],j['rq']=r.split(left_inds=[j['bond'],j['ir']],right_inds=None,
                absorb='left',cutoff=0.,bond_ind=j['qr'],get='tensors')
        self.stats['decompose_seconds']+=time.perf_counter()-before
        pairs=self.contract_many([(j['lr'].reindex({j['il']:j['pl']}),
                                   j['rl'].reindex({j['ir']:j['pr']})) for j in jobs])
        centers=self.contract_many([(t,Tensor(j['gate'],inds=[j['il'],j['ir'],j['pl'],j['pr']]))
                                    for t,j in zip(pairs,jobs)])
        before=time.perf_counter();log_scale=0.
        for j,center in zip(jobs,centers):
            j['lr'],s,j['rl']=center.split(left_inds=[j['ql'],j['il']],
                right_inds=[j['qr'],j['ir']],bond_ind=j['bond'],get='tensors',
                absorb=None,max_bond=cap,cutoff=cutoff)
            norm=float(s.data.norm())
            if norm<=0 or not np.isfinite(norm):raise FloatingPointError('invalid singular norm')
            gauges[j['bond']]=s.data/norm;log_scale+=np.log(norm)
        self.stats['decompose_seconds']+=time.perf_counter()-before
        reconstructed=self.contract_many([(a,b) for j in jobs
                                         for a,b in [(j['lq'],j['lr']),(j['rl'],j['rq'])]])
        requests=[(reconstructed[2*k+side],j['undo'][side],j['tl' if side==0 else 'tr'].inds)
                  for k,j in enumerate(jobs) for side in (0,1)]
        final=self.scale_many(requests)
        for k,j in enumerate(jobs):
            j['tl'].modify(data=final[2*k].data)
            j['tr'].modify(data=final[2*k+1].data)
        self.stats['layers']+=1;self.stats['gates']+=len(gates)
        self.stats['total_seconds']+=time.perf_counter()-start
        return float(log_scale)
