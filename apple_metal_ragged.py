"""Experimental ragged FP32 Metal contractions; never selected by solvers.

One dispatch evaluates differently shaped products and their output reductions.
The simple scalar kernel is a baseline for measuring launch/packing costs, not
an attempted replacement for tuned GEMM. All host conversion costs are exposed.
"""
from collections import OrderedDict
import numpy as np


class RaggedProducts:
    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        self.kernel = mx.fast.metal_kernel(
            name='pauli_ragged_products',
            input_names=['data', 'jobs', 'groups', 'offsets', 'limits'],
            output_names=['out'], source='''
            uint t = thread_position_in_grid.x;
            if (t >= limits[1]) return;
            uint lo = 0, hi = limits[0];
            while (lo + 1 < hi) {
                uint mid = (lo + hi) / 2;
                if (offsets[mid] <= t) lo = mid; else hi = mid;
            }
            uint local = t - offsets[lo];
            float sum = 0.0f;
            uint begin = groups[2*lo], end = begin + groups[2*lo+1];
            for (uint g = begin; g < end; ++g) {
                uint base = 5*g;
                uint a = jobs[base], b = jobs[base+1];
                uint k = jobs[base+3], n = jobs[base+4];
                uint row = local / n, col = local % n;
                float product = 0.0f;
                for (uint p = 0; p < k; ++p)
                    product += data[a+row*k+p] * data[b+p*n+col];
                sum += product;
            }
            out[t] = sum;
            ''')

    def prepare(self, lb, rb, plan, lo, ro):
        groups = OrderedDict()
        for job in plan:groups.setdefault(job[2], []).append(job)
        arrays=[]; jobs=[]; groupmeta=[]; offsets=[0]; descriptions=[]; size=0
        for key, entries in groups.items():
            groupmeta.extend((len(jobs)//5,len(entries)))
            for i,j,_,shape,(m,k,n) in entries:
                a=lb[i][1].transpose(lo).reshape(m,k)
                b=rb[j][1].transpose(ro).reshape(k,n)
                if np.iscomplexobj(a) or np.iscomplexobj(b):raise ValueError('real blocks required')
                jobs.extend((size,size+a.size,m,k,n));size+=a.size+b.size
                arrays.extend((a.ravel(),b.ravel()))
            offsets.append(offsets[-1]+m*n);descriptions.append((key,shape))
        if max(size,offsets[-1],len(jobs)) >= 2**32:
            raise ValueError('32-bit kernel indexing limit exceeded')
        mx=self.mx
        inputs=[mx.array(np.concatenate(arrays).astype(np.float32)),
                mx.array(jobs,dtype=mx.uint32),mx.array(groupmeta,dtype=mx.uint32),
                mx.array(offsets,dtype=mx.uint32),mx.array([len(groups),offsets[-1]],dtype=mx.uint32)]
        return inputs,offsets,descriptions

    def dispatch(self, prepared):
        inputs,offsets,_=prepared
        return self.kernel(inputs=inputs,grid=(offsets[-1],1,1),threadgroup=(128,1,1),
                           output_shapes=[(offsets[-1],)],output_dtypes=[self.mx.float32])[0]

    def resolve(self, out, prepared):
        self.mx.eval(out)
        host=np.asarray(out,dtype=np.float64)
        _,offsets,descriptions=prepared
        return {key:host[offsets[i]:offsets[i+1]].reshape(shape)
                for i,(key,shape) in enumerate(descriptions)}

    def __call__(self, lb, rb, plan, lo, ro):
        if not plan:return {}
        prepared=self.prepare(lb,rb,plan,lo,ro)
        return self.resolve(self.dispatch(prepared),prepared)
