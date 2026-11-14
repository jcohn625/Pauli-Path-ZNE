"""Experimental Apple Silicon shared-buffer GEMM queue; not a default backend.

CPU keeps FP64 authority. GPU products are explicitly mixed precision (FP32).
Only a resolved slot may be refilled. NumPy views are private to this class;
external writes are not used with autograd or retained compiled graphs.
"""
import numpy as np


class SharedBatch:
    def __init__(self,batch,m,k,n,*,budget_bytes=128*1024**2):
        self.shape=(batch,m,k,n)
        if any(not isinstance(x,int) or x<1 for x in self.shape):raise ValueError('positive integer dimensions required')
        # Two reusable input buffers plus one FP32 output. CPU FP64 result and
        # library workspaces are additional; this is not a process memory cap.
        self.buffer_bytes=4*batch*(m*k+k*n+m*n)
        if self.buffer_bytes>budget_bytes:raise ValueError('shared batch exceeds buffer budget')
        import mlx.core as mx
        self.mx=mx;self.a=mx.zeros((batch,m,k),dtype=mx.float32);self.b=mx.zeros((batch,k,n),dtype=mx.float32)
        mx.eval(self.a,self.b)
        self.av=np.array(self.a,copy=False);self.bv=np.array(self.b,copy=False)
        if self.av.flags.owndata or self.bv.flags.owndata:raise RuntimeError('shared NumPy views unavailable')
        self.pending=None

    def submit(self,pairs):
        if self.pending is not None:raise RuntimeError('resolve the in-flight batch before reusing its buffers')
        batch,m,k,n=self.shape
        if len(pairs)!=batch:raise ValueError('batch size mismatch')
        for a,b in pairs:
            if a.shape!=(m,k) or b.shape!=(k,n) or np.iscomplexobj(a) or np.iscomplexobj(b):
                raise ValueError('expected real matrices with the configured shapes')
        for i,(a,b) in enumerate(pairs):
            self.av[i]=a;self.bv[i]=b
        # A fresh graph for each use; no stale result is cached after refilling.
        self.pending=self.a@self.b
        self.mx.async_eval(self.pending)

    def resolve(self):
        if self.pending is None:raise RuntimeError('no submitted batch')
        out=self.pending
        self.mx.eval(out)
        result=np.asarray(out,dtype=np.float64)
        self.pending=None
        return result

    def __call__(self,pairs):
        self.submit(pairs)
        return self.resolve()


class HybridProducts:
    """Measured shape policy; uncalibrated jobs stay FP64 CPU.

    `gpu_shapes` must be selected by end-to-end calibration on this machine.
    Buffers are reused only for consecutive identical GPU batch shapes. A
    caller can submit GPU work and do independent CPU work before resolve().
    """
    def __init__(self,*,gpu_shapes=(),mixed_precision=False,budget_bytes=128*1024**2):
        self.gpu_shapes=set(map(tuple,gpu_shapes));self.mixed_precision=mixed_precision
        self.budget_bytes=budget_bytes;self.pool=None
        self.stats=dict(cpu_batches=0,gpu_batches=0,buffer_bytes=0)

    def __call__(self,pairs):
        if not pairs:return []
        a,b=pairs[0]
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('incompatible matrices')
        shape=(len(pairs),a.shape[0],a.shape[1],b.shape[1])
        compatible=all(x.shape==a.shape and y.shape==b.shape and not np.iscomplexobj(x) and not np.iscomplexobj(y) for x,y in pairs)
        footprint=4*shape[0]*(shape[1]*shape[2]+shape[2]*shape[3]+shape[1]*shape[3])
        if self.mixed_precision and compatible and shape in self.gpu_shapes and footprint<=self.budget_bytes:
            if self.pool is None or self.pool.shape!=shape:
                # Release the old input views before allocating a new shape.
                self.pool=None
                self.pool=SharedBatch(*shape,budget_bytes=self.budget_bytes)
            self.stats['gpu_batches']+=1;self.stats['buffer_bytes']=self.pool.buffer_bytes
            return list(self.pool(pairs))
        self.stats['cpu_batches']+=1
        return [x@y for x,y in pairs]
