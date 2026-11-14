"""Opt-in Metal contractions with selective CPU LAPACK factorizations.

Routing uses actual sector matrix dimensions. Device synchronization, NumPy
views, and returned MLX arrays are explicit; unified memory does not make the
boundary free. Factors remain FP32 on the GPU. Global sector rank selection
continues to live in peps_gpu_boundary, outside this engine.
"""
import time
import numpy as np
import scipy.linalg as la
from peps_gpu_decomp import GPUDecompositions


class HybridDecompositions(GPUDecompositions):
    def __init__(self, *args, cpu_qr_min=64, cpu_svd_min=16, **kwargs):
        super().__init__(*args, **kwargs)
        if cpu_qr_min<1 or cpu_svd_min<1:
            raise ValueError('CPU decomposition thresholds must be positive')
        self.cpu_qr_min=cpu_qr_min;self.cpu_svd_min=cpu_svd_min

    def _cpu(self, a, method):
        tick=time.perf_counter()
        self.mx.eval(a)
        host=np.asarray(a)
        if host.dtype!=np.float32 or host.ndim!=3:
            raise ValueError('hybrid decompositions require batched real FP32 arrays')
        if method=='qr':
            factors=[la.qr(block,mode='economic',check_finite=True) for block in host]
        else:
            factors=[la.svd(block,full_matrices=False,check_finite=True) for block in host]
        result=tuple(self.mx.array(np.stack([f[k] for f in factors])) for k in range(len(factors[0])))
        self.stats[f'cpu_{method}_batches']=self.stats.get(f'cpu_{method}_batches',0)+1
        self.stats[f'cpu_{method}_blocks']=self.stats.get(f'cpu_{method}_blocks',0)+len(host)
        self.stats['cpu_boundary_seconds']=self.stats.get('cpu_boundary_seconds',0.)+time.perf_counter()-tick
        self.stats['cpu_boundary_array_bytes']=self.stats.get('cpu_boundary_array_bytes',0)+host.nbytes+sum(x.nbytes for x in result)
        return result

    def qr(self,a):
        if min(a.shape[-2:])>=self.cpu_qr_min:return self._cpu(a,'qr')
        return super().qr(a)

    def svd(self,a):
        if min(a.shape[-2:])>=self.cpu_svd_min:return self._cpu(a,'svd')
        return super().svd(a)
