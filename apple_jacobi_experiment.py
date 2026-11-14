"""Research-only square FP32 Jacobi, never selected by production dispatch.

Small cores keep A and V in threadgroup memory; medium cores keep A there.
Larger cores use device storage. Stops after a full sweep without rotations.
Zero singular columns are returned as zero vectors, NOT an orthogonal nullspace.
Optional deflation stops rotating columns below a Frobenius-relative threshold;
its convergence flag certifies only this thresholded rotation criterion.
Consumers requiring a full isometry must complete it or use a CPU SVD.
"""
from peps_gpu_decomp import GPUDecompositions


class AdaptiveJacobi(GPUDecompositions):
    def __init__(self, sweeps=24, threads=128, shared=True, deflation_tol=0.):
        super().__init__(sweeps=sweeps)
        self.deflation_tol = float(deflation_tol)
        if not 0 <= self.deflation_tol < 1:
            raise ValueError("deflation tolerance must be in [0, 1)")
        self.threads = threads
        self.shared = shared

    def jacobi_info(self, a, sort=True):
        mx = self.mx
        batch, n, width = a.shape
        if width != n or n < 1 or n > 256:
            raise ValueError('square matrices of order 1 through 256 required')
        d = max(2, 1 << (n - 1).bit_length())
        threads = max(self.threads, d // 2)
        if threads % (d // 2):
            raise ValueError('threads must be a multiple of half the padded order')
        shared_a = self.shared and d <= 64
        shared_v = self.shared and d <= 32
        a = mx.pad(a, ((0, 0), (0, d-n), (0, d-n))) if d != n else a
        key = ('adaptive', d, threads, shared_a, shared_v, self.sweeps, self.deflation_tol)
        if key not in self.cache:
            # Column-major shared storage makes the row lanes adjacent.
            decl_a = f'threadgroup float aw[{d*d}];' if shared_a else 'device float* aw = rot + base;'
            decl_v = f'threadgroup float vw[{d*d}];' if shared_v else 'device float* vw = vec + base;'
            barrier = 'mem_flags::mem_threadgroup' if shared_a and shared_v else 'mem_flags::mem_device|mem_flags::mem_threadgroup'
            source = f'''
            constexpr uint D={d}, H={d//2}, T={threads}, L=T/H, S={self.sweeps};
            uint tid=thread_position_in_threadgroup.x;
            uint batch=threadgroup_position_in_grid.x, base=batch*D*D;
            uint pair=tid/L, lane=tid%L;
            {decl_a}
            {decl_v}
            threadgroup float aa[T],bb[T],gg[T],cs[H],sn[H];
            threadgroup uint changes[T], active;
            threadgroup float floor_norm;
            for(uint i=tid;i<D*D;i+=T) {{
                aw[(i%D)*D+i/D]=x[base+i];
                vw[(i%D)*D+i/D]=(i/D==i%D)?1.0f:0.0f;
            }}
            threadgroup_barrier({barrier});
            float norm_part=0;
            for(uint i=tid;i<D*D;i+=T)norm_part+=x[base+i]*x[base+i];
            aa[tid]=norm_part;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0) {{floor_norm=0;for(uint i=0;i<T;++i)floor_norm+=aa[i];floor_norm*={self.deflation_tol**2:.9e}f;}}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint used=0;
            for(uint sweep=0;sweep<S;++sweep) {{
                uint count=0;
                for(uint round=0;round<D-1;++round) {{
                    uint p=(pair==0)?D-1:(round+pair)%(D-1);
                    uint q=(pair==0)?round:(round+D-1-pair)%(D-1);
                    float av=0,bv=0,gv=0;
                    for(uint row=lane;row<D;row+=L) {{
                        float ap=aw[p*D+row], aq=aw[q*D+row];
                        av+=ap*ap; bv+=aq*aq; gv+=ap*aq;
                    }}
                    aa[tid]=av; bb[tid]=bv; gg[tid]=gv;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if(lane==0) {{
                        float alpha=0,beta=0,gamma=0;
                        for(uint t=0;t<L;++t) {{alpha+=aa[pair*L+t];beta+=bb[pair*L+t];gamma+=gg[pair*L+t];}}
                        float c=1,s=0;
                        float threshold=1e-6f*metal::sqrt(alpha)*metal::sqrt(beta);
                        if(alpha>floor_norm && beta>floor_norm && metal::fabs(gamma)>threshold && metal::fabs(gamma)>1e-30f) {{
                            float tau=(beta-alpha)/(2*gamma);
                            float t=(metal::fabs(tau)>1e10f)?0.5f/tau:
                                metal::copysign(1.0f,tau)/(metal::fabs(tau)+metal::sqrt(1+tau*tau));
                            c=metal::rsqrt(1+t*t);s=t*c;++count;
                        }}
                        cs[pair]=c;sn[pair]=s;
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float c=cs[pair],s=sn[pair];
                    for(uint row=lane;row<D;row+=L) {{
                        float ap=aw[p*D+row],aq=aw[q*D+row],vp=vw[p*D+row],vq=vw[q*D+row];
                        aw[p*D+row]=c*ap-s*aq;aw[q*D+row]=s*ap+c*aq;
                        vw[p*D+row]=c*vp-s*vq;vw[q*D+row]=s*vp+c*vq;
                    }}
                    threadgroup_barrier({barrier});
                }}
                changes[tid]=count;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(tid==0) {{active=0;for(uint i=0;i<T;++i)active+=changes[i];}}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                used=sweep+1;
                if(active==0)break;
            }}
            if(tid==0) {{info[batch*2]=used;info[batch*2+1]=(active==0)?1:0;}}
            // Read/write in column-major order for device arrays: no in-place transpose.
            for(uint i=tid;i<D*D;i+=T) {{rot[base+i]=aw[i];vec[base+i]=vw[i];}}
            '''
            if d > 64:
                source = source.replace('aw[(i%D)*D+i/D]', 'aw[i]').replace('vw[(i%D)*D+i/D]', 'vw[i]')
                for buffer in ('aw', 'vw'):
                    for col in ('p', 'q'):
                        source = source.replace(f'{buffer}[{col}*D+row]', f'{buffer}[row*D+{col}]')
            self.cache[key] = mx.fast.metal_kernel(
                name=f'adaptive_jacobi_{d}_{threads}_{int(self.shared)}_{self.sweeps}_{str(self.deflation_tol).replace('.', '_').replace('-', 'm')}',
                input_names=['x'], output_names=['rot','vec','info'], source=source)
        rot, vec, info = self.cache[key](inputs=[a], grid=(batch*threads,1,1),
            threadgroup=(threads,1,1), output_shapes=[(batch,d,d),(batch,d,d),(batch,2)],
            output_dtypes=[mx.float32,mx.float32,mx.int32])
        if d <= 64:
            rot = mx.swapaxes(rot,-1,-2)
            vec = mx.swapaxes(vec,-1,-2)
        column_scale = mx.max(mx.abs(rot),axis=-2)
        safe = mx.where(column_scale > 0, column_scale, 1)
        s = column_scale * mx.sqrt(mx.sum((rot/safe[:,None,:])**2,axis=-2))
        order = (mx.argsort(-s,axis=-1)[...,:n] if sort else
                 mx.broadcast_to(mx.arange(n),(batch,n)))
        s = mx.take_along_axis(s,order,axis=-1)
        rot = mx.take_along_axis(rot,order[:,None,:],axis=-1)[:,:n,:]
        u = rot / mx.where(s > 0,s,1)[:,None,:]
        v = mx.take_along_axis(vec,order[:,None,:],axis=-1)[:,:n,:]
        return u,s,mx.swapaxes(v,-1,-2),info

    def jacobi(self, a):
        return self.jacobi_info(a)[:3]
