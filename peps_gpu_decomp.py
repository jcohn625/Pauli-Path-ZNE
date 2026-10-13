"""Experimental real float32 GPU QR and one-sided Jacobi SVD.

Jacobi uses the round-robin pairing of the earlier NumbaQAOA MPS Metal
kernels, specialized to real blocks and fused into one dispatch per batch.
Householder QR supports rectangular and rank-deficient blocks. No CPU
factorizations, normal-equation SVDs, or intermediate tensor readbacks are used.
"""
class GPUDecompositions:
    def __init__(self, sweeps=16, optimized=True, parallel_qr=True):
        import mlx.core as mx
        if sweeps<1:raise ValueError('sweeps must be positive')
        self.mx=mx;self.sweeps=int(sweeps);self.cache={};self.optimized=optimized
        self.parallel_qr=parallel_qr and optimized
        self._qr_flush_bytes=64*2**20
        self.stats=dict(qr_batches=0,svd_batches=0,qr_blocks=0,svd_blocks=0,max_core=0)

    def qr(self,a):
        """Reduce wide and tall matrices before the single-workgroup kernel."""
        mx=self.mx;batch,m,n=a.shape
        self.stats.setdefault('qr_shapes',{})
        shape=f'{m}x{n}'
        self.stats['qr_shapes'][shape]=self.stats['qr_shapes'].get(shape,0)+batch
        if m<n:
            # The leading square block determines Q, including a complete
            # orthogonal basis when it is rank-deficient. Apply Q to the rest
            # with a GPU GEMM rather than looping over all wide columns.
            q,r=self._qr_direct(a[...,:m])
            return q,mx.concatenate((r,mx.swapaxes(q,-1,-2)@a[...,m:]),axis=-1)
        tile=max(2*n,64)
        count=m//tile
        if count>=2:
            # TSQR: no padding, since padded null-space vectors could spoil
            # orthogonality after cropping a rank-deficient input.
            base,extra=divmod(m,count);groups={};offset=0
            for i in range(count):
                size=base+(i<extra)
                groups.setdefault(size,[]).append((i,offset,offset+size));offset+=size
            if self.optimized:
                packed_groups=[];r_groups=[]
                for size,jobs in groups.items():
                    g=len(jobs);first=jobs[0][0]
                    packed=a[:,jobs[0][1]:jobs[-1][2],:].reshape(batch,g,size,n)
                    packed=mx.transpose(packed,(1,0,2,3)).reshape(g*batch,size,n)
                    q,r=self._qr_direct(packed)
                    packed_groups.append((size,g,first,q))
                    r_groups.append(mx.transpose(r.reshape(g,batch,n,n),(1,0,2,3)).reshape(batch,g*n,n))
                top,r=self.qr(mx.concatenate(r_groups,axis=-2))
                q_groups=[]
                for size,g,first,q in packed_groups:
                    factors=top[:,first*n:(first+g)*n,:].reshape(batch,g,n,n)
                    factors=mx.transpose(factors,(1,0,2,3))
                    product=q.reshape(g,batch,size,n)@factors
                    q_groups.append(mx.transpose(product,(1,0,2,3)).reshape(batch,g*size,n))
                return mx.concatenate(q_groups,axis=-2),r
            qs={};rs={}
            for size,jobs in groups.items():
                packed=mx.concatenate([a[:,lo:hi,:] for _,lo,hi in jobs],axis=0)
                q,r=self._qr_direct(packed)
                for j,(i,_,_) in enumerate(jobs):
                    qs[i]=q[j*batch:(j+1)*batch];rs[i]=r[j*batch:(j+1)*batch]
            top,r=self.qr(mx.concatenate([rs[i] for i in range(count)],axis=-2))
            q=mx.concatenate([qs[i]@top[:,i*n:(i+1)*n,:] for i in range(count)],axis=-2)
            return q,r
        return self._qr_direct(a)

    def _qr_direct(self,a):
        if min(a.shape[-2:])>128:
            return self._blocked_qr(a)
        return self._qr_kernel(a)

    def _blocked_qr(self,a):
        """Compact-WY Householder panels; trailing updates use GPU GEMM."""
        mx=self.mx;batch,m,n=a.shape;k=min(m,n)
        # Large lazy QR graphs can retain dozens of successively sized GEMM
        # buffers. Materialize bounded panel groups without copying to host.
        flush_large=self.optimized and batch*m*n*4>=self._qr_flush_bytes
        work=a;panels=[];r_rows=[]
        for j in range(0,k,32):
            width=min(32,k-j)
            active=work if self.optimized else work[:,j:,j:]
            _,r,v,tau=self._qr_kernel(active[...,:width],reflectors=True)
            gram=mx.swapaxes(v,-1,-2)@v
            key=('wy',width)
            if key not in self.cache:
                source=f'''
                constexpr uint B={width};
                uint b=thread_position_in_grid.x;
                for(uint i=0;i<B*B;++i)t[b*B*B+i]=0;
                for(uint j=0;j<B;++j) {{
                    t[b*B*B+j*B+j]=tau[b*B+j];
                    for(uint i=0;i<j;++i) {{
                        float sum=0;
                        for(uint l=i;l<j;++l)sum+=t[b*B*B+i*B+l]*g[b*B*B+l*B+j];
                        t[b*B*B+i*B+j]=-tau[b*B+j]*sum;
                    }}
                }}
                '''
                self.cache[key]=mx.fast.metal_kernel(name=f'peps_wy_{width}',
                    input_names=['g','tau'],output_names=['t'],source=source)
            t=self.cache[key](inputs=[gram,tau],grid=(batch,1,1),threadgroup=(1,1,1),
                output_shapes=[(batch,width,width)],output_dtypes=[mx.float32])[0]
            trailing=active[...,width:]
            trailing=trailing-v@(mx.swapaxes(t,-1,-2)@(mx.swapaxes(v,-1,-2)@trailing))
            if self.optimized:
                r_rows.append(mx.concatenate((mx.zeros((batch,width,j)),r,trailing[:,:width,:]),axis=-1))
                work=trailing[:,width:,:]
            else:
                panel=mx.pad(r,((0,0),(0,m-j-width),(0,0)))
                lower=mx.concatenate((work[:,j:,:j],panel,trailing),axis=-1)
                work=mx.concatenate((work[:,:j,:],lower),axis=-2)
            panels.append((j,v,t))
            if flush_large and (len(panels)%4==0 or j+width==k):
                mx.eval(work,*r_rows,*[p[1] for p in panels],*[p[2] for p in panels])
                self.stats['large_qr_flushes']=self.stats.get('large_qr_flushes',0)+1
        if self.optimized:
            q=mx.zeros((batch,m-k,0))
            for j,v,t in reversed(panels):
                width=v.shape[-1]
                upper=mx.concatenate((mx.broadcast_to(mx.eye(width),(batch,width,width)),
                    mx.zeros((batch,width,q.shape[-1]))),axis=-1)
                lower=mx.concatenate((mx.zeros((batch,q.shape[-2],width)),q),axis=-1)
                q=mx.concatenate((upper,lower),axis=-2)
                q=q-v@(t@(mx.swapaxes(v,-1,-2)@q))
                if flush_large and (j%128==0):
                    mx.eval(q)
                    self.stats['large_qr_flushes']=self.stats.get('large_qr_flushes',0)+1
            return q,mx.concatenate(r_rows,axis=-2)
        q=mx.broadcast_to(mx.eye(m,k,dtype=mx.float32),(batch,m,k))
        for j,v,t in reversed(panels):
            lower=q[:,j:,:]
            lower=lower-v@(t@(mx.swapaxes(v,-1,-2)@lower))
            q=mx.concatenate((q[:,:j,:],lower),axis=-2)
        return q,work[:,:k,:]

    def _qr_kernel(self,a,reflectors=False):
        mx=self.mx
        batch,m,n=a.shape;k=min(m,n);threads=128
        make_q=not (reflectors and self.optimized)
        key=('qr',m,n,make_q,self.parallel_qr)
        if key not in self.cache:
            source=f'''
            constexpr uint M={m}, N={n}, K={k}, T={threads};
            constexpr bool MAKE_Q={'true' if make_q else 'false'};
            uint tid=thread_position_in_threadgroup.x;
            uint batch=threadgroup_position_in_grid.x;
            uint rb=batch*M*N, qb=batch*M*K, vb=batch*M*K;
            threadgroup float sums[T];
            threadgroup float alpha, tau, vfirst;
            for(uint i=tid;i<M*N;i+=T) work[rb+i]=x[rb+i];
            for(uint i=tid;i<M*K;i+=T) {{if(MAKE_Q)q[qb+i]=(i/K==i%K)?1.0f:0.0f;v[vb+i]=0.0f;}}
            if(!MAKE_Q && tid==0 && batch==0)q[0]=0;
            threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
            for(uint j=0;j<K;++j) {{
                float norm=0;
                for(uint row=j+tid;row<M;row+=T) {{float z=work[rb+row*N+j];norm+=z*z;}}
                sums[tid]=norm;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(tid==0) {{
                    float total=0;for(uint t=0;t<T;++t) total+=sums[t];
                    float xj=work[rb+j*N+j];
                    alpha=-metal::copysign(metal::sqrt(total),xj);
                    vfirst=xj-alpha;
                    tau=(total>0)?(alpha-xj)/alpha:0.0f;
                    taus[batch*K+j]=tau;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for(uint row=j+tid;row<M;row+=T)
                    v[vb+row*K+j]=(row==j)?1.0f:((vfirst!=0)?work[rb+row*N+j]/vfirst:0.0f);
                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
                // Each thread updates a whole column; columns are independent.
                for(uint col=j+1+tid;col<N;col+=T) {{
                    float dot=0;for(uint row=j;row<M;++row)dot+=v[vb+row*K+j]*work[rb+row*N+col];
                    dot*=tau;
                    for(uint row=j;row<M;++row)work[rb+row*N+col]-=v[vb+row*K+j]*dot;
                }}
                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
                for(uint row=j+tid;row<M;row+=T)work[rb+row*N+j]=(row==j)?alpha:0.0f;
                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
            }}
            // Q = H0 H1 ... H(K-1), formed by reverse application to thin I.
            if(MAKE_Q)for(int j=int(K)-1;j>=0;--j) {{
                float tj=taus[batch*K+j];
                for(uint col=tid;col<K;col+=T) {{
                    float dot=0;for(uint row=j;row<M;++row)dot+=v[vb+row*K+j]*q[qb+row*K+col];
                    dot*=tj;
                    for(uint row=j;row<M;++row)q[qb+row*K+col]-=v[vb+row*K+j]*dot;
                }}
                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
            }}
            for(uint i=tid;i<K*N;i+=T) r[batch*K*N+i]=work[rb+i];
            '''
            if self.parallel_qr:
                # Cooperating lanes reduce each column dot product. All
                # threads execute every round, including inactive columns.
                def parallel_update(buffer,base,stride,first,stop,coefficient):
                    return f'''
                    for(uint round=0;round<({stop}-({first})+31)/32;++round) {{
                        uint group=tid/4,lane=tid%4,col={first}+round*32+group;
                        float dot=0;
                        if(col<{stop})for(uint row=j+lane;row<M;row+=4)
                            dot+=v[vb+row*K+j]*{buffer}[{base}+row*{stride}+col];
                        sums[tid]=dot;
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        float projection=0;
                        for(uint lane2=0;lane2<4;++lane2)projection+=sums[group*4+lane2];
                        projection*={coefficient};
                        if(col<{stop})for(uint row=j+lane;row<M;row+=4)
                            {buffer}[{base}+row*{stride}+col]-=v[vb+row*K+j]*projection;
                        threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
                    }}
                    '''
                start=source.index('                // Each thread updates')
                end=source.index('                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);',start)
                source=source[:start]+parallel_update('work','rb','N','j+1','N','tau')+source[end:]
                start=source.index('                for(uint col=tid;col<K;col+=T)')
                end=source.index('                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);',start)
                source=source[:start]+parallel_update('q','qb','K','0','K','tj')+source[end:]
            self.cache[key]=mx.fast.metal_kernel(name=f'peps_householder_{m}_{n}_q{int(make_q)}_p{int(self.parallel_qr)}',
                input_names=['x'],output_names=['q','r','work','v','taus'],source=source)
        q,r,work,v,tau=self.cache[key](inputs=[a],grid=(batch*threads,1,1),threadgroup=(threads,1,1),
            output_shapes=[(batch,m,k) if make_q else (1,),(batch,k,n),(batch,m,n),(batch,m,k),(batch,k)],
            output_dtypes=[mx.float32]*5)
        self.stats['qr_batches']+=1;self.stats['qr_blocks']+=batch
        return (q,r,v,tau) if reflectors else (q,r)

    def jacobi(self,a):
        mx=self.mx;batch,n,_=a.shape
        d=max(2,1<<(n-1).bit_length())
        if d>512:raise ValueError(f'Jacobi core {n} exceeds experimental limit 512')
        a=mx.pad(a,((0,0),(0,d-n),(0,d-n))) if d!=n else a
        threads=256;half=d//2;lanes=threads//half
        key=('jacobi',d,self.sweeps)
        if key not in self.cache:
            source=f'''
            constexpr uint D={d}, H={half}, L={lanes}, T={threads}, S={self.sweeps};
            uint tid=thread_position_in_threadgroup.x;
            uint batch=threadgroup_position_in_grid.x, base=batch*D*D;
            uint pair=tid/L,lane=tid%L;
            threadgroup float aa[T],bb[T],gg[T],cs[H],sn[H];
            for(uint i=tid;i<D*D;i+=T) {{a[base+i]=x[base+i];v[base+i]=(i/D==i%D)?1.0f:0.0f;}}
            threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
            for(uint sweep=0;sweep<S;++sweep) for(uint round=0;round<D-1;++round) {{
                uint p=(pair==0)?D-1:(round+pair)%(D-1);
                uint q=(pair==0)?round:(round+D-1-pair)%(D-1);
                float av=0,bv=0,gv=0;
                for(uint row=lane;row<D;row+=L) {{
                    float ap=a[base+row*D+p],aq=a[base+row*D+q];
                    av+=ap*ap;bv+=aq*aq;gv+=ap*aq;
                }}
                aa[tid]=av;bb[tid]=bv;gg[tid]=gv;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if(lane==0) {{
                    float alpha=0,beta=0,gamma=0;
                    for(uint t=0;t<L;++t) {{alpha+=aa[pair*L+t];beta+=bb[pair*L+t];gamma+=gg[pair*L+t];}}
                    float c=1,s=0;
                    if(metal::fabs(gamma)>1e-6f*metal::sqrt(alpha*beta) && metal::fabs(gamma)>1e-30f) {{
                        float tau=(beta-alpha)/(2*gamma);
                        float t=(metal::fabs(tau)>1e10f)?0.5f/tau:
                            metal::copysign(1.0f,tau)/(metal::fabs(tau)+metal::sqrt(1.0f+tau*tau));
                        c=metal::rsqrt(1+t*t);s=t*c;
                    }}
                    cs[pair]=c;sn[pair]=s;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float c=cs[pair],s=sn[pair];
                for(uint row=lane;row<D;row+=L) {{
                    uint ip=base+row*D+p,iq=base+row*D+q;
                    float ap=a[ip],aq=a[iq],vp=v[ip],vq=v[iq];
                    a[ip]=c*ap-s*aq;a[iq]=s*ap+c*aq;
                    v[ip]=c*vp-s*vq;v[iq]=s*vp+c*vq;
                }}
                threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
            }}
            '''
            self.cache[key]=mx.fast.metal_kernel(name=f'peps_jacobi_{d}_{self.sweeps}',
                input_names=['x'],output_names=['a','v'],source=source)
        rot,v=self.cache[key](inputs=[a],grid=(batch*threads,1,1),threadgroup=(threads,1,1),
            output_shapes=[(batch,d,d)]*2,output_dtypes=[mx.float32]*2)
        s=mx.sqrt(mx.sum(rot*rot,axis=-2))
        order=mx.argsort(-s,axis=-1)[...,:n]
        s=mx.take_along_axis(s,order,axis=-1)
        u=mx.take_along_axis(rot,order[:,None,:],axis=-1)[:,:n,:]/mx.maximum(s[:,None,:],1e-30)
        v=mx.take_along_axis(v,order[:,None,:],axis=-1)[:,:n,:]
        self.stats['max_core']=max(self.stats['max_core'],n)
        return u,s,mx.swapaxes(v,-1,-2)

    def svd(self,a):
        mx=self.mx;batch,m,n=a.shape
        wide=m<n
        x=mx.swapaxes(a,-1,-2) if wide else a
        scale=mx.maximum(mx.max(mx.abs(x),axis=(-2,-1),keepdims=True),1e-30)
        q,r=self.qr(x/scale)
        u,s,vh=self.jacobi(r)
        u=q@u;s=s*scale[...,0]
        self.stats['svd_batches']+=1;self.stats['svd_blocks']+=batch
        return (mx.swapaxes(vh,-1,-2),s,mx.swapaxes(u,-1,-2)) if wide else (u,s,vh)
