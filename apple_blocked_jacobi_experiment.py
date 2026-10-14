"""Research-only blocked one-sided Jacobi using Metal Gram-core rotations.

Disjoint column-block pairs form small Gram matrices. The existing shared-memory
Metal Jacobi rotates those cores; batched GEMMs apply their right rotations to
A and V. This explores blocked data movement, not a production SVD: Gram cores
square the condition number, a fixed sweep budget does not certify convergence,
and zero/tiny singular vectors are not completed into an orthonormal basis.
"""
from apple_jacobi_experiment import AdaptiveJacobi


class BlockedGramJacobi:
    def __init__(self, block=16, sweeps=4, inner_sweeps=8, polish=False):
        if block not in (8,16,32) or sweeps<1 or inner_sweeps<1:
            raise ValueError('block must be 8, 16 or 32 and sweep budgets positive')
        self.block=block;self.sweeps=sweeps
        self.local=AdaptiveJacobi(sweeps=inner_sweeps,threads=128,shared=True,deflation_tol=1e-7)
        self.polish=AdaptiveJacobi(sweeps=24,threads=256,shared=True,deflation_tol=1e-7) if polish else None
        self.mx=self.local.mx
        self.permutations={}

    def jacobi(self,a):
        mx=self.mx
        batch,n,width=a.shape;b=self.block
        if width!=n or n%(2*b):
            raise ValueError('square matrix order must be divisible by twice block size')
        blocks=n//b;pairs=blocks//2
        if n not in self.permutations:
            rounds=[]
            for r in range(blocks-1):
                order=[]
                for pair in range(pairs):
                    p=blocks-1 if pair==0 else (r+pair)%(blocks-1)
                    q=r if pair==0 else (r+blocks-1-pair)%(blocks-1)
                    order.extend(range(p*b,(p+1)*b));order.extend(range(q*b,(q+1)*b))
                inv=sorted(range(n),key=order.__getitem__)
                rounds.append((mx.array(order),mx.array(inv)))
            self.permutations[n]=rounds
        rot=a;v=mx.broadcast_to(mx.eye(n),(batch,n,n))
        for _ in range(self.sweeps):
            for order,inv in self.permutations[n]:
                panel=mx.take(rot,order,axis=-1).reshape(batch,n,pairs,2*b).transpose(0,2,1,3)
                gram=mx.swapaxes(panel,-1,-2)@panel
                # Right singular vectors of the PSD Gram core approximate its
                # eigenvectors. Only the orthogonal rotation V is used; no
                # inverse square roots of small Gram eigenvalues are taken.
                # Keep local columns UNSORTED: sorting each pair redistributes
                # columns between blocks and can make cyclic sweeps stall.
                _,_,vh,_=self.local.jacobi_info(gram.reshape(batch*pairs,2*b,2*b),sort=False)
                w=mx.swapaxes(vh,-1,-2).reshape(batch,pairs,2*b,2*b)
                rotated=(panel@w).transpose(0,2,1,3).reshape(batch,n,n)
                vp=mx.take(v,order,axis=-1).reshape(batch,n,pairs,2*b).transpose(0,2,1,3)
                rotated_v=(vp@w).transpose(0,2,1,3).reshape(batch,n,n)
                rot=mx.take(rotated,inv,axis=-1);v=mx.take(rotated_v,inv,axis=-1)
                # Bound lazy graph/temporary growth across outer sweeps.
                mx.eval(rot,v)
        if self.polish is not None:
            u,s,vh=self.polish.jacobi(rot)
            return u,s,vh @ mx.swapaxes(v,-1,-2)
        scale=mx.max(mx.abs(rot),axis=-2);safe=mx.where(scale>0,scale,1)
        s=scale*mx.sqrt(mx.sum((rot/safe[:,None,:])**2,axis=-2))
        order=mx.argsort(-s,axis=-1)
        s=mx.take_along_axis(s,order,axis=-1)
        rot=mx.take_along_axis(rot,order[:,None,:],axis=-1)
        v=mx.take_along_axis(v,order[:,None,:],axis=-1)
        u=rot/mx.where(s>0,s,1)[:,None,:]
        return u,s,mx.swapaxes(v,-1,-2)
