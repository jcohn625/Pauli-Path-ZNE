"""U(1)-resolved blocked operator MPS for axially symmetric Pauli noise.

Local orthonormal operator basis: I, Z, (X+iY)/sqrt(2), (X-iY)/sqrt(2).
Bond charge is cumulative adjoint Sz charge. Tensor storage remains dense;
QR and SVD are performed in separate charge blocks with a GLOBAL bond cap.
This is not an SU(2) implementation and does not support absolute-path Q.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import numba as nb
from optimized_pauli_svd import RandomizedSVD

C = np.zeros((4,4),complex)
C[0,0]=C[3,1]=1
C[1,2]=C[1,3]=1/np.sqrt(2)
C[2,2]=1j/np.sqrt(2);C[2,3]=-1j/np.sqrt(2)
CB=np.kron(C,C)
Q1=np.array([0,0,1,-1])
QP=(Q1[:,None]+Q1[None,:]).ravel()

@nb.njit(cache=True)
def transfer(theta, inputs, outputs, coefficients):
    out=np.zeros_like(theta)
    for k in range(len(coefficients)):
        a,b=inputs[k]//16,inputs[k]%16
        c,d=outputs[k]//16,outputs[k]%16
        for l in range(theta.shape[0]):
            for r in range(theta.shape[3]):
                out[l,c,d,r]+=coefficients[k]*theta[l,a,b,r]
    return out

def gate_data(solver,phi):
    trans,coeff,n=solver.build_heisenberg_pauli_transfer(phi)
    old=np.zeros((16,16))
    for i in range(16):
        for k in range(n[i]):old[trans[i,k],i]+=coeff[i,k]
    gate=CB.conj().T@old@CB
    assert np.max(np.abs(gate[QP[:,None]!=QP[None,:]]))<1e-13
    gate[np.abs(gate)<1e-14]=0
    ins=[];outs=[];co=[]
    for a in range(16):
        for b in range(16):
            mid=(a%4)*4+b//4
            for new in np.flatnonzero(gate[:,mid]):
                ins.append(16*a+b)
                outs.append(16*((a//4)*4+new//4)+(new%4)*4+b%4)
                co.append(gate[new,mid])
    return gate,(np.array(ins),np.array(outs),np.array(co))

@dataclass
class U1MPS:
    physical_charges=QP
    add=staticmethod(np.add)
    subtract=staticmethod(np.subtract)
    dtype=complex
    tensors:list
    charges:list
    log_scale:float=0.
    stats:dict=field(default_factory=lambda:dict(svd_blocks=0,qr_blocks=0,max_sector=0,
                                                dense_svd_elements=0,sector_svd_elements=0))

    @classmethod
    def product(cls,local_vectors):
        tensors=[];charges=[np.array([0])]
        cumulative=0
        for v in local_vectors:
            occupied=np.unique(cls.physical_charges[np.abs(v)>1e-13])
            if len(occupied)!=1:raise ValueError('Boundary must have definite U(1) charge')
            cumulative=int(cls.add(cumulative,int(occupied[0])));charges.append(np.array([cumulative]))
            tensors.append(v.astype(cls.dtype).reshape(1,16,1))
        return cls(tensors,charges)

    def even(self,gate):
        self.tensors=[np.einsum('pq,lqr->lpr',gate,a,optimize=True) for a in self.tensors]

    def noise(self,eta):
        if not np.allclose(eta[:,0],eta[:,1],rtol=0,atol=1e-14):
            raise ValueError('U(1) requires eta_X == eta_Y at every site')
        d=np.column_stack([np.ones(len(eta)),eta[:,2],eta[:,0],eta[:,0]])
        for i,a in enumerate(self.tensors):
            self.tensors[i]=a*(d[2*i,:,None]*d[2*i+1,None,:]).reshape(1,16,1)

    def canonicalize(self):
        for site in range(len(self.tensors)-1,0,-1):
            a=self.tensors[site];ql=self.charges[site];qr=self.charges[site+1]
            matrix=a.reshape(len(ql),-1)
            colq=self.subtract(qr[None,:],self.physical_charges[:,None]).ravel()
            pieces=[];newq=[]
            for q in np.unique(ql):
                ri=np.flatnonzero(ql==q);ci=np.flatnonzero(colq==q)
                if not len(ci):continue
                block=matrix[np.ix_(ri,ci)]
                if not np.any(block):continue
                u,r=la.qr(block.conj().T,mode='economic',check_finite=True)
                pieces.append((ri,ci,u.conj().T,r.conj().T,len(newq)))
                newq.extend([q]*u.shape[1]);self.stats['qr_blocks']+=1
            new=np.zeros((len(newq),matrix.shape[1]),self.dtype)
            link=np.zeros((len(ql),len(newq)),self.dtype)
            for ri,ci,u,r,start in pieces:
                ix=np.arange(start,start+len(u))
                new[np.ix_(ix,ci)]=u;link[np.ix_(ri,ix)]=r
            self.tensors[site]=new.reshape(len(newq),16,len(qr))
            self.tensors[site-1]=np.einsum('lpa,ab->lpb',self.tensors[site-1],link,optimize=True)
            self.charges[site]=np.array(newq,dtype=int)

    def odd(self,tables,cap,rsvd,n_iter=3,residual_callback=None):
        self.canonicalize()
        for site in range(len(self.tensors)-1):
            a,b=self.tensors[site:site+2]
            theta=np.einsum('lam,mbr->labr',a,b,optimize=True)
            theta=transfer(theta,*tables)
            ql,qr=self.charges[site],self.charges[site+2]
            rowq=self.add(ql[:,None],self.physical_charges[None,:]).ravel()
            colq=self.subtract(qr[None,:],self.physical_charges[:,None]).ravel()
            matrix=theta.reshape(len(rowq),len(colq))
            pieces=[];candidates=[]
            self.stats['dense_svd_elements']+=matrix.size
            for q in np.intersect1d(rowq,colq):
                ri=np.flatnonzero(rowq==q);ci=np.flatnonzero(colq==q)
                block=matrix[np.ix_(ri,ci)]
                if not np.any(block):continue
                self.stats['svd_blocks']+=1
                self.stats['sector_svd_elements']+=block.size
                self.stats['max_sector']=max(self.stats['max_sector'],min(block.shape))
                target=min(cap+24,*block.shape)
                if target>=min(block.shape):
                    u,s,v=la.svd(block,full_matrices=False,lapack_driver='gesdd',check_finite=True)
                else:u,s,v=rsvd(block,target,n_iter)
                idx=len(pieces);pieces.append((q,ri,ci,u,s,v))
                candidates.extend((float(x),idx,j) for j,x in enumerate(s) if x>0)
            if not candidates:raise ValueError('Zero operator encountered')
            candidates.sort(key=lambda t:t[0],reverse=True)
            # Relative cutoff is invariant to the tracked global normalization.
            chosen=[t for t in candidates[:cap] if t[0]>1e-12*candidates[0][0]]
            left=np.zeros((len(rowq),len(chosen)),self.dtype)
            right=np.zeros((len(chosen),len(colq)),self.dtype);newq=[]
            for k,(_,idx,j) in enumerate(chosen):
                q,ri,ci,u,s,v=pieces[idx]
                left[ri,k]=u[:,j];right[k,ci]=s[j]*v[j];newq.append(q)
            # Optional diagnostic observes the actual discarded matrix without
            # changing the factors or subsequent forward trajectory.
            if residual_callback is not None:
                residual_callback(self,site,matrix,left,right)
            self.tensors[site]=left.reshape(len(ql),16,len(chosen))
            self.tensors[site+1]=right.reshape(len(chosen),16,len(qr))
            self.charges[site+1]=np.array(newq,dtype=int)

    def rescale(self):
        for i,a in enumerate(self.tensors):
            scale=np.max(np.abs(a))
            if scale==0:raise ValueError('Zero MPS norm')
            self.tensors[i]=a/scale;self.log_scale+=np.log(scale)

    def leakage(self):
        largest=0.
        for i,a in enumerate(self.tensors):
            allowed=self.add(self.charges[i][:,None,None],self.physical_charges[None,:,None])==self.charges[i+1][None,None,:]
            largest=max(largest,float(np.max(np.abs(a[~allowed]),initial=0)))
        return largest

def run(solver,n,layers,q1,q2,eta,cap,terminal=2,n_iter=3,backend='cpu'):
    if backend!='cpu':
        raise ValueError('U1 Metal disabled: full-circuit validation failed the imaginary-observable check. Use CPU.')
    if n<2 or n%2 or cap<1 or not (0<=q1<n and 0<=q2<n) or q1==q2:
        raise ValueError('Use even n, positive cap, and two distinct in-range sites')
    if not (0<=terminal<=layers):raise ValueError('Invalid terminal depth')
    local=[]
    for i in range(n//2):
        v=np.zeros(16);v[4*(3 if 2*i in (q1,q2) else 0)+(3 if 2*i+1 in (q1,q2) else 0)]=1
        local.append(CB.conj().T@v)
    prefix=U1MPS.product(local)
    boundary=np.zeros(16);boundary[[0,5,10,15]]=[1,-1,-1,-1]
    suffix=U1MPS.product([CB.conj().T@boundary for _ in local])
    forward,ft=gate_data(solver,np.pi/16);backward,bt=gate_data(solver,-np.pi/16)
    rsvd=RandomizedSVD('lu',backend=backend,audit=False)
    for _ in range(layers-terminal):
        prefix.even(forward);prefix.noise(eta);prefix.odd(ft,cap,rsvd,n_iter)
        prefix.noise(eta);prefix.rescale()
    for _ in range(terminal):
        suffix.noise(eta);suffix.odd(bt,cap,rsvd,n_iter);suffix.noise(eta)
        suffix.even(backward)
    env=np.ones((1,1),complex)
    for a,b in zip(prefix.tensors,suffix.tensors):
        env=np.einsum('ab,apr,bps->rs',env,a,b.conj(),optimize=True)
    value=env[0,0]*np.exp(prefix.log_scale+suffix.log_scale)
    if abs(value.imag)>1e-9:raise ValueError(f'Observable acquired imaginary part: {value}')
    return float(value.real),dict(prefix=prefix.stats,suffix=suffix.stats,rsvd=rsvd.stats,
        max_bond=max(map(len,prefix.charges)),charge_leakage=max(prefix.leakage(),suffix.leakage()))
