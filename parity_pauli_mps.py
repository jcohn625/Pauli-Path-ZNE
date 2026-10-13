"""Real Z2 x Z2 charge-block MPS, valid for arbitrary Pauli-diagonal noise."""
import numpy as np
from u1_pauli_mps import U1MPS
from optimized_pauli_svd import RandomizedSVD

Q=np.array([0,3,1,2]) # The two conserved relative Pauli-count parities.
class ParityMPS(U1MPS):
    physical_charges=np.bitwise_xor(Q[:,None],Q[None,:]).ravel()
    add=staticmethod(np.bitwise_xor)
    subtract=staticmethod(np.bitwise_xor)
    dtype=float

    def noise(self,eta):
        d=np.column_stack([np.ones(len(eta)),eta])
        for i,a in enumerate(self.tensors):
            self.tensors[i]=a*(d[2*i,:,None]*d[2*i+1,None,:]).reshape(1,16,1)

def gates(solver,phi,unsigned=False):
    trans,c,n=solver.build_heisenberg_pauli_transfer(phi)
    if unsigned:c=np.abs(c)
    gate=np.zeros((16,16))
    for i in range(16):
        for k in range(n[i]):gate[trans[i,k],i]+=c[i,k]
    q=ParityMPS.physical_charges
    assert np.max(abs(gate[q[:,None]!=q[None,:]]))==0
    na,nb,co,counts=solver.build_odd_pair_transfer_tables(trans,c,n)
    ins=[];outs=[];coeff=[]
    for a in range(16):
        for b in range(16):
            for k in range(counts[a,b]):
                ins.append(16*a+b);outs.append(16*int(na[a,b,k])+int(nb[a,b,k]));coeff.append(co[a,b,k])
    return gate,(np.array(ins),np.array(outs),np.array(coeff))

def run(solver,n,layers,q1,q2,eta,cap,terminal=2,n_iter=3,backend='cpu',phi=np.pi/16,unsigned=False):
    if n<2 or n%2 or cap<1 or not (0<=q1<n and 0<=q2<n) or q1==q2:
        raise ValueError('Use even n, positive cap, and two distinct in-range sites')
    if not 0<=terminal<=layers:raise ValueError('Invalid terminal depth')
    local=[]
    for i in range(n//2):
        v=np.zeros(16);v[4*(3 if 2*i in (q1,q2) else 0)+(3 if 2*i+1 in (q1,q2) else 0)]=1
        local.append(v)
    prefix=ParityMPS.product(local)
    b=np.zeros(16);b[[0,5,10,15]]=[1,1,1,1] if unsigned else [1,-1,-1,-1]
    suffix=ParityMPS.product([b for _ in local])
    fg,ft=gates(solver,phi,unsigned);bg,bt=gates(solver,-phi,unsigned)
    rsvd=RandomizedSVD('lu',backend=backend)
    for _ in range(layers-terminal):
        prefix.even(fg);prefix.noise(eta);prefix.odd(ft,cap,rsvd,n_iter)
        prefix.noise(eta);prefix.rescale()
    for _ in range(terminal):
        suffix.noise(eta);suffix.odd(bt,cap,rsvd,n_iter);suffix.noise(eta);suffix.even(bg)
    env=np.ones((1,1))
    for a,b in zip(prefix.tensors,suffix.tensors):
        env=np.einsum('ab,apr,bps->rs',env,a,b,optimize=True)
    value=float(env[0,0]*np.exp(prefix.log_scale+suffix.log_scale))
    assert np.isfinite(value)
    return value,dict(prefix=prefix.stats,suffix=suffix.stats,rsvd=rsvd.stats,
        max_bond=max(map(len,prefix.charges)),charge_leakage=max(prefix.leakage(),suffix.leakage()))
