"""CNOT-layer Pauli-Lindblad compiler with transported Z2 x Z2 labels.

Operations act on Pauli coefficient vectors in backwards (Heisenberg) order.
The original noise factors stay diagonal; no dense matching PTM is constructed.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from cnot_zne.pauli_zne.circuit import PAULIS, DIRECTIONS, rotation_blocks, cnot, ptm

Q = (0, 3, 1, 2)
BASIS = np.array([np.kron(a, b) for a in PAULIS for b in PAULIS])
ANTI = np.array([[int(p != 0 and p != a) for a in (1,2,3)] for p in range(4)])
SWAP = np.eye(16).reshape(4,4,4,4).transpose(1,0,2,3).reshape(16,16)


@dataclass(frozen=True)
class Operation:
    sites: tuple
    matrix: np.ndarray
    before: tuple
    after: tuple
    kind: str
    location: tuple = ()  # forward (step, matching, CX occurrence)

    def adjoint(self):
        return Operation(self.sites, self.matrix.T, self.after, self.before,
                         self.kind, self.location)


def check_operation(op, tolerance=2e-12):
    def summed(maps):
        q=np.array([0])
        for values in maps:q=(q[:,None]^np.asarray(values)[None,:]).ravel()
        return q
    incoming,outgoing=map(summed,(op.before,op.after))
    leakage=float(np.max(abs(op.matrix[outgoing[:,None]!=incoming[None,:]]),initial=0))
    if leakage>tolerance:raise ValueError(f'{op.kind} breaks transported parity: {leakage}')
    return leakage


def _frame(generators):
    axes=[]
    for generator in generators:
        coefficients=np.einsum('aij,ji->a',BASIS,generator).real/4
        k=int(np.argmax(abs(coefficients)))
        if not np.allclose(generator,coefficients[k]*BASIS[k],atol=2e-12,rtol=0) or abs(abs(coefficients[k])-1)>2e-12:
            raise ValueError('decomposition does not transport the symmetry to Pauli generators')
        axes.append(divmod(k,4))
    return tuple(tuple(sum((1<<bit)*int(p!=0 and axes[bit][site]!=0 and p!=axes[bit][site])
                           for bit in range(2)) for p in range(4)) for site in range(2))


@lru_cache(maxsize=128)
def elementary(phi, unsigned=False):
    """Forward local circuit, represented by backwards transfer matrices."""
    generators=[BASIS[15].copy(),BASIS[10].copy()]  # ZZ, YY -> Q=(0,3,1,2)
    frame=(Q,Q);result=[]
    for stage,rotations in enumerate(rotation_blocks(phi)):
        for site,u in enumerate(rotations):
            full=np.kron(u,PAULIS[0]) if site==0 else np.kron(PAULIS[0],u)
            generators=[full@g@full.conj().T for g in generators]
            new=_frame(generators)
            matrix=ptm(u)
            if unsigned:matrix=abs(matrix)
            op=Operation((site,),matrix,(new[site],),(frame[site],),'rotation')
            check_operation(op)
            if not (op.before==op.after and np.array_equal(matrix,np.eye(4))):
                result.append((stage,op))
            frame=new
        if stage<3:
            u=cnot(DIRECTIONS[stage]);generators=[u@g@u.conj().T for g in generators]
            new=_frame(generators);matrix=ptm(u)
            if unsigned:matrix=abs(matrix)
            op=Operation((0,1),matrix,new,frame,'cx')
            check_operation(op);result.append((stage,op));frame=new
    if frame!=(Q,Q):raise ValueError('matching did not restore the original parity frame')
    return tuple(result)


def _rates(values, shape, name):
    a=np.asarray(values,dtype=float)
    try:a=np.broadcast_to(a,shape)
    except ValueError as e:raise ValueError(f'{name} must broadcast to {shape}; got {a.shape}') from e
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError(f'{name} must be finite and nonnegative')
    return a


def normalize_noise(lattice, steps, singles, pair_rates=0., edges=None):
    if edges is None:edges=sorted({tuple(sorted(b)) for layer in lattice.layers for b in layer.bonds})
    edges=tuple(tuple(e) for e in edges)
    if any(len(e)!=2 or any(not isinstance(q,(int,np.integer)) for q in e) or
           e[0]>=e[1] or e[0]<0 or e[1]>=lattice.n_sites for e in edges):
        raise ValueError('noise edges must be unique integer pairs 0 <= i < j < N')
    if len(set(edges))!=len(edges):raise ValueError('duplicate noise edges')
    prefix=(steps,len(lattice.layers),3)
    single=_rates(singles,prefix+(lattice.n_sites,3),'single rates')
    pair=_rates(pair_rates,prefix+(len(edges),3,3),'pair rates')
    return single,pair,edges


def compile_operations(lattice, steps, phi, *, singles=(.001,.002,.003), pair_rates=0.,
                       edges=None, scale=1., unsigned=False):
    """Once-per-CX-layer noise, including idle sites and all supplied edges.

    Rate axes: (step, matching, CX occurrence, site, axis), or
    (step, matching, CX occurrence, edge, left axis, right axis).
    Any trailing shape broadcasting to these full shapes is accepted.
    Empty matchings are omitted, including their noise, matching cnot_zne.
    """
    if not isinstance(steps,int) or steps<0 or not np.isfinite(scale) or scale<0:
        raise ValueError('steps must be a nonnegative integer; scale must be finite and nonnegative')
    if not np.isfinite(phi):raise ValueError('phi must be finite')
    single,pair,edges=normalize_noise(lattice,steps,singles,pair_rates,edges)
    local=elementary(float(phi),unsigned);forward=[];frames=[Q]*lattice.n_sites
    # Local elementary records carry backwards matrices and later/earlier maps.
    for t in range(steps):
        for layer_index,layer in enumerate(lattice.layers):
            if not layer.bonds:continue
            for stage in range(4):
                for _,template in (entry for entry in local if entry[0]==stage and entry[1].kind=='rotation'):
                    for bond in layer.bonds:
                        sites=tuple(bond[x] for x in template.sites)
                        if tuple(frames[q] for q in sites)!=template.after:raise ValueError('frame mismatch')
                        forward.append(Operation(sites,template.matrix,template.before,template.after,
                                                 'rotation',(t,layer_index,stage)))
                        for q,f in zip(sites,template.before):frames[q]=f
                if stage==3:continue
                template=next(op for k,op in local if k==stage and op.kind=='cx')
                for bond in layer.bonds:
                    if tuple(frames[q] for q in bond)!=template.after:raise ValueError('CX frame mismatch')
                    forward.append(Operation(tuple(bond),template.matrix,template.before,template.after,
                                             'cx',(t,layer_index,stage)))
                    for q,f in zip(bond,template.before):frames[q]=f
                # All CNOTs have happened before this ONE graph-wide channel.
                location=(t,layer_index,stage)
                for q,rates in enumerate(single[t,layer_index,stage]):
                    if not np.any(rates) or scale==0:continue
                    diag=np.exp(-2*scale*(ANTI@rates))
                    forward.append(Operation((q,),np.diag(diag),(frames[q],),(frames[q],),'single_noise',location))
                for (i,j),rates in zip(edges,pair[t,layer_index,stage]):
                    if not np.any(rates) or scale==0:continue
                    xor=np.logical_xor(ANTI[:,None,:,None],ANTI[None,:,None,:])
                    diag=np.exp(-2*scale*np.einsum('pqab,ab->pq',xor,rates)).ravel()
                    frame=(frames[i],frames[j])
                    forward.append(Operation((i,j),np.diag(diag),frame,frame,'pair_noise',location))
            if frames!=[Q]*lattice.n_sites:raise ValueError('full matching failed to restore parity')
    operations=list(reversed(forward))
    for op in operations:check_operation(op)
    return operations


def _swap(i,j,frames,location=()):
    return Operation((i,j),SWAP,(frames[i],frames[j]),(frames[j],frames[i]),'routing_swap',location)


def route_operations(operations, n, path):
    """Route two-site factors along paths, restoring site order after each.

    `path(i,j)` returns physical-site labels along a simple connected path.
    Physical parity maps follow each SWAP; no dense nonlocal gate is built.
    """
    frames=[Q]*n;result=[]
    for original in operations:
        if tuple(frames[q] for q in original.sites)!=original.before:raise ValueError('route input frame mismatch')
        if len(original.sites)==1:
            result.append(original)
            frames[original.sites[0]]=original.after[0]
            continue
        i,j=original.sites;route=list(path(i,j))
        if len(route)<2 or route[0]!=i or route[-1]!=j or len(set(route))!=len(route):
            raise ValueError('invalid routing path')
        for a,b in zip(route[:-2],route[1:-1]):
            op=_swap(a,b,frames,original.location);result.append(op)
            frames[a],frames[b]=op.after
        op=Operation((route[-2],j),original.matrix,original.before,original.after,original.kind,original.location)
        result.append(op);frames[route[-2]],frames[j]=op.after
        for a,b in reversed(list(zip(route[:-2],route[1:-1]))):
            op=_swap(a,b,frames,original.location);result.append(op)
            frames[a],frames[b]=op.after
    if frames!=[Q]*n:raise ValueError('routed circuit did not restore the parity frame')
    for op in result:check_operation(op)
    return result
