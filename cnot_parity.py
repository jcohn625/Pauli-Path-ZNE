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


@dataclass(frozen=True)
class CompactOperation:
    """Diagonal vector or signed permutation; dense PTM is materialized on demand."""
    sites: tuple
    values: np.ndarray
    before: tuple
    after: tuple
    kind: str
    location: tuple = ()
    columns: tuple | None = None

    @property
    def diagonal(self):return self.values if self.columns is None else None

    @property
    def matrix(self):
        if self.columns is None:return np.diag(self.values)
        result=np.zeros((len(self.values),len(self.values)))
        result[np.arange(len(self.values)),self.columns]=self.values
        return result

    def adjoint(self):
        if self.columns is None:
            return CompactOperation(self.sites,self.values,self.after,self.before,self.kind,self.location)
        inverse=np.argsort(self.columns)
        return CompactOperation(self.sites,self.values[inverse],self.after,self.before,
                                self.kind,self.location,tuple(inverse))


def compact_operation(op):
    if isinstance(op,CompactOperation):return op
    matrix=op.matrix;diagonal=np.diag(matrix)
    if np.count_nonzero(matrix)==np.count_nonzero(diagonal):
        values=diagonal.copy();values.flags.writeable=False
        return CompactOperation(op.sites,values,op.before,op.after,op.kind,op.location)
    columns=np.argmax(abs(matrix),axis=1)
    if np.count_nonzero(matrix)==len(matrix) and len(set(columns))==len(matrix):
        values=matrix[np.arange(len(matrix)),columns].copy();values.flags.writeable=False
        return CompactOperation(op.sites,values,op.before,op.after,op.kind,op.location,tuple(columns))
    return op


def check_operation(op, tolerance=2e-12):
    if op.kind=='noise_mpo':
        if op.before!=op.after:raise ValueError('diagonal noise MPO changed frames')
        return 0.
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


def _compile_operations_uncached(lattice, steps, phi, *, singles=(.001,.002,.003), pair_rates=0.,
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


@lru_cache(maxsize=8)
def _operation_template(lattice, steps, phi, edges, unsigned):
    # Rates and scale do not change circuit topology or transported charges.
    ops=_compile_operations_uncached(lattice,steps,phi,singles=1.,pair_rates=1.,
                                     edges=edges,unsigned=unsigned)
    for op in ops:op.matrix.flags.writeable=False
    return tuple(compact_operation(op) for op in ops)


def compile_operations(lattice, steps, phi, *, singles=(.001,.002,.003), pair_rates=0.,
                       edges=None, scale=1., unsigned=False):
    """Instantiate a cached topology with validated, possibly scheduled rates."""
    if not isinstance(steps,int) or steps<0 or not np.isfinite(scale) or scale<0:
        raise ValueError('steps must be a nonnegative integer; scale must be finite and nonnegative')
    if not np.isfinite(phi):raise ValueError('phi must be finite')
    single,pair,edges=normalize_noise(lattice,steps,singles,pair_rates,edges)
    edge_index={e:k for k,e in enumerate(edges)};result=[];matrices={}
    xor=np.logical_xor(ANTI[:,None,:,None],ANTI[None,:,None,:])
    for op in _operation_template(lattice,steps,float(phi),edges,unsigned):
        if op.kind not in ('single_noise','pair_noise'):
            result.append(op);continue
        if scale==0:continue
        t,li,k=op.location
        rates=single[t,li,k,op.sites[0]] if op.kind=='single_noise' else pair[t,li,k,edge_index[op.sites]]
        if not np.any(rates):continue
        key=(op.kind,rates.tobytes())
        if key not in matrices:
            exponent=ANTI@rates if op.kind=='single_noise' else np.einsum('pqab,ab->pq',xor,rates).ravel()
            values=np.exp(-2*scale*exponent);values.flags.writeable=False
            matrices[key]=values
        result.append(CompactOperation(op.sites,matrices[key],op.before,op.after,op.kind,op.location))
    return result


def _swap(i,j,frames,location=()):
    return CompactOperation((i,j),np.ones(16),(frames[i],frames[j]),(frames[j],frames[i]),'routing_swap',location,tuple(np.argmax(SWAP,axis=1)))


def _route_noise_group(group, initial_frames, path):
    """Retain site permutations across one diagonal noise insertion, then undo."""
    frames=list(initial_frames);positions=list(range(len(frames)));labels=list(positions)
    swaps=[];result=[]
    def swap(a,b,location):
        op=_swap(a,b,frames,location);result.append(op)
        frames[a],frames[b]=op.after
        labels[a],labels[b]=labels[b],labels[a]
        positions[labels[a]]=a;positions[labels[b]]=b
    for original in group:
        i,j=(positions[q] for q in original.sites)
        if tuple(frames[q] for q in (i,j))!=original.before:raise ValueError('noise routing frame mismatch')
        route=list(path(i,j))
        if len(route)<2 or route[0]!=i or route[-1]!=j or len(set(route))!=len(route):raise ValueError('invalid routing path')
        for a,b in zip(route[:-2],route[1:-1]):
            swap(a,b,original.location);swaps.append((a,b))
        result.append(compact_operation(Operation((route[-2],j),original.matrix,original.before,original.after,original.kind,original.location)))
    for a,b in reversed(swaps):swap(a,b,group[-1].location)
    if frames!=initial_frames:raise ValueError('noise routing did not restore frames')
    return result


def route_operations(operations, n, path):
    """Route two-site factors, sharing SWAPs within diagonal noise insertions.

    `path(i,j)` returns physical-site labels along a simple connected path.
    Physical parity maps follow each SWAP; no dense nonlocal gate is built.
    Shared routing is used only if it needs fewer SWAPs than separate routing.
    """
    frames=[Q]*n;result=[];index=0;next_group=0
    while index<len(operations):
        original=operations[index]
        if original.kind=='pair_noise' and index>=next_group:
            end=index+1
            while end<len(operations) and operations[end].kind=='pair_noise' and operations[end].location==original.location:end+=1
            group=operations[index:end]
            next_group=end
            independent=sum(2*(len(list(path(*op.sites)))-2)+1 for op in group)
            shared=_route_noise_group(group,frames,path) if independent>len(group) else group
            if len(shared)<independent:
                result.extend(shared);index=end;continue
        index+=1
        if tuple(frames[q] for q in original.sites)!=original.before:raise ValueError('route input frame mismatch')
        if original.kind=='noise_mpo':
            result.append(original);continue
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
        result.append(compact_operation(op));frames[route[-2]],frames[j]=op.after
        for a,b in reversed(list(zip(route[:-2],route[1:-1]))):
            op=_swap(a,b,frames,original.location);result.append(op)
            frames[a],frames[b]=op.after
    if frames!=[Q]*n:raise ValueError('routed circuit did not restore the parity frame')
    for op in result:check_operation(op)
    return result


def fuse_operations(operations):
    """Fuse transfers on at most two sites, commuting only disjoint supports.

    Work on already unsigned transfers for Q. Never cross matching boundaries
    or routing SWAPs, so residual locations and routing diagnostics stay useful.
    """
    pending=[];result=[];location=None
    def embed(op,sites):
        if op.sites==sites:return op.matrix
        if len(op.sites)==1:
            return np.kron(op.matrix,np.eye(4)) if op.sites[0]==sites[0] else np.kron(np.eye(4),op.matrix)
        return op.matrix.reshape(4,4,4,4).transpose(1,0,3,2).reshape(16,16)
    for op in operations:
        boundary=op.location[:2]
        if boundary!=location or op.kind in ('routing_swap','noise_mpo'):
            result.extend(pending);pending=[];location=boundary
        if op.kind in ('routing_swap','noise_mpo'):result.append(op);continue
        hits=[p for p in pending if set(p.sites)&set(op.sites)]
        sites=tuple(sorted(set(op.sites).union(*(set(p.sites) for p in hits))))
        if len(sites)>2:
            result.extend(hits)
            pending=[p for p in pending if not set(p.sites)&set(op.sites)]
            pending.append(op);continue
        if not hits:pending.append(op);continue
        before=dict(zip(op.sites,op.before));after={}
        matrix=np.eye(4**len(sites))
        for p in hits:
            before.update(zip(p.sites,p.before));after.update(zip(p.sites,p.after))
            matrix=embed(p,sites)@matrix
        after.update(zip(op.sites,op.after))
        matrix=embed(op,sites)@matrix
        pending=[p for p in pending if not set(p.sites)&set(op.sites)]
        fused=Operation(sites,matrix,tuple(before[q] for q in sites),tuple(after[q] for q in sites),'fused',op.location)
        check_operation(fused);pending.append(compact_operation(fused))
    result.extend(pending)
    return result


def sweep_operations(operations):
    """Order consecutive disjoint supports to reduce canonical-center travel."""
    result=[];batch=[];occupied=set();center=0
    def flush():
        nonlocal center
        if not batch:return
        ordered=sorted(batch,key=lambda op:min(op.sites))
        if abs(center-min(ordered[-1].sites))<abs(center-min(ordered[0].sites)):ordered.reverse()
        result.extend(ordered);center=max(ordered[-1].sites)
    for op in operations:
        if occupied.intersection(op.sites):flush();batch=[];occupied=set()
        batch.append(op);occupied.update(op.sites)
    flush()
    return result
