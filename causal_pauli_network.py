"""Observable-first spacetime tensor networks for Pauli-path calculations.

No MPS or snake ordering is used. The compiled graph is the primary model;
contraction planning and exact (optionally sliced) execution are separate.
"""
from collections import Counter
from dataclasses import dataclass
from math import log10
import heapq
import time
import numpy as np
from lattice_circuit import backward_operations


def log_greedy_path(inputs):
    """Integer-rank greedy fallback for graphs whose cost overflows floats."""
    active={i:set(term) for i,term in enumerate(inputs)}
    incidence={}
    for i,term in active.items():
        for ix in term:
            incidence.setdefault(ix,set()).add(i)
    heap=[]
    def push(a,b):
        if a>b:a,b=b,a
        heapq.heappush(heap,(len(active[a]^active[b]),len(active[a]|active[b]),a,b))
    for nodes in incidence.values():
        if len(nodes)==2:
            push(*nodes)
    path=[]
    while len(active)>1:
        while heap:
            _,_,a,b=heapq.heappop(heap)
            if a in active and b in active:break
        else:
            a,b=sorted(active)[:2]
        left,right=active.pop(a),active.pop(b)
        new=len(inputs)+len(path)
        path.append((a,b))
        active[new]=left^right
        neighbors=set()
        for ix in left|right:
            incidence[ix].discard(a);incidence[ix].discard(b)
            if ix in active[new]:
                neighbors.update(incidence[ix]);incidence[ix].add(new)
        for other in neighbors:
            push(new,other)
    return path


@dataclass
class Factor:
    data: np.ndarray
    indices: tuple[str, ...]
    kind: str
    sites: tuple[int, ...] = ()


@dataclass
class ContractionPlan:
    tree: object
    signature: tuple
    optimizer: str
    seconds: float

    def estimate(self):
        cost=self.tree.contraction_cost()
        log_cost=log10(max(cost,1))
        return dict(optimizer=self.optimizer, planning_seconds=self.seconds,
                    largest_intermediate_bytes=int(self.tree.max_size())*8,
                    estimated_peak_tensor_bytes=int(self.tree.peak_size())*8,
                    estimated_flops=float(cost) if log_cost<308 else None,
                    log10_estimated_flops=log_cost,
                    slices=int(self.tree.nslices),
                    contraction_width_bits=float(self.tree.contraction_width()))


class ContractionBudgetExceeded(RuntimeError):
    """The plan exceeds an explicitly configured exact-execution budget."""


@dataclass
class PauliNetwork:
    factors: list[Factor]
    metadata: dict
    scalar: float = 1.0

    def validate(self):
        counts=Counter()
        for f in self.factors:
            if f.data.shape != (4,)*len(f.indices):
                raise ValueError('all open Pauli indices must have dimension four')
            counts.update(f.indices)
        if any(n != 2 for n in counts.values()):
            raise ValueError('closed network indices must occur exactly twice')

    def simplify_leaves(self):
        """Exactly absorb single-index boundary factors, without SVD/truncation."""
        nodes={i:f for i,f in enumerate(self.factors)}
        incidence={}
        for i,f in nodes.items():
            for label in f.indices:
                incidence.setdefault(label,set()).add(i)
        queue=[i for i,f in nodes.items() if len(f.indices)<=1]
        while queue:
            i=queue.pop()
            if i not in nodes:
                continue
            f=nodes.pop(i)
            if not f.indices:
                self.scalar *= float(f.data)
                continue
            label=f.indices[0]
            incidence[label].remove(i)
            other=next(iter(incidence.pop(label)))
            g=nodes[other]
            axis=g.indices.index(label)
            axes=list(range(g.data.ndim))
            output=[a for a in axes if a!=axis]
            data=np.einsum(g.data,axes,f.data,[axis],output,optimize=False)
            inds=tuple(x for x in g.indices if x!=label)
            nodes[other]=Factor(data,inds,g.kind,g.sites)
            if len(inds)<=1:
                queue.append(other)
        self.factors=list(nodes.values())
        self.validate()
        self.metadata['simplified_tensors']=len(self.factors)
        return self

    @property
    def signature(self):
        return tuple(f.indices for f in self.factors)

    def plan(self, optimizer='greedy', repeats=16, slice_target_bytes=None):
        """Plan only: no large intermediate tensors are allocated here."""
        if not self.factors:
            return None
        import cotengra as ctg
        if optimizer not in ('greedy','hyper','log-greedy'):
            raise ValueError('optimizer must be greedy, log-greedy or hyper')
        if repeats<1 or (slice_target_bytes is not None and slice_target_bytes<8):
            raise ValueError('invalid planner budget')
        from opt_einsum import get_symbol
        symbols={ix:get_symbol(i) for i,ix in enumerate(dict.fromkeys(ix for f in self.factors for ix in f.indices))}
        inputs=tuple(tuple(symbols[ix] for ix in f.indices) for f in self.factors)
        sizes={ix:4 for term in inputs for ix in term}
        start=time.perf_counter()
        if optimizer=='log-greedy':
            tree=ctg.ContractionTree.from_path(inputs,(),sizes,ssa_path=log_greedy_path(inputs))
        elif optimizer=='greedy':
            try:
                tree=ctg.array_contract_tree(inputs,(),sizes,optimize='greedy')
            except OverflowError:
                optimizer='log-greedy-fallback'
                tree=ctg.ContractionTree.from_path(inputs,(),sizes,ssa_path=log_greedy_path(inputs))
        else:
            opt=ctg.HyperOptimizer(methods=['greedy'],minimize='combo',max_repeats=repeats,
                                   parallel=False,optlib='random',progbar=False)
            tree=opt.search(inputs,(),sizes)
        if slice_target_bytes is not None and tree.max_size()*8 > slice_target_bytes:
            tree=tree.slice(target_size=slice_target_bytes//8,seed=0)
        return ContractionPlan(tree,self.signature,optimizer,time.perf_counter()-start)

    def contract(self, plan=None, *, max_peak_bytes=512*2**20, max_flops=2e9,
                 max_slices=4096):
        """Exact contraction, guarded before execution. Slicing is exact too.

        The peak estimate counts tensors, not all Python/BLAS/runtime overhead.
        It is not an operating-system memory limit. Budgets apply to the chosen
        plan and do not imply that every alternative plan is intractable.
        """
        if max_peak_bytes<8 or max_flops<=0 or max_slices<1:
            raise ValueError('execution budgets must be positive')
        if not self.factors:
            return dict(value=self.scalar,seconds=0.,**self.metadata)
        if plan is None:
            plan=self.plan()
        if plan.signature != self.signature:
            raise ValueError('plan does not match this network topology')
        info=plan.estimate()
        if (info['estimated_peak_tensor_bytes'] > max_peak_bytes or
            info['log10_estimated_flops'] > log10(max_flops) or info['slices'] > max_slices):
            raise ContractionBudgetExceeded(f'Exact plan exceeds budget: {info}')
        start=time.perf_counter()
        # Native einsum avoids this host's BLAS dot/matmul floating-point flags.
        value=plan.tree.contract([f.data for f in self.factors],backend='numpy',
                                prefer_einsum=True,implementation='autoray',autojit=False)
        value=float(value)*self.scalar
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite network contraction')
        return dict(value=value,seconds=time.perf_counter()-start,**info,**self.metadata)


def compile_network(pauli,lattice,*,pairs,n_steps,phi,lam_xyz=(0.,0.,0.),
                    scale=1.,unsigned=False,noise_model='independent',
                    noise_placement='gate',noise_timing='after',prune=True,simplify=True):
    """Compile a closed Pauli-transfer graph for a singlet-product initial state.

    The identity is fixed by each supported Heisenberg gate and Pauli channel,
    including elementwise-absolute local transfers. Hence untouched identity
    wires/gates can be removed exactly in both signed and unsigned networks.
    Initial singlet partners outside the cone contribute <P I>=delta(P,I).
    """
    codes=['IXYZ'.index(c) if isinstance(c,str) else c for c in pauli]
    n=lattice.n_sites
    if len(codes)!=n or any(c not in range(4) for c in codes):
        raise ValueError('one Pauli code per physical site is required')
    pairs=tuple(tuple(p) for p in pairs)
    if any(len(p)!=2 for p in pairs) or sorted(q for p in pairs for q in p)!=list(range(n)):
        raise ValueError('singlet pairs must cover every physical site exactly once')
    operations=backward_operations(lattice,n_steps,phi,lam_xyz,scale,noise_model,
                                   noise_placement,noise_timing,unsigned)
    factors=[]
    current={}
    owners={}
    next_index=0
    def wire():
        nonlocal next_index
        label=f'p{next_index}'
        next_index+=1
        return label
    for q,c in enumerate(codes):
        if c or not prune:
            ix=wire()
            current[q]=ix
            owners[ix]=len(factors)
            factors.append(Factor(np.eye(4)[c],(ix,),'observable',(q,)))
    included=0
    skipped=0
    history=[len(current)]
    gate_sites=[]
    for kind,endpoints,data in operations:
        if kind=='noise':
            for q,ix in current.items():
                if np.array_equal(data[q],np.ones(4)):
                    continue
                f=factors[owners[ix]]
                shape=[1]*f.data.ndim
                shape[f.indices.index(ix)]=4
                f.data=f.data*data[q].reshape(shape)
            continue
        a,b=endpoints
        if prune and a not in current and b not in current:
            skipped+=1
            continue
        old=[current.get(a),current.get(b)]
        new=(wire(),wire())
        selectors=(slice(None),slice(None),slice(None) if old[0] is not None else 0,
                   slice(None) if old[1] is not None else 0)
        tensor=data.reshape(4,4,4,4)[selectors]
        inds=new+tuple(ix for ix in old if ix is not None)
        idx=len(factors)
        factors.append(Factor(tensor,inds,'gate',(a,b)))
        for q,ix in zip((a,b),new):
            current[q]=ix
            owners[ix]=idx
        included+=1
        gate_sites.append((a,b))
        history.append(len(current))
    boundary=np.diag([1.,1.,1.,1.] if unsigned else [1.,-1.,-1.,-1.])
    for a,b in pairs:
        if a in current and b in current:
            factors.append(Factor(boundary,(current[a],current[b]),'singlet',(a,b)))
        elif a in current or b in current:
            q=a if a in current else b
            factors.append(Factor(np.eye(4)[0],(current[q],),'singlet_marginal',(q,)))
    metadata=dict(n_sites=n,n_steps=n_steps,unsigned=unsigned,pruned=prune,
                  active_sites=sorted(current),active_site_count=len(current),
                  included_gates=included,skipped_gates=skipped,
                  total_gates=included+skipped,compiled_tensors=len(factors),
                  active_site_history=history,gate_sites=gate_sites)
    network=PauliNetwork(factors,metadata)
    network.validate()
    return network.simplify_leaves() if simplify else network
