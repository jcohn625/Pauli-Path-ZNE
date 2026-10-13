"""Experimental 2D Pauli-coefficient PEPS: simple update + boundary compression.

This is an approximation with TWO independent convergence parameters: the
spatial PEPS bond cap and the contraction boundary cap. Neither error is certified.
Opt into float64 SciPy GEMM for boundary contractions with boundary_backend='blas'.
This changes arithmetic order, but not the sweep or truncation algorithm.
"""
import os
from pathlib import Path
from contextlib import ExitStack
import time
import numpy as np
from lattice_circuit import backward_operations
from causal_pauli_network import compile_network


def _evaluate_peps_impl(pauli,lattice,*,nx,ny,pairs,n_steps,phi,lam_xyz=(0.,0.,0.),scale=1.,
                  unsigned=False,bond_cap=2,boundary_cap=16,cutoff=1e-12,
                  noise_model='independent',noise_placement='gate',noise_timing='after',
                  symmetry='none',gate_backend='quimb',capture_layer=None,
                  capture_boundary=None,boundary_backend='einsum',jacobi_sweeps=16):
    if boundary_backend not in ('einsum','blas','metal-jacobi'):
        raise ValueError('unknown boundary backend')
    if boundary_backend=='metal-jacobi' and (symmetry!='z2z2' or jacobi_sweeps<1):
        raise ValueError('Metal Jacobi requires z2z2 symmetry and positive sweeps')
    if gate_backend not in ('quimb','batched-cpu','metal'):
        raise ValueError('unknown gate backend')
    if gate_backend!='quimb' and symmetry!='z2z2':
        raise ValueError('batched gate backends require z2z2 symmetry')
    if symmetry not in ('none', 'z2z2'):
        raise ValueError('symmetry must be none or z2z2')
    if nx*ny!=lattice.n_sites or nx<1 or ny<1 or bond_cap<1 or boundary_cap<1:
        raise ValueError('invalid rectangle or bond caps')
    pairs=tuple(tuple(p) for p in pairs)
    def coord(q):return divmod(q,nx)
    def adjacent(a,b):
        ay,ax=coord(a);by,bx=coord(b)
        return abs(ay-by)+abs(ax-bx)==1
    if any(not adjacent(a,b) for layer in lattice.layers for a,b in layer.bonds):
        raise ValueError('PEPS backend requires nearest-neighbor rectangle gates')
    if any(len(p)!=2 or not adjacent(*p) for p in pairs):
        raise ValueError('PEPS closure currently requires nearest-neighbor singlet pairs')
    # Reuse compiler validation and causal support semantics, independent of MPS.
    compile_network(pauli,lattice,pairs=pairs,n_steps=n_steps,phi=phi,
                            lam_xyz=lam_xyz,scale=scale,unsigned=unsigned,
                            noise_model=noise_model,noise_placement=noise_placement,
                            noise_timing=noise_timing)
    os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parent/'.numba_cache'))
    import quimb.tensor as qtn
    codes=['IXYZ'.index(c) if isinstance(c,str) else c for c in pauli]
    start=time.perf_counter()
    symmetric = symmetry == 'z2z2'
    if symmetric:
        import pauli_peps_symmetry as sym
        charge = sym.total_charge(codes)
        # The singlet functional is neutral, in both signed and unsigned modes.
        if charge != (0, 0):
            return dict(value=0.,sign=0.,log_abs_value=None,log_scale=0.,
                        seconds=time.perf_counter()-start,evolve_seconds=0.,closure_seconds=0.,
                        boundary_seconds=0.,final_contract_seconds=0.,boundary_backend=boundary_backend,
                        boundary_gpu_stats=None,
                        jacobi_sweeps=jacobi_sweeps if boundary_backend=='metal-jacobi' else None,
                        bond_cap=bond_cap,boundary_cap=boundary_cap,included_gates=0,
                        active_site_count=sum(c != 0 for c in codes),n_sites=lattice.n_sites,
                        n_steps=n_steps,unsigned=unsigned,method='peps-simple-update',
                        symmetry=symmetry,observable_charge=charge,symmetry_exact_zero=True,
                        gate_backend=gate_backend,layer_stats=None,
                        symmetry_stats=None,boundary_symmetry_stats=None,
                        accuracy_certified=True,cutoff=cutoff,approximation='exact symmetry selection rule')
    peps=qtn.PEPS.product_state({coord(q):np.eye(4)[c] for q,c in enumerate(codes)})
    if symmetric:
        sym.convert_product(peps, codes, nx)
    gate_cache = {}
    gauges={}
    active={q for q,c in enumerate(codes) if c}
    operations=backward_operations(lattice,n_steps,phi,lam_xyz,scale,noise_model,
                                   noise_placement,noise_timing,unsigned)
    engine=None
    if gate_backend!='quimb':
        from peps_metal_layer import LayerEngine
        engine=LayerEngine('metal' if gate_backend=='metal' else 'cpu')
    if engine is not None or capture_layer is not None:
        if not symmetric:raise ValueError('layer capture requires z2z2 symmetry')
        grouped=[];pending=[]
        for operation in operations:
            if operation[0]=='gate':pending.append(operation)
            else:
                if pending:grouped.append(('gate_batch',pending,None));pending=[]
                grouped.append(operation)
        if pending:grouped.append(('gate_batch',pending,None))
        operations=grouped
    last_batch=max((i for i,op in enumerate(operations) if op[0]=='gate_batch'),default=-1)
    included=0
    log_scale=0.0
    for op_index,(kind,endpoints,data) in enumerate(operations):
        if kind=='noise':
            for q in active:
                if not np.array_equal(data[q],np.ones(4)):
                    diagonal = sym.diagonal(data[q]) if symmetric else data[q]
                    peps[coord(q)].multiply_index_diagonal_(peps.site_ind(*coord(q)),diagonal)
            continue
        if kind=='gate_batch':
            gates=[]
            for _,(a,b),dense_gate in endpoints:
                if a not in active and b not in active:continue
                key=dense_gate.tobytes()
                if key not in gate_cache:gate_cache[key]=sym.gate(dense_gate)
                gates.append((a,b,gate_cache[key]))
                active.update((a,b))
            if gates:
                if capture_layer is not None and op_index==last_batch:
                    capture_layer(peps,gauges,gates,nx,bond_cap,cutoff)
                if engine is not None:
                    log_scale+=engine.apply(peps,gauges,gates,nx,bond_cap,cutoff)
                else:
                    for a,b,g in gates:
                        info={}
                        peps.gate_simple_(g,(coord(a),coord(b)),gauges=gauges,renorm=False,
                            info=info,max_bond=bond_cap,cutoff=cutoff)
                        key,s=next(iter(info.items()));norm=float(s.norm())
                        if norm<=0 or not np.isfinite(norm):raise FloatingPointError('invalid singular norm')
                        gauges[key[1]]=s/norm;log_scale+=np.log(norm)
                included+=len(gates)
            continue
        a,b=endpoints
        if a not in active and b not in active:continue
        info={}
        if symmetric:
            key = data.tobytes()
            if key not in gate_cache:
                gate_cache[key] = sym.gate(data)
            data = gate_cache[key]
        peps.gate_simple_(data,(coord(a),coord(b)),gauges=gauges,renorm=False,info=info,
                          max_bond=bond_cap,cutoff=cutoff)
        key,s=next(iter(info.items()))
        norm=float(s.norm()) if symmetric else float(np.sqrt(np.sum(s*s)))
        if not np.isfinite(norm) or norm<=0:
            raise FloatingPointError('invalid simple-update normalization')
        gauges[key[1]]=s/norm
        log_scale+=np.log(norm)
        active.update((a,b));included+=1
    peps.gauge_simple_insert(gauges)
    symmetry_stats = sym.statistics(peps) if symmetric else None
    evolve_seconds=time.perf_counter()-start
    # Each singlet functional is diagonal in Pauli labels. Identify its two
    # physical legs and absorb its signs, then fuse this extra spatial bond.
    signs=np.array([1.,1.,1.,1.] if unsigned else [1.,-1.,-1.,-1.])
    for a,b in pairs:
        ia,ib=peps.site_ind(*coord(a)),peps.site_ind(*coord(b))
        peps[coord(a)].multiply_index_diagonal_(ia,sym.diagonal(signs) if symmetric else signs)
        if symmetric:
            sym.flip_physical_dual(peps[coord(b)], ib)
        peps.reindex_({ib:ia})
    peps.fuse_multibonds_()
    closure_seconds=time.perf_counter()-start-evolve_seconds
    if capture_boundary is not None:
        capture_boundary(peps,log_scale)
    boundary_start=time.perf_counter()
    from peps_boundary import boundary_backend as boundary_context
    boundary_engine=None
    with ExitStack() as stack:
        if boundary_backend=='metal-jacobi':
            from peps_gpu_boundary import gpu_boundary,to_gpu
            boundary_engine=stack.enter_context(gpu_boundary(jacobi_sweeps))
            peps=to_gpu(peps,boundary_engine.mx)
        else:
            stack.enter_context(boundary_context(boundary_backend))
        boundary=peps.contract_boundary(max_bond=boundary_cap,cutoff=cutoff,
                                       final_contract=False,equalize_norms=1.0)
        boundary_seconds=time.perf_counter()-boundary_start
        boundary_stats = sym.statistics(boundary) if symmetric else None
        final_start=time.perf_counter()
        tree=boundary.contraction_tree(optimize='greedy')
        mantissa,exponent=tree.contract(boundary.arrays,strip_exponent=True,check_zero=True)
        exponent+=boundary.exponent
        mantissa=float(mantissa);exponent=float(exponent)
        final_contract_seconds=time.perf_counter()-final_start
    mantissa=float(mantissa)
    if not np.isfinite(mantissa):raise FloatingPointError('nonfinite PEPS contraction')
    log_abs=None if mantissa==0 else float(np.log(abs(mantissa))+exponent*np.log(10)+log_scale)
    sign=float(np.sign(mantissa))
    value=0. if sign==0 else (sign*float(np.exp(log_abs)) if log_abs<np.log(np.finfo(float).max) else None)
    return dict(value=value,sign=sign,log_abs_value=log_abs,log_scale=float(log_scale),seconds=time.perf_counter()-start,evolve_seconds=evolve_seconds,
                closure_seconds=closure_seconds,boundary_seconds=boundary_seconds,
                boundary_backend=boundary_backend,
                boundary_gpu_stats=boundary_engine.stats if boundary_engine is not None else None,
                jacobi_sweeps=jacobi_sweeps if boundary_backend=='metal-jacobi' else None,
                final_contract_seconds=final_contract_seconds,bond_cap=bond_cap,boundary_cap=boundary_cap,
                included_gates=included,active_site_count=len(active),n_sites=lattice.n_sites,
                n_steps=n_steps,unsigned=unsigned,method='peps-simple-update',
                symmetry=symmetry, symmetry_stats=symmetry_stats, boundary_symmetry_stats=boundary_stats,
                gate_backend=gate_backend,layer_stats=engine.stats if engine else None,
                observable_charge=charge if symmetric else None,symmetry_exact_zero=False,
                accuracy_certified=False,cutoff=cutoff,
                approximation='spatial simple update and boundary-MPS contraction')


def _safe_tensordot(a,b,axes=2):
    if hasattr(a, 'blocks') or hasattr(b, 'blocks'):
        from autoray import do
        return do('tensordot', a, b, axes=axes)
    if type(a).__module__.startswith('mlx') or type(b).__module__.startswith('mlx'):
        import mlx.core as mx
        return mx.tensordot(a,b,axes=axes)
    from peps_boundary import use_blas, blas_tensordot
    if use_blas():
        return blas_tensordot(a,b,axes)
    if isinstance(axes,int):
        left=list(range(a.ndim-axes,a.ndim));right=list(range(axes))
    else:
        left,right=axes
        left=[i%a.ndim for i in left];right=[i%b.ndim for i in right]
    ai=list(range(a.ndim));bi=list(range(a.ndim,a.ndim+b.ndim))
    for i,j in zip(left,right):bi[j]=ai[i]
    out=[ai[i] for i in range(a.ndim) if i not in left]+[bi[i] for i in range(b.ndim) if i not in right]
    return np.einsum(a,ai,b,bi,out,optimize=False)


def _safe_einsum(equation, *arrays):
    if any(hasattr(a, 'blocks') for a in arrays):
        from autoray import do
        return do('einsum', equation, *arrays)
    if any(type(a).__module__.startswith('mlx') for a in arrays):
        import mlx.core as mx
        return mx.einsum(equation,*arrays)
    return np.einsum(equation, *arrays, optimize=False)


def evaluate_peps(*args,**kwargs):
    # Scoped cotengra implementation avoids host BLAS FP flags without hiding
    # warnings. The library setting is process-global; run workers serially or
    # in separate processes, not concurrent threads.
    from cotengra.contract import default_implementation
    from autoray import get_lib_fn, register_function
    original=get_lib_fn('numpy','tensordot')
    register_function('numpy','tensordot',_safe_tensordot)
    try:
        # Keep one dispatching implementation for both backends: cotengra's
        # expression cache does not include the ambient default implementation.
        # A cached NumPy-only callable would otherwise receive block arrays
        # when users switch backends within the same Python process.
        with default_implementation((_safe_einsum, _safe_tensordot)):
            return _evaluate_peps_impl(*args,**kwargs)
    finally:
        register_function('numpy','tensordot',original)
