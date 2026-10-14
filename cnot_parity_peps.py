"""Apply elementary transfers to PEPS while transporting physical parity maps."""
import numpy as np
from cnot_parity import Q,compile_operations,route_operations,fuse_operations


def _charge_map(values):
    return tuple((int(x)&1,int(x)>>1) for x in values)


def sparse_gate(op):
    from pauli_peps_symmetry import symmray
    n=len(op.sites)
    return symmray().Z2Z2Array.from_dense(op.matrix.reshape((4,)*(2*n)),
        [_charge_map(x) for x in op.after+op.before],
        [False]*n+[True]*n,charge=(0,0),invalid_sectors='raise')


def evolve(peps,gauges,lattice,*,nx,steps,phi,scale,unsigned,noise,cap,cutoff,engine=None):
    from pauli_peps_symmetry import statistics
    from peps_metal_layer import DiagonalGate,LayerEngine
    noise_engine=engine if engine is not None else LayerEngine('cpu',direct_blocks=True)
    noise_engine.direct_blocks=True
    def coord(q):return divmod(q,nx)
    def path(i,j):
        y,x=coord(i);yy,xx=coord(j);result=[i]
        while x!=xx:x+=1 if x<xx else -1;result.append(y*nx+x)
        while y!=yy:y+=1 if y<yy else -1;result.append(y*nx+x)
        return result
    operations=compile_operations(lattice,steps,phi,scale=scale,unsigned=unsigned,**noise)
    operations=route_operations(operations,lattice.n_sites,path)
    elementary_count=len(operations)
    cx_count=sum(op.kind=='cx' for op in operations)
    operations=fuse_operations(operations)
    frames=[Q]*lattice.n_sites;log_scale=0.;cache={};batch=[];occupied=set()
    def flush():
        nonlocal log_scale
        if batch:
            log_scale+=engine.apply(peps,gauges,batch,nx,cap,cutoff)
            batch.clear();occupied.clear()
    for op in operations:
        if engine is not None and (len(op.sites)==1 or occupied.intersection(op.sites)):flush()
        if tuple(frames[q] for q in op.sites)!=op.before:raise ValueError('PEPS frame mismatch')
        key=(op.matrix.tobytes(),op.before,op.after)
        if key not in cache:
            diagonal=(len(op.sites)==2 and op.before==op.after and
                      np.count_nonzero(op.matrix-np.diag(np.diag(op.matrix)))==0)
            cache[key]=DiagonalGate(np.diag(op.matrix).copy(),op.before) if diagonal else sparse_gate(op)
        gate=cache[key]
        if len(op.sites)==1:
            peps[coord(op.sites[0])].gate_(gate,peps.site_ind(*coord(op.sites[0])))
        elif engine is not None:
            i,j=op.sites
            batch.append((i,j,gate));occupied.update(op.sites)
        elif isinstance(gate,DiagonalGate):
            log_scale+=noise_engine.apply(peps,gauges,[(op.sites[0],op.sites[1],gate)],nx,cap,cutoff)
        else:
            info={}
            peps.gate_simple_(gate,tuple(coord(q) for q in op.sites),gauges=gauges,
                              renorm=False,info=info,max_bond=cap,cutoff=cutoff)
            (_,bond),s=next(iter(info.items()));norm=float(s.norm())
            if not np.isfinite(norm) or norm<=0:raise FloatingPointError('nonfinite/zero PEPS singular norm')
            gauges[bond]=s/norm;log_scale+=np.log(norm)
        for q,f in zip(op.sites,op.after):frames[q]=f
    if engine is not None:flush()
    if frames!=[Q]*lattice.n_sites:raise ValueError('final PEPS parity frame not restored')
    return log_scale,dict(method='cnot-layer-transported-parity-peps',operations=len(operations),
        diagonal_updates=noise_engine.stats['diagonal_gates'],
        elementary_operations=elementary_count,cnot_gates=cx_count,routing_swaps=sum(op.kind=='routing_swap' for op in operations),
        final_standard_frame=True,noise_placement='after each nonempty CNOT layer',
        **statistics(peps))
