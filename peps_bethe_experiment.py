"""Experimental finite boundary-MPS Bethe estimator for a fixed signed PEPS.

Uses two independently compressed boundary sweeps. This evaluates a Bethe
functional; it does not solve the variational MP-BP fixed-point equations.
No positivity, entrywise absolute values, or norm-network replacement is used.
"""
import math
import time


def _scalar(tn):
    return float(tn.contract(all,optimize='greedy'))


def boundary_messages(tn,nx,ny,cap):
    if tn.exponent!=0:raise ValueError('pass external tensor scaling separately')
    top={0:tn.select(tn.x_tag(0)).copy()};bottom={ny-1:tn.select(tn.x_tag(ny-1)).copy()}
    work=tn.copy()
    for y in range(1,ny-1):
        for x in range(nx):
            work.contract_((work.site_tag(y-1,x),work.site_tag(y,x)),which='any')
        work.canonize_row(y,sweep='left',equalize_norms=False)
        for x in range(nx-1):
            work.compress_between(work.site_tag(y,x),work.site_tag(y,x+1),
                max_bond=cap,cutoff=0.,absorb='right',equalize_norms=False)
        top[y]=work.select(work.x_tag(y)).copy()
    work=tn.copy()
    for y in range(ny-2,0,-1):
        work.contract_boundary_from_(xrange=(y,y+1),yrange=(0,nx-1),from_which='xmax',
            max_bond=cap,cutoff=0.,equalize_norms=False)
        bottom[y]=work.select(work.x_tag(y)).copy()
    return top,bottom


def bethe_contract(tn,nx,ny,cap):
    if nx<2 or ny<2 or cap<1:raise ValueError('positive cap and rectangle required')
    start=time.perf_counter();top,bottom=boundary_messages(tn,nx,ny,cap)
    numerators=[];denominators=[]
    for y in range(ny):
        net=tn.select(tn.x_tag(y)).copy()
        if y>0:net=net & top[y-1]
        if y<ny-1:net=net & bottom[y+1]
        numerators.append(_scalar(net))
    for y in range(ny-1):denominators.append(_scalar(top[y] & bottom[y+1]))
    if any(not math.isfinite(x) or x==0 for x in numerators+denominators):
        return dict(value=None,status='zero_or_nonfinite_local_overlap',numerators=numerators,
                    denominators=denominators,seconds=time.perf_counter()-start)
    sign=math.prod(1 if x>0 else -1 for x in numerators+denominators)
    logarithm=math.fsum(math.log(abs(x)) for x in numerators)-math.fsum(math.log(abs(x)) for x in denominators)
    return dict(value=sign*math.exp(logarithm),status='ok',numerators=numerators,denominators=denominators,
                seconds=time.perf_counter()-start,forward=_scalar(top[ny-2] & bottom[ny-1]))


def row_residual_contract(tn, nx, ny, cap, environment_cap):
    """Same truncations/guides as streaming_residuals; telescope within rows.

    The total correction only needs the first and last overlap in a row.
    This intentionally returns row records, not individual bond diagnostics.
    """
    from peps_boundary_residual import backward_environments
    if nx < 2 or ny < 2 or cap < 1:
        raise ValueError('positive cap and rectangle at least 2x2 required')
    if tn.exponent != 0:
        raise ValueError('pass external tensor scaling separately')
    start = time.perf_counter()
    environments = backward_environments(tn, nx, ny, environment_cap)
    work = tn.copy()
    records = []
    for y in range(1, ny-1):
        for x in range(nx):
            work.contract_((work.site_tag(y-1,x),work.site_tag(y,x)),which='any')
        work.canonize_row(y,sweep='left',equalize_norms=False)
        environment = environments.pop(y)
        before = _scalar(work.select(work.x_tag(y)) & environment)
        for x in range(nx-1):
            work.compress_between(work.site_tag(y,x),work.site_tag(y,x+1),
                max_bond=cap,cutoff=0.,absorb='right',equalize_norms=False)
        after = _scalar(work.select(work.x_tag(y)) & environment)
        records.append(dict(row=y,before=before,after=after,correction=before-after))
    approximate = after if records else _scalar(work)
    correction = math.fsum(r['correction'] for r in records)
    return dict(approximate=approximate,corrected=approximate+correction,
        correction=correction,contributions=records,granularity='row',
        overlap_count=2*len(records),seconds=time.perf_counter()-start)
