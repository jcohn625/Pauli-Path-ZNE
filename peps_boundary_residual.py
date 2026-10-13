"""Residual corrections for a symmetry-preserving boundary sweep.

Every forward bond truncation stores before/after boundary networks. Their
difference is contracted with one shared backward environment. Exact suffix
networks give an exact telescoping identity for the fixed input PEPS.

The retained-snapshot functions provide a CPU reference. streaming_residuals
evaluates corrections immediately and also supports GPU arrays under the
gpu_boundary context, retaining only backward row environments and scalars.
"""
import math
import time


def scalar(tn):
    return float(tn.contract(all,optimize='greedy'))


def forward_residuals(tn,nx,ny,cap,cutoff=0.):
    if cap<1 or nx<2 or ny<2:raise ValueError('positive cap and rectangle at least 2x2 required')
    if tn.exponent!=0:raise ValueError('pass tensor scaling separately; expected zero TN exponent')
    work=tn.copy();records=[];start=time.perf_counter()
    # Leave the last two-row strip for exact scalar closure. Each compression
    # is measured on the actual already-truncated forward trajectory.
    for y in range(1,ny-1):
        for x in range(nx):
            work.contract_((work.site_tag(y-1,x),work.site_tag(y,x)),which='any')
        work.canonize_row(y,sweep='left',equalize_norms=False)
        for x in range(nx-1):
            before=work.select(work.x_tag(y)).copy()
            work.compress_between(work.site_tag(y,x),work.site_tag(y,x+1),
                                  max_bond=cap,cutoff=cutoff,absorb='right',equalize_norms=False)
            after=work.select(work.x_tag(y)).copy()
            records.append(dict(row=y,bond=x,before=before,after=after))
    approximate=scalar(work)
    return dict(approximate=approximate,records=records,seconds=time.perf_counter()-start)


def backward_environments(tn,nx,ny,cap,cutoff=0.):
    if cap is None:
        # Independent exact suffix networks: no boundary approximation and no
        # use of the known exact scalar in the residual calculation.
        return {y:tn.select([tn.x_tag(j) for j in range(y+1,ny)],which='any').copy()
                for y in range(1,ny-1)}
    if cap<1:raise ValueError('positive environment cap required')
    work=tn.copy();environments={ny-2:work.select(work.x_tag(ny-1)).copy()}
    for y in range(ny-2,1,-1):
        work.contract_boundary_from_(xrange=(y,y+1),yrange=(0,nx-1),from_which='xmax',
                                     max_bond=cap,cutoff=cutoff,equalize_norms=False)
        environments[y-1]=work.select(work.x_tag(y)).copy()
    return environments


def correct_residuals(tn,nx,ny,forward,environment_cap,cutoff=0.):
    start=time.perf_counter()
    environments=backward_environments(tn,nx,ny,environment_cap,cutoff)
    setup_seconds=time.perf_counter()-start
    contributions=[]
    for record in forward['records']:
        environment=environments[record['row']]
        before=scalar(record['before'] & environment)
        after=scalar(record['after'] & environment)
        contributions.append(dict(row=record['row'],bond=record['bond'],
                                  before=before,after=after,correction=before-after))
    correction=math.fsum(r['correction'] for r in contributions)
    return dict(approximate=forward['approximate'],correction=correction,
                corrected=forward['approximate']+correction,environment_cap=environment_cap,
                contributions=contributions,environment_seconds=setup_seconds,
                seconds=time.perf_counter()-start,
                absolute_correction_sum=math.fsum(abs(r['correction']) for r in contributions))


def streaming_residuals(tn,nx,ny,cap,environment_cap,cutoff=0.,materialize=None,progress=None):
    """Correct each truncation immediately; retain only scalar residual records.

    Backward boundaries are cached once per row. Successive after/before
    overlaps are identical within a row, so reuse that scalar between splits.
    No discarded tensor or forward boundary snapshot survives a split.
    ``materialize`` can evaluate lazy device arrays without a host transfer.
    """
    if cap<1 or nx<2 or ny<2:raise ValueError('positive cap and rectangle at least 2x2 required')
    if tn.exponent!=0:raise ValueError('pass tensor scaling separately; expected zero TN exponent')
    start=time.perf_counter()
    environments=backward_environments(tn,nx,ny,environment_cap,cutoff)
    if materialize:
        for env in environments.values():materialize(env)
    environment_seconds=time.perf_counter()-start
    if progress:progress(dict(stage='environments',seconds=environment_seconds))
    work=tn.copy();contributions=[];overlap_seconds=0.
    for y in range(1,ny-1):
        row_start=time.perf_counter()
        for x in range(nx):
            work.contract_((work.site_tag(y-1,x),work.site_tag(y,x)),which='any')
        work.canonize_row(y,sweep='left',equalize_norms=False)
        environment=environments.pop(y)
        if materialize:materialize(work)
        tick=time.perf_counter()
        before=scalar(work.select(work.x_tag(y)) & environment)
        overlap_seconds+=time.perf_counter()-tick
        for x in range(nx-1):
            work.compress_between(work.site_tag(y,x),work.site_tag(y,x+1),
                                  max_bond=cap,cutoff=cutoff,absorb='right',equalize_norms=False)
            if materialize:materialize(work.select(work.x_tag(y)))
            tick=time.perf_counter()
            after=scalar(work.select(work.x_tag(y)) & environment)
            overlap_seconds+=time.perf_counter()-tick
            contributions.append(dict(row=y,bond=x,before=before,after=after,correction=before-after))
            before=after
        if progress:progress(dict(stage='forward_row',row=y,seconds=time.perf_counter()-row_start))
    # The final row's environment is the untouched bottom row, hence its last
    # overlap is already the raw scalar closure of this forward sweep.
    approximate=before if contributions else scalar(work)
    correction=math.fsum(r['correction'] for r in contributions)
    return dict(approximate=approximate,correction=correction,corrected=approximate+correction,
                environment_cap=environment_cap,contributions=contributions,
                environment_seconds=environment_seconds,overlap_seconds=overlap_seconds,
                seconds=time.perf_counter()-start,forward_snapshots_retained=0,
                absolute_correction_sum=math.fsum(abs(r['correction']) for r in contributions))
