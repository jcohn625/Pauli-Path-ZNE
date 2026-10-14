"""Research-only signed finite-strip environment projector.

At each forward bond cut contract the row with a fixed backward guide, leaving
that bond open. SVD of this generally non-Hermitian E selects an orthogonal
projector P = U_k U_k^H. Insert P without trace renormalization. Singular values
compete globally across Z2 x Z2 sectors. This is a full-bond environment pilot,
not the reduced-core RTM sweep or a stationary message-passing algorithm.
The repeated open-bond contractions deliberately favor simplicity over speed.
The alternative method="rtm" uses cached KQ prefixes in a right-to-left row
sweep; neither method iterates the backward guide or implements a full tree.
"""
import math
import time
from quimb.tensor.tensor_core import rand_uuid
from peps_boundary_residual import scalar, backward_environments


def compress_guided_bond(work, environment, y, x, cap, cutoff=0.):
    left=work[work.site_tag(y,x)]
    right=work[work.site_tag(y,x+1)]
    bond,=left.bonds(right)
    if left.ind_size(bond)<=cap and cutoff==0.:
        return
    opened=work.select(work.x_tag(y)) & environment
    lc,rc=rand_uuid(),rand_uuid()
    opened[work.site_tag(y,x)].reindex_({bond:lc})
    opened[work.site_tag(y,x+1)].reindex_({bond:rc})
    matrix=opened.contract(all,optimize='greedy',output_inds=[lc,rc])
    u,_,_=matrix.split(left_inds=[lc],method='svd',max_bond=cap,
                       cutoff=cutoff,absorb=None,get='tensors')
    # Contraction is bilinear (no conjugate on the guide). Both gate matrices
    # have shape (retained, old); conjugation also reverses symmray dual flags.
    left.gate_(u.data.conj().transpose((1,0)),bond)
    right.gate_(u.data.transpose((1,0)),bond)


def guided_residuals(tn,nx,ny,cap,environment_cap,cutoff=0.,check=False,method="projector"):
    if cap<1 or nx<2 or ny<2:
        raise ValueError('positive cap and rectangle at least 2x2 required')
    if tn.exponent!=0:
        raise ValueError('pass tensor scaling separately; expected zero TN exponent')
    if method not in ("projector", "rtm"):
        raise ValueError("method must be projector or rtm")
    start=time.perf_counter()
    environments=backward_environments(tn,nx,ny,environment_cap,cutoff)
    environment_seconds=time.perf_counter()-start
    work=tn.copy();contributions=[];compression_seconds=0.
    for y in range(1,ny-1):
        for x in range(nx):
            work.contract_((work.site_tag(y-1,x),work.site_tag(y,x)),which='any')
        work.canonize_row(y,sweep='left',equalize_norms=False)
        env=environments.pop(y)
        before=scalar(work.select(work.x_tag(y)) & env)
        tick=time.perf_counter()
        if method == "rtm":
            compress_rtm_row(work,env,y,nx,cap,cutoff)
        else:
            for x in range(nx-1):
                compress_guided_bond(work,env,y,x,cap,cutoff)
        compression_seconds+=time.perf_counter()-tick
        if check:
            for tensor in work:tensor.data.check()
        after=scalar(work.select(work.x_tag(y)) & env)
        contributions.append(dict(row=y,before=before,after=after,correction=before-after))
    approximate=after if contributions else scalar(work)
    correction=math.fsum(r['correction'] for r in contributions)
    return dict(approximate=approximate,corrected=approximate+correction,
                correction=correction,contributions=contributions,granularity='row',
                environment_cap=environment_cap,environment_seconds=environment_seconds,
                compression_seconds=compression_seconds,seconds=time.perf_counter()-start,
                absolute_correction_sum=math.fsum(abs(r['correction']) for r in contributions))


def compress_rtm_row(work, environment, y, nx, cap, cutoff=0., environment_row=None):
    """Finite right-to-left KQ pilot, using cached bilinear left prefixes.

    Left-canonical F and E define K at every cut. For local forward core Q,
    retain right singular vectors Vh of KQ, emit Vh, and absorb Q V into the
    preceding site. No inverse, Gram matrix, or scalar fitting is used. Only
    one finite row is updated; the backward messages are not iterated.
    """
    work.canonize_row(y,sweep='right',equalize_norms=False)
    env=environment.copy()
    # A compressed suffix has one tensor per column. Exact uncompressed
    # suffixes are useful as correction references, but aren't MPS guides.
    if env.num_tensors!=nx:
        raise ValueError('RTM compression requires an MPS guide (finite environment_cap)')
    env.canonize_row(y+1 if environment_row is None else environment_row,
                     sweep='right',equalize_norms=False)
    prefixes={};prefix=None
    for x in range(nx-1):
        pair=work[work.site_tag(y,x)] & env[env.y_tag(x)]
        prefix=pair.contract(all,optimize='greedy') if prefix is None else (prefix & pair).contract(all,optimize='greedy')
        prefixes[x+1]=prefix
    for x in range(nx-1,0,-1):
        current=work[work.site_tag(y,x)];previous=work[work.site_tag(y,x-1)]
        bond,=previous.bonds(current)
        if current.ind_size(bond)<=cap and cutoff==0.:continue
        right_inds=[i for i in current.inds if i!=bond]
        weighted=prefixes[x] @ current
        left_inds=[i for i in weighted.inds if i not in right_inds]
        _,_,vh=weighted.split(left_inds=left_inds,right_inds=right_inds,method='svd',
                               absorb=None,cutoff=cutoff,max_bond=cap,get='tensors')
        core=current @ vh.conj()
        updated=previous @ core
        current.modify(data=vh.data,inds=vh.inds)
        previous.modify(data=updated.data,inds=updated.inds)
