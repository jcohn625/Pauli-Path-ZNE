"""Experimental finite bidirectional RTM boundary messages for signed PEPS.

Every message update absorbs an original row into a newly computed incoming
message, using the opposite directed message as a fixed RTM guide. Messages
are neither averaged nor normalized to a fitted scalar. Cut agreement and
iteration stability are diagnostics only, not certified error bounds.
"""
import math
import time
from peps_bethe_experiment import boundary_messages
from peps_environment_experiment import compress_rtm_row
from peps_boundary_residual import scalar, streaming_residuals


def _cut_diagnostics(top,bottom,ny):
    values=[scalar(top[y] & bottom[y+1]) for y in range(ny-1)]
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError('nonfinite boundary-message overlap')
    largest=max(abs(v) for v in values)
    return dict(cut_values=values,mean=math.fsum(values)/len(values),
                relative_spread=(max(values)-min(values))/largest if largest else 0.)


def _updated_message(tn,incoming,guide,y,previous,nx,cap,cutoff,check):
    work=incoming & tn.select(tn.x_tag(y))
    for x in range(nx):
        work.contract_((work.site_tag(previous,x),work.site_tag(y,x)),which='any')
    opposite=y+1 if previous<y else y-1
    compress_rtm_row(work,guide,y,nx,cap,cutoff,environment_row=opposite)
    if check:
        for tensor in work:tensor.data.check()
    return work.select(work.x_tag(y)).copy()


def refined_messages(tn,nx,ny,cap,sweeps=2,cutoff=0.,check=False):
    if nx<2 or ny<2 or cap<1 or sweeps<0 or int(sweeps)!=sweeps:
        raise ValueError('positive cap, rectangle at least 2x2, and nonnegative integer sweeps required')
    if tn.exponent!=0:raise ValueError('pass tensor scaling separately')
    if cutoff!=0.:
        raise ValueError('message-refinement pilot currently requires cutoff=0')
    start=time.perf_counter()
    top,bottom=boundary_messages(tn,nx,ny,cap)
    history=[dict(sweep=0,seconds=time.perf_counter()-start,**_cut_diagnostics(top,bottom,ny))]
    for sweep in range(1,sweeps+1):
        for y in range(1,ny-1):
            top[y]=_updated_message(tn,top[y-1],bottom[y+1],y,y-1,nx,cap,cutoff,check)
        for y in range(ny-2,0,-1):
            bottom[y]=_updated_message(tn,bottom[y+1],top[y-1],y,y+1,nx,cap,cutoff,check)
        history.append(dict(sweep=sweep,seconds=time.perf_counter()-start,**_cut_diagnostics(top,bottom,ny)))
    return dict(top=top,bottom=bottom,history=history,seconds=time.perf_counter()-start)


def refined_residuals(tn,nx,ny,cap,environment_cap,sweeps=2,check=False):
    """Apply ordinary forward residuals using independently rebuilt bottom guides."""
    start=time.perf_counter()
    messages=refined_messages(tn,nx,ny,environment_cap,sweeps=sweeps,check=check)
    result=streaming_residuals(tn,nx,ny,cap,environment_cap,granularity='row',
                              backward_guides={y:messages['bottom'][y+1] for y in range(1,ny-1)})
    result['residual_seconds']=result['seconds']
    result['refinement_seconds']=messages['seconds']
    result['message_history']=messages['history']
    result['seconds']=time.perf_counter()-start
    return result
