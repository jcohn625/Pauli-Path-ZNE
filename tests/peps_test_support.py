import numpy as np
from autoray import get_lib_fn, register_function
from cotengra.contract import default_implementation
from lattice_circuit import square
from pauli_peps import evaluate_peps, _safe_einsum, _safe_tensordot
from peps_boundary import boundary_backend

def fixture(nx,ny,steps,cap,scale,sites=None):
    lat=square(nx,ny);obs=list('I'*(nx*ny))
    if sites is None:sites=(34,39) if nx==ny==8 else ((5,10) if nx==ny==4 else (0,nx*ny-1))
    if len(sites)!=2 or len(set(sites))!=2 or any(i<0 or i>=nx*ny for i in sites):
        raise ValueError('observable requires two distinct valid sites')
    for i in sites:obs[i]='Z'
    saved=[]
    class Captured(Exception):pass
    def capture(tn,log_scale):saved.append((tn.copy(),log_scale));raise Captured
    try:
        evaluate_peps(obs,lat,nx=nx,ny=ny,pairs=list(zip(lat.order[::2],lat.order[1::2])),
            n_steps=steps,phi=np.pi/16,lam_xyz=(.001,.002,.003),scale=scale,
            symmetry='z2z2',bond_cap=cap,capture_boundary=capture)
    except Captured:pass
    return saved[0]
