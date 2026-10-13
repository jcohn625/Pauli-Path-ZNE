"""Same sparse Pauli transfer, with contiguous right-bond access."""
import numpy as np
import numba as nb

@nb.njit(cache=True)
def odd_transfer(theta, odd_new_a, odd_new_b, odd_coeffs, odd_n_branches):
    chi_l,_,_,chi_r=theta.shape
    out=np.zeros(theta.shape,dtype=theta.dtype)
    for l in range(chi_l):
        for a in range(16):
            for b in range(16):
                for k in range(odd_n_branches[a,b]):
                    new_a=int(odd_new_a[a,b,k]);new_b=int(odd_new_b[a,b,k])
                    c=odd_coeffs[a,b,k]
                    for r in range(chi_r):
                        out[l,new_a,new_b,r]+=c*theta[l,a,b,r]
    return out
