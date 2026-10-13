"""Shared physical lattice and Pauli-transfer circuit definitions (NumPy only).

Physical sites are integers; tensor positions are a separate permutation.
Transfers use output,input indexing. Forward gate U=exp(-i phi (XX+YY+ZZ)/2).
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np


@dataclass(frozen=True)
class BondLayer:
    name: str
    bonds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Lattice:
    n_sites: int
    layers: tuple[BondLayer, ...]
    order: tuple[int, ...]

    def __post_init__(self):
        if self.n_sites < 1 or sorted(self.order) != list(range(self.n_sites)):
            raise ValueError('order must be a permutation of all physical sites')
        for layer in self.layers:
            touched = [q for pair in layer.bonds for q in pair]
            if any(len(pair) != 2 for pair in layer.bonds):
                raise ValueError('bonds must have two endpoints')
            if len(set(touched)) != len(touched):
                raise ValueError('each sublayer must be a matching')
            if any(q < 0 or q >= self.n_sites for q in touched):
                raise ValueError('bond endpoint outside lattice')


def chain(n_sites):
    return Lattice(n_sites, tuple(BondLayer(name, tuple((i, i+1)
        for i in range(parity, n_sites-1, 2)))
        for parity, name in enumerate(('even', 'odd'))), tuple(range(n_sites)))


def square(nx, ny):
    """Open rectangle, physical id=y*nx+x; snake across the shorter dimension."""
    if nx < 1 or ny < 1:
        raise ValueError('dimensions must be positive')
    layers = []
    for axis in ('horizontal', 'vertical'):
        for parity, name in enumerate(('even', 'odd')):
            bonds = []
            for y in range(ny):
                for x in range(nx):
                    if axis == 'horizontal' and x % 2 == parity and x+1 < nx:
                        bonds.append((y*nx+x, y*nx+x+1))
                    if axis == 'vertical' and y % 2 == parity and y+1 < ny:
                        bonds.append((y*nx+x, (y+1)*nx+x))
            layers.append(BondLayer(f'{axis}_{name}', tuple(bonds)))
    if nx <= ny:
        order = tuple(y*nx+x for y in range(ny)
                      for x in (range(nx) if y % 2 == 0 else reversed(range(nx))))
    else:
        order = tuple(y*nx+x for x in range(nx)
                      for y in (range(ny) if x % 2 == 0 else reversed(range(ny))))
    return Lattice(nx*ny, tuple(layers), order)


@lru_cache(maxsize=128)
def heisenberg_transfer(phi):
    """Backward Heisenberg transfer, matching the original solver's +phi table."""
    p = np.array([np.eye(2), [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
    basis = np.array([np.kron(a,b) for a in p for b in p])
    h = sum(np.kron(a,a) for a in p[1:])
    e, v = np.linalg.eigh(h)
    u = (v * np.exp(-0.5j*phi*e)) @ v.conj().T
    evolved = u.conj().T @ basis @ u
    transfer = np.einsum('aij,bji->ab', basis, evolved).real / 4
    transfer[np.abs(transfer) < 1e-14] = 0
    transfer.setflags(write=False)
    return transfer


def noise_schedule(lattice, n_steps, lam_xyz, scale, model):
    shape = (n_steps,len(lattice.layers),lattice.n_sites,3)
    lam = np.asarray(lam_xyz,dtype=float)
    if lam.shape not in ((3,), (lattice.n_sites,3), shape[1:], shape):
        raise ValueError(f'lambda shape must be (3,), (N,3), (layers,N,3), or {shape}')
    if not np.isfinite(scale) or scale < 0 or np.any(~np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError('noise rates and scale must be finite and nonnegative')
    lam = np.broadcast_to(lam,shape)
    e = np.exp(-2*lam)
    if model == 'independent':
        eta = np.stack((e[...,1]*e[...,2],e[...,0]*e[...,2],e[...,0]*e[...,1]),axis=-1)
    elif model == 'legacy_sum':
        eta = np.stack((e[...,1]+e[...,2]-1,e[...,0]+e[...,2]-1,e[...,0]+e[...,1]-1),axis=-1)
        if np.any(eta < 0):
            raise ValueError('legacy_sum factors must be nonnegative for continuous noise scaling')
    else:
        raise ValueError('unknown noise model')
    # Same eta**s convention as the existing ZNE benchmarks.
    return np.concatenate((np.ones(shape[:-1]+(1,)),eta**scale),axis=-1)


def backward_operations(lattice, n_steps, phi, lam_xyz, scale=1.,
                        noise_model='independent', noise_placement='gate',
                        noise_timing='after', unsigned=False):
    """noise_timing is in FORWARD circuit time; default gate then noise.

    'before' supports legacy gate-noise convention (backward gate then noise).
    'gate' damps touched sites; 'layer' damps all sites, including empty layers.
    """
    if not isinstance(n_steps,int) or n_steps < 0:
        raise ValueError('n_steps must be a nonnegative integer')
    if noise_placement not in ('gate','layer') or noise_timing not in ('before','after'):
        raise ValueError('invalid noise placement or timing')
    angles = np.broadcast_to(np.asarray(phi,dtype=float),(n_steps,len(lattice.layers)))
    if np.any(~np.isfinite(angles)):
        raise ValueError('angles must be finite')
    factors = noise_schedule(lattice,n_steps,lam_xyz,scale,noise_model)
    ops = []
    for t in reversed(range(n_steps)):
        for l in reversed(range(len(lattice.layers))):
            layer = lattice.layers[l]
            f = factors[t,l].copy()
            if noise_placement == 'gate':
                touched = {q for pair in layer.bonds for q in pair}
                for q in range(lattice.n_sites):
                    if q not in touched:
                        f[q] = 1
            transfer = heisenberg_transfer(float(angles[t,l]))
            if unsigned:
                transfer = abs(transfer)
            gates = [('gate',(a,b),transfer) for a,b in reversed(layer.bonds)]
            noise = [('noise',None,f)]
            ops.extend(noise+gates if noise_timing == 'after' else gates+noise)
    return ops
