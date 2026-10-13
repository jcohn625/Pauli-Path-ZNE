"""Z2 x Z2 blocks for conserved relative Pauli-count parities.

In I,X,Y,Z order the charges are (0,0),(1,1),(1,0),(0,1):
(N_X+N_Y, N_X+N_Z) mod 2. Neutral strings have all three
count parities equal, not necessarily all even.
"""
from pathlib import Path
import sys
import numpy as np

CHARGES = ((0, 0), (1, 1), (1, 0), (0, 1))


def symmray():
    try:
        import symmray as sr
    except ImportError:
        local = Path(__file__).resolve().parent / '.symmetry_deps'
        if not (local / 'symmray').is_dir():
            raise ImportError('Z2xZ2 PEPS requires symmray==0.4.0; install it or use symmetry="none"') from None
        sys.path.insert(0, str(local))
        import symmray as sr
    return sr


def total_charge(codes):
    return tuple(sum(CHARGES[c][i] for c in codes) % 2 for i in (0, 1))


def convert_product(peps, codes, nx):
    sr = symmray()
    for q, c in enumerate(codes):
        coord = divmod(q, nx)
        t = peps[coord]
        physical = peps.site_ind(*coord)
        maps, duals = [], []
        for ix in t.inds:
            if ix == physical:
                maps.append(CHARGES)
                duals.append(False)
            else:
                maps.append(((0, 0),))
                other = next(tid for tid in peps.ind_map[ix] if peps.tensor_map[tid] is not t)
                # Any consistent orientation works for self-inverse Z2 charges.
                this = next(tid for tid in peps.ind_map[ix] if peps.tensor_map[tid] is t)
                duals.append(this > other)
        t.modify(data=sr.Z2Z2Array.from_dense(t.data, maps, duals,
                                             charge=CHARGES[c], invalid_sectors='raise'))


def gate(data):
    return symmray().Z2Z2Array.from_dense(
        data.reshape(4, 4, 4, 4), [CHARGES]*4,
        [False, False, True, True], charge=(0, 0), invalid_sectors='raise')


def diagonal(values):
    return symmray().BlockVector({q: np.array([v]) for q, v in zip(CHARGES, values)})


def flip_physical_dual(tensor, index):
    a = tensor.data
    indices = list(a.indices)
    axis = tensor.inds.index(index)
    indices[axis] = indices[axis].conj()
    tensor.modify(data=a.copy_with(indices=tuple(indices)))


def statistics(peps):
    stored = sum(block.size for t in peps for block in t.data.blocks.values())
    dense = sum(int(np.prod(t.shape)) for t in peps)
    maximum = max((peps.ind_size(ix) for ix in peps.inner_inds()), default=1)
    for t in peps:
        t.data.check()
    return dict(stored_elements=stored, dense_equivalent_elements=dense,
                storage_fraction=stored/dense if dense else 0., max_bond=maximum,
                tensor_charge_checks_passed=True)
