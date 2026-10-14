"""Experimental structural plans for NumPy Z2xZ2 PEPS splits (not enabled).

Numerical decompositions remain quimb/symmray's reference implementation.
Only charge metadata and slice/reshape instructions are retained in caches.
"""
from collections import OrderedDict
import numpy as np
from quimb.tensor import Tensor
_REFERENCE_SPLIT = Tensor.split

_FUSE = OrderedDict()
_UNFUSE = OrderedDict()
_LIMIT = 512


def _remember(cache, key, build):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    plan = build()
    cache[key] = plan
    if len(cache) > _LIMIT:
        cache.popitem(last=False)
    return plan


def _key(a):
    return (tuple(ix.hashkey() for ix in a.indices), tuple(a.blocks))


def fuse_matrix(a, nleft):
    """Fuse contiguous left/right groups using symmray's exact sector ordering."""
    if a.ndim == 2 and nleft == 1:
        return a
    groups = (tuple(range(nleft)), tuple(range(nleft, a.ndim)))
    def build():
        # Adapter for the pinned symmray 0.4 metadata interface.
        from symmray.sparse.sparse_array_common import cached_fuse_block_info
        ng, single, perm, pos, _, _, _, indices, mapping = cached_fuse_block_info(a, groups)
        slices = []
        for g in range(ng):
            lookup = {}
            if g not in single:
                for q, extents in indices[pos+g].subinfo.extents.items():
                    offset = 0
                    for sub, size in extents.items():
                        lookup[q, sub] = slice(offset, offset+size)
                        offset += size
            slices.append(lookup)
        jobs = []; shapes = {}
        for sector in a.blocks:
            shape, target, subsectors = mapping[sector]
            selector = [slice(None)]*len(indices)
            for g, sub in enumerate(subsectors):
                if g not in single:
                    selector[pos+g] = slices[g][target[pos+g], sub]
            jobs.append((sector, target, tuple(selector), shape))
            shapes[target] = tuple(ix.size_of(q) for ix, q in zip(indices, target))
        return indices, perm, tuple(shapes.items()), tuple(jobs)
    indices, perm, shapes, jobs = _remember(_FUSE, (_key(a), nleft), build)
    blocks = {q:np.zeros(shape, dtype=next(iter(a.blocks.values())).dtype) for q, shape in shapes}
    for source, target, selector, shape in jobs:
        blocks[target][selector] = a.blocks[source].transpose(perm).reshape(shape)
    return a.copy_with(indices=indices, blocks=blocks)


def unfuse_axis(a, axis):
    def build():
        subinfo = a.indices[axis].subinfo
        slices = {}
        for q, extents in subinfo.extents.items():
            offset = 0; parts = []
            for sub, size in extents.items():
                shape = tuple(ix.size_of(c) for ix, c in zip(subinfo.indices, sub))
                parts.append((sub, slice(offset, offset+size), shape)); offset += size
            slices[q] = parts
        jobs = []
        for sector, block in a.blocks.items():
            for sub, slc, shape in slices[sector[axis]]:
                key = sector[:axis]+sub+sector[axis+1:]
                selector = (slice(None),)*axis+(slc,)
                jobs.append((sector, key, selector, block.shape[:axis]+shape+block.shape[axis+1:]))
        indices = a.indices[:axis]+subinfo.indices+a.indices[axis+1:]
        return indices, tuple(jobs)
    indices, jobs = _remember(_UNFUSE, (_key(a), axis), build)
    blocks = {key:a.blocks[source][selector].reshape(shape) for source, key, selector, shape in jobs}
    return a.copy_with(indices=indices, blocks=blocks)


def split_tensor(tensor, *, left_inds, right_inds, bond_ind, absorb,
                 cutoff=0., max_bond=None, get='tensors'):
    """Restricted split adapter for the three PEPS simple-update splits."""
    from quimb.tensor import Tensor
    from quimb.tensor.decomp import array_split
    left = tuple(ix for ix in tensor.inds if ix not in right_inds) if left_inds is None else tuple(left_inds)
    right = tuple(ix for ix in tensor.inds if ix not in left) if right_inds is None else tuple(right_inds)
    if (get != 'tensors' or not left or not right or type(tensor.data).__name__ != 'Z2Z2Array'
            or tensor.data.backend != 'numpy' or absorb not in ('left', 'right', None)):
        return _REFERENCE_SPLIT(tensor, left_inds=left, right_inds=right, bond_ind=bond_ind,
                            absorb=absorb, cutoff=cutoff, max_bond=max_bond, get=get)
    a = tensor.transpose(*left, *right).data
    matrix = fuse_matrix(a, len(left))
    # Tensor.split uses relative singular-value cutoff, unlike array_split's
    # default discarded-weight cutoff. Preserve the tensor API's semantics.
    l, s, r = array_split(matrix, absorb=absorb, cutoff=cutoff,
                          cutoff_mode='rel', max_bond=max_bond)
    if len(left) != 1:l = unfuse_axis(l, 0)
    if len(right) != 1:r = unfuse_axis(r, 1)
    tl = Tensor(l, inds=(*left, bond_ind), tags=tensor.tags,
                left_inds=left if absorb in ('right', None) else None)
    tr = Tensor(r, inds=(bond_ind, *right), tags=tensor.tags,
                left_inds=right if absorb in ('left', None) else None)
    if absorb is None:
        return tl, Tensor(s, inds=(bond_ind,), tags=tensor.tags), tr
    return tl, tr
