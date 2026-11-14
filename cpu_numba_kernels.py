"""Strict FP64 CPU kernels around (not replacing) BLAS/LAPACK.

No fastmath or internal threading: preserve multiplication order and avoid BLAS
oversubscription. The switches also support matched reference benchmarks.
"""
import os
from functools import lru_cache
import numpy as np
from numba import njit

# Measured CPU policy. Experimental kernels remain available to benchmarks.
ENABLED = {name: os.environ.get('PAULI_CPU_NUMBA', '1') != '0' and
           name in ('scale', 'gather', 'select', 'scatter')
           for name in ('scale', 'gather', 'select', 'scatter', 'transfer')}


@njit(cache=True)
def gather(matrix, rows, cols):
    out = np.empty((len(rows), len(cols)), dtype=matrix.dtype)
    for i in range(len(rows)):
        for j in range(len(cols)):
            out[i, j] = matrix[rows[i], cols[j]]
    return out


@njit(cache=True)
def select_sorted(values, offsets, cap):
    """Stable k-way merge of descending spectra; ties retain piece/index order."""
    heads = offsets[:-1].copy()
    selected = np.empty((min(cap, len(values)), 2), dtype=np.int64)
    positive = 0
    for v in values:
        if v > 0:
            positive += 1
    count = 0
    threshold = 0.
    for k in range(len(selected)):
        best = 0.
        piece = -1
        for p in range(len(heads)):
            if heads[p] < offsets[p+1] and values[heads[p]] > best:
                best = values[heads[p]]
                piece = p
        if piece < 0:
            break
        if k == 0:
            threshold = 1e-13 * best
        if best <= threshold:
            break
        selected[count, 0] = piece
        selected[count, 1] = heads[piece] - offsets[piece]
        heads[piece] += 1
        count += 1
    return selected[:count], positive


@njit(cache=True)
def scatter_svd(left, right, newq, selected, piece, charge, rows, cols, u, s, v):
    for k in range(len(selected)):
        if selected[k, 0] != piece:
            continue
        j = selected[k, 1]
        newq[k] = charge
        for i in range(len(rows)):
            left[rows[i], k] = u[i, j]
        for i in range(len(cols)):
            right[k, cols[i]] = s[j] * v[j, i]


@njit(cache=True)
def transfer(theta, columns, weights):
    """One allocation for a physical permutation followed by signed scaling."""
    out = np.empty_like(theta)
    for i in range(theta.shape[0]):
        for p in range(theta.shape[1]):
            for j in range(theta.shape[2]):
                out[i, p, j] = theta[i, columns[p], j] * weights[p]
    return out


@lru_cache(maxsize=512)
def _scale_layout(shape, axes):
    padded = shape + (1,) * (5-len(shape))
    strides = [0] * 5
    stride = 1
    for axis in reversed(axes):
        strides[axis] = stride
        stride *= shape[axis]
    return padded, tuple(strides), tuple(shape[a] for a in axes)


@njit(cache=True)
def _scale_pack(a, weights, active, strides):
    out = np.empty(a.size, dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                for l in range(a.shape[3]):
                    for m in range(a.shape[4]):
                        x = a[i,j,k,l,m]
                        if active[0]: x = x * weights[0][i]
                        if active[1]: x = x * weights[1][j]
                        if active[2]: x = x * weights[2][k]
                        if active[3]: x = x * weights[3][l]
                        if active[4]: x = x * weights[4][m]
                        out[i*strides[0]+j*strides[1]+k*strides[2]+l*strides[3]+m*strides[4]] = x
    return out


_ONE = np.ones(1)


def scale_pack(block, weights, axes):
    """Weights are five FP64 vectors or None; output is packed in axes order."""
    _, strides, shape = _scale_layout(block.shape, axes)
    # Adding singleton axes preserves arbitrary strides without copying input.
    view = block[(...,) + (None,) * (5-block.ndim)]
    active = tuple(w is not None for w in weights)
    vectors = tuple(_ONE if w is None else w for w in weights)
    return _scale_pack(view, vectors, active, strides).reshape(shape)
