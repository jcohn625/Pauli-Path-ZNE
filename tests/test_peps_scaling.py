"""Check block gauge scaling against dense broadcasts across axis permutations."""
import numpy as np
import pytest
from quimb.tensor import Tensor
from pauli_peps_symmetry import symmray, CHARGES
from peps_metal_layer import LayerEngine


@pytest.mark.parametrize('use_numba', [False, True])
@pytest.mark.parametrize('scaled', [(), (0,), (0, 2, 3)])
@pytest.mark.parametrize('axes', [(0, 1, 2, 3), (3, 1, 0, 2)])
def test_scale_many_preserves_input_and_charge_blocks(scaled, axes, use_numba, monkeypatch):
    import cpu_numba_kernels
    monkeypatch.setitem(cpu_numba_kernels.ENABLED, "scale", use_numba)
    sr = symmray()
    maps = [CHARGES*2, CHARGES, CHARGES*2, CHARGES]
    rng = np.random.default_rng(172)
    dense = rng.normal(size=tuple(map(len, maps)))
    data = sr.Z2Z2Array.from_dense(dense, maps, [False, False, True, True],
                                 charge=(0, 0), invalid_sectors='ignore')
    t = Tensor(data, inds=('a', 'b', 'c', 'd'), tags=['site'])
    original = data.to_dense().copy()
    expected = original.copy()
    scales = {}
    for axis in scaled:
        values = {q:rng.uniform(.2, 2., size=maps[axis].count(q)) for q in CHARGES}
        scales[t.inds[axis]] = sr.BlockVector(values)
        # Dense symmray axes are sorted by charge, not original from_dense order.
        v = np.concatenate([values[q] for q in sorted(values)])
        shape = [1]*4; shape[axis] = len(v)
        expected *= v.reshape(shape)
    engine = LayerEngine('cpu')
    out, = engine.scale_many([(t, scales, tuple(t.inds[a] for a in axes))])
    np.testing.assert_array_equal(out.data.to_dense(), expected.transpose(axes))
    np.testing.assert_array_equal(t.data.to_dense(), original)
    assert out.tags == t.tags
    assert engine.stats['scale_blocks'] == len(data.blocks)
    for key, block in out.data.blocks.items():
        assert not any(np.shares_memory(block, x) for x in data.blocks.values())
    out.data.check()
