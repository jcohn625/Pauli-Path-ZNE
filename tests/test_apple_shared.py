import os
import numpy as np
import pytest
from apple_silicon_hybrid import HybridProducts,SharedBatch


def test_cpu_default_precision_and_shape_fallback():
    rng=np.random.default_rng(12);pairs=[(rng.normal(size=(3,5)),rng.normal(size=(5,2))) for _ in range(2)]
    engine=HybridProducts(gpu_shapes=[(2,3,5,2)])
    result=engine(pairs)
    for x,(a,b) in zip(result,pairs):np.testing.assert_array_equal(x,a@b)
    assert engine.pool is None and engine.stats['cpu_batches']==1
    assert engine([])==[]


def test_budget_guard_before_mlx_allocation():
    with pytest.raises(ValueError,match='budget'):SharedBatch(8,512,512,512,budget_bytes=1)
    with pytest.raises(ValueError,match='positive'):SharedBatch(0,4,4,4)


@pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires Metal')
def test_reused_shared_buffers_and_inflight_guard():
    rng=np.random.default_rng(123);engine=SharedBatch(3,24,17,11)
    assert not engine.av.flags.owndata and not engine.bv.flags.owndata
    for repeat in range(8):
        pairs=[(rng.normal(size=(24,17)),rng.normal(size=(17,11))) for _ in range(3)]
        engine.submit(pairs)
        with pytest.raises(RuntimeError,match='in-flight'):engine.submit(pairs)
        actual=engine.resolve();reference=np.array([a@b for a,b in pairs])
        np.testing.assert_allclose(actual,reference,atol=4e-6,rtol=3e-5)
        assert actual.dtype==np.float64
    with pytest.raises(RuntimeError,match='no submitted'):engine.resolve()


@pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires Metal')
def test_measured_shape_dispatch_and_memory_fallback():
    pairs=[(np.eye(16),np.eye(16)) for _ in range(2)]
    engine=HybridProducts(gpu_shapes=[(2,16,16,16)],mixed_precision=True)
    np.testing.assert_array_equal(engine(pairs),np.array([np.eye(16)]*2))
    assert engine.stats['gpu_batches']==1
    small=HybridProducts(gpu_shapes=[(2,16,16,16)],mixed_precision=True,budget_bytes=1)
    small(pairs)
    assert small.pool is None and small.stats['cpu_batches']==1
