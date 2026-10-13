import json
import os
from pathlib import Path
import numpy as np
import pytest
from tensor_zne import evaluate_mps, evaluate_peps, curves, write_outputs
from lattice_circuit import chain, Lattice, square
from causal_pauli_network import compile_network


@pytest.mark.parametrize('unsigned', [False, True])
@pytest.mark.parametrize('scale', [0., 1.7])
def test_mps_residual_against_independent_exact_network(unsigned, scale):
    # Historical MPS backward order: even, noise, odd, noise.
    base = chain(6)
    lat = Lattice(6, tuple(reversed(base.layers)), base.order)
    exact = compile_network('IIZIZI', lat, pairs=[(0,1),(2,3),(4,5)],
        n_steps=3, phi=.23, lam_xyz=(.001,.003,.008), scale=scale,
        noise_placement='layer', noise_timing='before', unsigned=unsigned).contract()['value']
    result = evaluate_mps(n=6, steps=3, sites=(2,4), phi=.23, lam=(.001,.003,.008),
        scale=scale, cap=2, guide_cap=64, terminal=1, unsigned=unsigned)
    assert result['corrected'] == pytest.approx(exact, rel=1e-10, abs=1e-11)
    assert abs(result['correction']) > 1e-8


@pytest.mark.parametrize('unsigned', [False, True])
def test_peps_residual_workflow_exact_small_system(unsigned):
    lat = square(2,3)
    exact = compile_network('ZZIIII', lat, pairs=list(zip(lat.order[::2],lat.order[1::2])),
        n_steps=1, phi=.19, lam_xyz=(.001,.002,.003), scale=1.3,
        unsigned=unsigned).contract()['value']
    result = evaluate_peps(nx=2, ny=3, steps=1, sites=(0,1), phi=.19, scale=1.3,
        cap=64, boundary_cap=1, environment_cap=None, unsigned=unsigned)
    assert result['corrected'] == pytest.approx(exact, rel=1e-9, abs=1e-10)


def test_q_normalization_and_existing_fit_loader(tmp_path):
    result = curves(evaluate_mps, [1.,2.], n=6, steps=3, sites=(2,4), cap=4,
                    guide_cap=64, terminal=1)
    zero = result['rows'][0]
    assert zero['ratio_corrected'] == 1.
    assert zero['q_plus_corrected'] == 1.
    assert zero['q_minus_corrected'] == 1.
    for r in result['rows']:
        assert r['plus_num_corrected'] - r['minus_num_corrected'] == pytest.approx(r['observable_corrected'])
    prefix = tmp_path/'mps'
    write_outputs(prefix, result, {'mode':'both'}, 'layer')
    from fit_zne_compare import load_q_groups, load_observables, q_curve_from_groups
    groups, samples = load_q_groups(str(prefix)+'_q_groups.csv')
    qp, qm = q_curve_from_groups(groups['tensor_zne']['layer'], [0.,1.,2.], 'pointwise_mom')
    np.testing.assert_allclose(qp, [r['q_plus_corrected'] for r in result['rows']])
    values, _ = load_observables(str(prefix)+'_observables.csv')
    assert values['tensor_zne'][1.] == result['rows'][1]['observable_corrected']
    assert samples == 0
    assert json.loads(prefix.with_suffix('.json').read_text())['config']['mode'] == 'both'


def test_undefined_ratio_and_absent_sector_are_not_clipped(tmp_path):
    def zero(**kw):
        return dict(approximate=0., correction=0., corrected=0.)
    result = curves(zero, [0.,1.])
    assert result['rows'][1]['ratio_corrected'] is None
    assert result['rows'][1]['q_plus_corrected'] is None
    prefix = tmp_path/'zero'
    write_outputs(prefix, result, {'mode':'both'}, 'layer')
    assert not Path(str(prefix)+'_q_groups.csv').exists()



@pytest.mark.skipif(os.environ.get('TEST_PEPS_GPU_BOUNDARY') != '1',
                    reason='requires explicit Metal access')
@pytest.mark.parametrize('unsigned', [False, True])
def test_peps_workflow_metal_matches_cpu(unsigned):
    kw = dict(nx=2, ny=3, steps=1, cap=4, boundary_cap=2, environment_cap=8,
              unsigned=unsigned)
    cpu = evaluate_peps(**kw)
    gpu = evaluate_peps(**kw, gate_backend='metal', boundary_backend='metal-jacobi')
    assert gpu['corrected'] == pytest.approx(cpu['corrected'], rel=2e-4, abs=2e-6)


@pytest.mark.skipif(os.environ.get('TEST_MPS_METAL') != '1',
                    reason='requires explicit Metal access')
def test_mps_metal_randomized_products_and_residual_workflow():
    from optimized_pauli_svd import RandomizedSVD
    rng = np.random.default_rng(8)
    a = rng.normal(size=(100, 12)) @ rng.normal(size=(12, 80))
    u, s, v = RandomizedSVD('lu', backend='metal')(a, 16, 2)
    assert np.linalg.norm((u*s)@v-a) / np.linalg.norm(a) < 1e-5
    cpu = evaluate_mps(n=6, steps=3, sites=(2,4), cap=4, guide_cap=32, terminal=1)
    gpu = evaluate_mps(n=6, steps=3, sites=(2,4), cap=4, guide_cap=32, terminal=1, backend='metal')
    assert gpu['corrected'] == pytest.approx(cpu['corrected'], rel=1e-5, abs=1e-7)
