import itertools
import numpy as np
import pytest
from scipy.linalg import expm
from pauli_zne.circuit import (
    PAULIS,
    rotation_blocks,
    cnot,
    DIRECTIONS,
    transfers,
    gates,
    apply_axes,
    noise_diagonals,
)
from pauli_zne.config import load_config, schedules
from pauli_zne.exact import statevector_value, density_value
from pauli_zne.mps import evaluate, modes


def configuration(n=4, l=2, placement="before_cnot"):
    c, b = load_config(
        overrides={
            "circuit": {"N": n, "L": l, "phi": 0.27, "O": "IZZI" if n == 4 else "XX"},
            "noise": {"placement": placement},
            "mps": {"svd": "full", "cutoff": 1e-13},
        }
    )
    s, p = schedules(c, b)
    rng = np.random.default_rng(41)
    s[:] = rng.uniform(0, 0.013, s.shape)
    p[:] = rng.uniform(0, 0.003, p.shape)
    return c, s, p


@pytest.mark.parametrize("phi", [0.0, 0.196349540849, 0.7])
def test_kak_unitary(phi):
    blocks = rotation_blocks(phi)
    u = np.kron(*blocks[0])
    for j, d in enumerate(DIRECTIONS):
        u = np.kron(*blocks[j + 1]) @ cnot(d) @ u
    h = sum(np.kron(p, p) for p in PAULIS[1:])
    ref = expm(-0.5j * phi * h)
    phase = np.vdot(ref, u)
    phase /= abs(phase)
    np.testing.assert_allclose(u / phase, ref, atol=2e-14)


@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("placement", ["before_cnot", "after_cnot"])
def test_mps_matches_independent_density(n, placement):
    c, s, p = configuration(n, placement=placement)
    for scale in [0.0, 0.8, 3.0]:
        actual = evaluate(c, s, p, scale, 16)["value"]
        expected = density_value(c, s, p, scale)
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=1e-10)
    np.testing.assert_allclose(
        evaluate(c, s, p, 0, 16)["value"], statevector_value(c), atol=1e-11
    )


def dense_paths(c, s, p, scale, unsigned):
    n = c["circuit"]["N"]
    tensor = np.zeros((4,) * n)
    tensor[tuple("IXYZ".index(x) for x in c["circuit"]["O"])] = 1
    rotations, cnots = transfers(c["circuit"]["phi"])
    if unsigned:
        rotations = tuple(tuple(abs(x) for x in b) for b in rotations)
        cnots = tuple(abs(x) for x in cnots)

    def noise(a, t, ch):
        local, pair = noise_diagonals(s[t, ch], p[t, ch], scale)
        for q in range(n):
            sh = [1] * n
            sh[q] = 4
            a = a * local[q].reshape(sh)
        for q in range(n - 1):
            sh = [1] * n
            sh[q] = sh[q + 1] = 4
            a = a * pair[q].reshape(sh)
        return a

    def rotate(a, qs, stage):
        for q in qs:
            for side in [0, 1]:
                a = apply_axes(a, rotations[stage][side], (q + side,))
        return a

    for t in range(c["circuit"]["L"] - 1, -1, -1):
        for parity in [1, 0]:
            qs = gates(n, parity)
            if not qs:
                continue
            tensor = rotate(tensor, qs, 3)
            for j in range(2, -1, -1):
                d = DIRECTIONS[j]
                ch = 2 * parity + d
                if c["noise"]["placement"] == "after_cnot":
                    tensor = noise(tensor, t, ch)
                for q in qs:
                    tensor = apply_axes(tensor, cnots[d].reshape(16, 16), (q, q + 1))
                if c["noise"]["placement"] == "before_cnot":
                    tensor = noise(tensor, t, ch)
                tensor = rotate(tensor, qs, j)
    value = 0.0
    for codes in itertools.product(range(4), repeat=n):
        terminal = 1
        for q in range(0, n, 2):
            if codes[q] != codes[q + 1]:
                terminal = 0
                break
            if codes[q] != 0 and not unsigned:
                terminal *= -1
        value += tensor[codes] * terminal
    return value


@pytest.mark.parametrize("placement", ["before_cnot", "after_cnot"])
def test_unsigned_paths_and_stable_q(placement):
    c, s, p = configuration(placement=placement)
    o = []
    a = []
    for scale in [0.0, 1.0, 2.0, 4.0, 8.0]:
        signed = evaluate(c, s, p, scale, 16)
        absolute = evaluate(c, s, p, scale, 16, True)
        np.testing.assert_allclose(
            signed["value"], dense_paths(c, s, p, scale, False), atol=1e-11
        )
        np.testing.assert_allclose(
            absolute["value"], dense_paths(c, s, p, scale, True), atol=1e-11
        )
        assert absolute["value"] >= abs(signed["value"]) - 1e-12
        o.append(signed["value"])
        a.append(absolute["log_abs"])
    m = modes(o, a)
    aa = np.exp(a)
    np.testing.assert_allclose(m["Q_plus"], (aa + o) / (aa[0] + o[0]), atol=1e-12)
    np.testing.assert_allclose(m["Q_minus"], (aa - o) / (aa[0] - o[0]), atol=1e-12)
    assert m["Q_avg"][0] == pytest.approx(1)
    assert m["Q_contrast"][0] == pytest.approx(0)


def test_noise_placement_is_not_silently_swapped():
    c, s, p = configuration()
    before = density_value(c, s, p, 2)
    c["noise"]["placement"] = "after_cnot"
    after = density_value(c, s, p, 2)
    assert abs(before - after) > 1e-7


def test_randomized_svd_agrees_at_sufficient_rank():
    c, s, p = configuration()
    expected = evaluate(c, s, p, 1.0, 16)["value"]
    c["mps"].update(svd="randomized", oversample=2, power_iterations=3)
    np.testing.assert_allclose(
        evaluate(c, s, p, 1.0, 16)["value"], expected, atol=1e-11
    )


def test_ratio_and_sector_degeneracies():
    assert modes([0.0, 0.1], [0.0, 0.0])["ratio"] is None
    assert (
        modes([1.0, 0.9], [0.0, np.log(0.9)])["status"]
        == "undefined_sector_normalization"
    )
    m = modes([0.1, 0.05], [1000.0, 999.0])
    assert np.isfinite(m["Q_contrast"]).all()  # usable even if Qdiff underflows


def test_randomized_range_finder_on_known_low_rank_matrix():
    from pauli_zne.mps import MPS

    c, _ = load_config()
    c["mps"].update(oversample=2, power_iterations=2)
    rng = np.random.default_rng(22)
    matrix = rng.normal(size=(64, 6)) @ rng.normal(size=(6, 80))
    solver = MPS("ZZ", 6, c["mps"])
    left, right = solver.split(matrix)
    np.testing.assert_allclose(left @ right, matrix, atol=1e-11, rtol=1e-10)
