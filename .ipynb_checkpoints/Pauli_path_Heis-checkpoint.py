import numpy as np
import numba as nb
from numba import njit, prange
from math import sin, cos

# 0->I, 1->X, 2->Y, 3->Z
CHAR_TO_INT = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
INT_TO_CHAR = np.array(['I', 'X', 'Y', 'Z'])

def encode_2q(s: str) -> int:
    return 4 * CHAR_TO_INT[s[0]] + CHAR_TO_INT[s[1]]

def decode_2q(code: int) -> str:
    return INT_TO_CHAR[code // 4] + INT_TO_CHAR[code % 4]


P2 = {}
P2['II']=['II']
P2['XX']=['XX']
P2['YY']=['YY']
P2['ZZ']=['ZZ']
P2['XI'] = ['XI','IX','YZ','ZY']
P2['YI'] = ['YI','IY','ZX','XZ']
P2['ZI'] = ['ZI','IZ','XY','YX']
P2['IX'] = ['IX','XI','ZY','YZ']
P2['IY'] = ['IY','YI','XZ','ZX']
P2['IZ'] = ['IZ','ZI','YX','XY']
P2['XY'] = ['XY','YX','ZI','IZ']
P2['YX'] = ['YX','XY','IZ','ZI']
P2['ZX'] = ['ZX','XZ','YI','IY']
P2['XZ'] = ['XZ','ZX','IY','YI']
P2['YZ'] = ['YZ','ZY','XI','IX']
P2['ZY'] = ['ZY','YZ','XI','IX']

COM = ['II', 'XX', 'YY', 'ZZ']


def build_transition_tables(phi: float):
    n_codes = 16

    trans_codes = np.full((n_codes, 4), -1, dtype=np.int8)
    probs = np.zeros((n_codes, 4), dtype=np.float64)
    is_commuting = np.zeros(n_codes, dtype=np.bool_)
    branch_sign = np.ones((n_codes, 4), dtype=np.int8)

    c = np.cos(phi)
    s = np.sin(phi)
    sin2 = np.sin(2.0 * phi)
    abs_sin2 = abs(sin2)
    amp_factr = 1.0 + abs_sin2

    for s2q in COM:
        is_commuting[encode_2q(s2q)] = True

    for key, outs in P2.items():
        in_code = encode_2q(key)

        for k, s_out in enumerate(outs):
            trans_codes[in_code, k] = encode_2q(s_out)

        if is_commuting[in_code]:
            probs[in_code, 0] = 1.0
        else:
            probs[in_code, 0] = c * c / amp_factr
            probs[in_code, 1] = s * s / amp_factr
            probs[in_code, 2] = 0.5 * abs_sin2 / amp_factr
            probs[in_code, 3] = 0.5 * abs_sin2 / amp_factr

    # Correct sign patterns for positive sin(2phi)
    neg_pos = ['XI', 'YI', 'ZI', 'IX', 'IY', 'IZ', 'ZY']   # [+, +, -, +]
    pos_neg = ['XY', 'YX', 'ZX', 'XZ', 'YZ']               # [+, +, +, -]

    for key in neg_pos:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = -1
        branch_sign[in_code, 3] = +1

    for key in pos_neg:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = +1
        branch_sign[in_code, 3] = -1

    # If sin(2phi) < 0, flip both commutator signs
    if sin2 < 0.0:
        branch_sign[:, 2] *= -1
        branch_sign[:, 3] *= -1

    return trans_codes, probs, is_commuting, branch_sign, amp_factr

@njit
def _sample_branch(probs_row):
    r = np.random.random()
    cum = 0.0
    for k in range(4):
        p = probs_row[k]
        if p <= 0.0:
            continue
        cum += p
        if r < cum:
            return k

    # fallback
    max_idx = 0
    max_val = -1.0
    for k in range(4):
        if probs_row[k] > max_val:
            max_val = probs_row[k]
            max_idx = k
    return max_idx

@njit
def apply_gate_with_weight(
    pauli, q1, q2,
    trans_codes, probs, is_commuting, branch_sign,
    amp_factr,
    sign, amp
):
    p1 = pauli[q1]
    p2 = pauli[q2]
    in_code = 4 * p1 + p2

    if is_commuting[in_code]:
        return sign, amp

    branch = _sample_branch(probs[in_code])
    out_code = trans_codes[in_code, branch]

    pauli[q1] = out_code // 4
    pauli[q2] = out_code % 4

    amp *= amp_factr
    sign *= branch_sign[in_code, branch]

    return sign, amp

@njit
def apply_1q_diag_noise(pauli, q, eta, sign, amp, damp):
    """
    Single-qubit diagonal noise:
      I -> I
      X,Y,Z -> eta * same

    Only damp is updated.
    """
    if pauli[q] != 0:
        damp *= eta
    return sign, amp, damp


@njit
def apply_1q_noise_layer(pauli, eta, sign, amp, damp):
    for q in range(pauli.shape[0]):
        if pauli[q] != 0:
            damp *= eta
    return sign, amp, damp


@njit
def apply_1q_noise_layer_eta_array(pauli, etas, sign, amp, damp):
    for q in range(pauli.shape[0]):
        if pauli[q] != 0:
            damp *= etas[q]
    return sign, amp, damp

@njit
def evolve_one_path_with_1q_noise(
    init_pauli,
    gates,
    trans_codes, probs, is_commuting, branch_sign,
    amp_factr,
    eta
):
    n_qubits = init_pauli.shape[0]
    pauli = np.empty(n_qubits, dtype=np.int8)
    for i in range(n_qubits):
        pauli[i] = init_pauli[i]

    sign = np.int8(1)
    amp = 1.0
    damp = 1.0

    for g in range(gates.shape[0]):
        q1 = gates[g, 0]
        q2 = gates[g, 1]

        # unitary step updates sign and amp only
        sign, amp = apply_gate_with_weight(
            pauli, q1, q2,
            trans_codes, probs, is_commuting, branch_sign,
            amp_factr,
            sign, amp
        )

        # noise step updates damp only
        sign, amp, damp = apply_1q_diag_noise(pauli, q1, eta, sign, amp, damp)
        sign, amp, damp = apply_1q_diag_noise(pauli, q2, eta, sign, amp, damp)

    return pauli, sign, amp, damp

@njit(parallel=True)
def evolve_many_paths_with_1q_noise(
    init_pauli,
    gates,
    trans_codes, probs, is_commuting, branch_sign,
    amp_factr,
    eta,
    n_samples
):
    n_qubits = init_pauli.shape[0]

    out_paulis = np.empty((n_samples, n_qubits), dtype=np.int8)
    out_signs = np.empty(n_samples, dtype=np.int8)
    out_amps = np.empty(n_samples, dtype=np.float64)
    out_damps = np.empty(n_samples, dtype=np.float64)

    for s in prange(n_samples):
        pauli = np.empty(n_qubits, dtype=np.int8)
        for i in range(n_qubits):
            pauli[i] = init_pauli[i]

        sign = np.int8(1)
        amp = 1.0
        damp = 1.0

        for g in range(gates.shape[0]):
            q1 = gates[g, 0]
            q2 = gates[g, 1]

            sign, amp = apply_gate_with_weight(
                pauli, q1, q2,
                trans_codes, probs, is_commuting, branch_sign,
                amp_factr,
                sign, amp
            )

            # touched-qubit noise
            sign, amp, damp = apply_1q_diag_noise(pauli, q1, eta, sign, amp, damp)
            sign, amp, damp = apply_1q_diag_noise(pauli, q2, eta, sign, amp, damp)

            # or use this instead if you want full-layer noise:
            # sign, amp, damp = apply_1q_noise_layer(pauli, eta, sign, amp, damp)

        out_signs[s] = sign
        out_amps[s] = amp
        out_damps[s] = damp

        for i in range(n_qubits):
            out_paulis[s, i] = pauli[i]

    return out_paulis, out_signs, out_amps, out_damps

@njit
def obs_value_numba(P):
    o_sign = 1
    n_qubits = P.shape[0]

    for j in range(0, n_qubits, 2):
        a = P[j]
        b = P[j + 1]

        if a != b:
            return 0

        if a != 0:
            o_sign = -o_sign

    return o_sign

@njit(parallel=True)
def bond_projector_contribs(P_out, s_out, amp_out, damp_out):
    n_samples = P_out.shape[0]
    contribs = np.zeros(n_samples, dtype=np.float64)

    for k in prange(n_samples):
        val = obs_value_numba(P_out[k])
        contribs[k] = s_out[k] * amp_out[k] * damp_out[k] * val

    return contribs

@njit(parallel=True)
def contrib_stats(x):
    total_1 = 0.0
    total_2 = 0.0
    n_samples=x.shape[0]
    for i in prange(x.shape[0]):
        total_1 += x[i]
        total_2 += x[i]**2

    m1 = total_1/n_samples
    m2 = total_2/n_samples


    return m1, np.sqrt(m2 - m1**1)





