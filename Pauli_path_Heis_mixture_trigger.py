import numpy as np
import numba as nb
from numba import njit, prange

# 0->I, 1->X, 2->Y, 3->Z
CHAR_TO_INT = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
INT_TO_CHAR = np.array(['I', 'X', 'Y', 'Z'])


def encode_2q(s: str) -> int:
    return 4 * CHAR_TO_INT[s[0]] + CHAR_TO_INT[s[1]]


def decode_2q(code: int) -> str:
    return INT_TO_CHAR[code // 4] + INT_TO_CHAR[code % 4]


P2 = {}
P2['II'] = ['II']
P2['XX'] = ['XX']
P2['YY'] = ['YY']
P2['ZZ'] = ['ZZ']
P2['XI'] = ['XI', 'IX', 'YZ', 'ZY']
P2['YI'] = ['YI', 'IY', 'ZX', 'XZ']
P2['ZI'] = ['ZI', 'IZ', 'XY', 'YX']
P2['IX'] = ['IX', 'XI', 'ZY', 'YZ']
P2['IY'] = ['IY', 'YI', 'XZ', 'ZX']
P2['IZ'] = ['IZ', 'ZI', 'YX', 'XY']
P2['XY'] = ['XY', 'YX', 'ZI', 'IZ']
P2['YX'] = ['YX', 'XY', 'IZ', 'ZI']
P2['ZX'] = ['ZX', 'XZ', 'YI', 'IY']
P2['XZ'] = ['XZ', 'ZX', 'IY', 'YI']
P2['YZ'] = ['YZ', 'ZY', 'XI', 'IX']
P2['ZY'] = ['ZY', 'YZ', 'XI', 'IX']

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

    neg_pos = ['XI', 'YI', 'ZI', 'IX', 'IY', 'IZ', 'ZY']
    pos_neg = ['XY', 'YX', 'ZX', 'XZ', 'YZ']

    for key in neg_pos:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = -1
        branch_sign[in_code, 3] = +1

    for key in pos_neg:
        in_code = encode_2q(key)
        branch_sign[in_code, 2] = +1
        branch_sign[in_code, 3] = -1

    if sin2 < 0.0:
        branch_sign[:, 2] *= -1
        branch_sign[:, 3] *= -1

    return trans_codes, probs, is_commuting, branch_sign, amp_factr


def make_even_odd_layers(n_qubits: int):
    even = []
    odd = []
    for i in range(0, n_qubits - 1, 2):
        even.append((i, i + 1))
    for i in range(1, n_qubits - 1, 2):
        odd.append((i, i + 1))
    return np.array(even, dtype=np.int64), np.array(odd, dtype=np.int64)


@njit
def pauli_diag_factors_from_lambda(lam_xyz):
    n_qubits = lam_xyz.shape[0]
    eta_xyz = np.empty((n_qubits, 3), dtype=np.float64)

    for q in range(n_qubits):
        lx = lam_xyz[q, 0]
        ly = lam_xyz[q, 1]
        lz = lam_xyz[q, 2]

        ex = np.exp(-2.0 * lx)
        ey = np.exp(-2.0 * ly)
        ez = np.exp(-2.0 * lz)

        eta_xyz[q, 0] = ey*ez #ey + ez - 1.0
        eta_xyz[q, 1] = ex*ez #ex + ez - 1.0
        eta_xyz[q, 2] = ex*ey #ex + ey - 1.0

    return eta_xyz


@njit
def apply_noise(pauli, eta_xyz, damp):
    for q in range(pauli.shape[0]):
        p = pauli[q]
        if p == 1:
            damp *= eta_xyz[q, 0]
        elif p == 2:
            damp *= eta_xyz[q, 1]
        elif p == 3:
            damp *= eta_xyz[q, 2]
    return damp


@njit
def sample_branch_row(prob_row):
    r = np.random.random()
    cum = 0.0
    for k in range(4):
        p = prob_row[k]
        if p <= 0.0:
            continue
        cum += p
        if r < cum:
            return k

    best = 0
    bestp = -1.0
    for k in range(4):
        if prob_row[k] > bestp:
            bestp = prob_row[k]
            best = k
    return best


@njit
def obs_value(P):
    val = 1
    for i in range(0, P.shape[0], 2):
        a = P[i]
        b = P[i + 1]
        if a != b:
            return 0
        if a != 0:
            val = -val
    return val


@njit
def is_nonzero_path(P):
    for i in range(0, P.shape[0], 2):
        if P[i] != P[i + 1]:
            return False
    return True


@njit
def max_domain_size(P):
    n_bonds = P.shape[0] // 2
    last_bad = -1
    dmax = 0

    for b in range(n_bonds):
        if P[2 * b] != P[2 * b + 1]:
            if last_bad != -1:
                gap = b - last_bad - 1
                if gap > dmax:
                    dmax = gap
            last_bad = b

    return dmax


@njit
def free_even_layer_numba(pauli, even_gates,
                          trans_codes, probs, is_commuting, branch_sign, amp_factr,
                          sign, amp):
    for g in range(even_gates.shape[0]):
        q1 = even_gates[g, 0]
        q2 = even_gates[g, 1]
        code = 4 * pauli[q1] + pauli[q2]

        branch = sample_branch_row(probs[code])

        if not is_commuting[code]:
            amp *= amp_factr
            sign *= branch_sign[code, branch]

        out_code = trans_codes[code, branch]
        pauli[q1] = out_code // 4
        pauli[q2] = out_code % 4

    return sign, amp


@njit
def free_odd_layer_numba(pauli, odd_gates,
                         trans_codes, probs, is_commuting, branch_sign, amp_factr,
                         sign, amp):
    for g in range(odd_gates.shape[0]):
        q1 = odd_gates[g, 0]
        q2 = odd_gates[g, 1]
        code = 4 * pauli[q1] + pauli[q2]

        branch = sample_branch_row(probs[code])

        if not is_commuting[code]:
            amp *= amp_factr
            sign *= branch_sign[code, branch]

        out_code = trans_codes[code, branch]
        pauli[q1] = out_code // 4
        pauli[q2] = out_code % 4

    return sign, amp


@njit
def _build_mixture_row(prob_row, allowed, alpha):
    qmix = np.zeros(4, dtype=np.float64)
    Z = 0.0

    for k in range(4):
        if allowed[k] == 1:
            Z += prob_row[k]

    if Z <= 0.0:
        for k in range(4):
            qmix[k] = prob_row[k]
        return qmix

    omalpha = 1.0 - alpha
    for k in range(4):
        pk = prob_row[k]
        qk = alpha * pk
        if allowed[k] == 1:
            qk += omalpha * (pk / Z)
        qmix[k] = qk

    return qmix


@njit
def mixed_even_gate_update(pauli, bond_idx,
                           trans_codes, probs, is_commuting, branch_sign, amp_factr,
                           alpha_t,
                           sign, amp, log_wi):
    n_qubits = pauli.shape[0]
    n_bonds = n_qubits // 2

    i = 2 * bond_idx
    p1 = pauli[i]
    p2 = pauli[i + 1]

    if p1 == p2:
        return sign, amp, log_wi

    code = 4 * p1 + p2
    prob_row = probs[code]

    has_left_good = False
    has_right_good = False
    left_boundary_symbol = -1
    right_boundary_symbol = -1

    if bond_idx - 1 >= 0:
        li = 2 * (bond_idx - 1)
        if pauli[li] == pauli[li + 1]:
            has_left_good = True
            left_boundary_symbol = pauli[li + 1]

    if bond_idx + 1 < n_bonds:
        ri = 2 * (bond_idx + 1)
        if pauli[ri] == pauli[ri + 1]:
            has_right_good = True
            right_boundary_symbol = pauli[ri]

    allowed = np.zeros(4, dtype=np.int8)

    for k in range(4):
        pk = prob_row[k]
        if pk <= 0.0:
            continue

        out_code = trans_codes[code, k]
        left_out = out_code // 4
        right_out = out_code % 4

        ok = True
        if has_left_good and left_out == left_boundary_symbol:
            ok = False
        if has_right_good and right_out == right_boundary_symbol:
            ok = False

        if ok:
            allowed[k] = 1

    qmix = _build_mixture_row(prob_row, allowed, alpha_t)
    branch = sample_branch_row(qmix)

    pk = prob_row[branch]
    qk = qmix[branch]
    if pk > 0.0 and qk > 0.0:
        log_wi += np.log(pk) - np.log(qk)

    if not is_commuting[code]:
        amp *= amp_factr
        sign *= branch_sign[code, branch]

    out_code = trans_codes[code, branch]
    pauli[i] = out_code // 4
    pauli[i + 1] = out_code % 4

    return sign, amp, log_wi


@njit
def mixed_even_layer_numba(pauli,
                           trans_codes, probs, is_commuting, branch_sign, amp_factr,
                           alpha_t,
                           sign, amp, log_wi):
    n_bonds = pauli.shape[0] // 2
    for b in range(n_bonds):
        sign, amp, log_wi = mixed_even_gate_update(
            pauli, b,
            trans_codes, probs, is_commuting, branch_sign, amp_factr,
            alpha_t,
            sign, amp, log_wi
        )
    return sign, amp, log_wi


@njit
def mixed_odd_gate_update(pauli, q1, q2,
                          trans_codes, probs, is_commuting, branch_sign, amp_factr,
                          alpha_t,
                          sign, amp, log_wi):
    p1 = pauli[q1]
    p2 = pauli[q2]

    if p1 == p2:
        return sign, amp, log_wi

    code = 4 * p1 + p2
    prob_row = probs[code]
    left_neighbor = pauli[q1 - 1]

    allowed = np.zeros(4, dtype=np.int8)

    for k in range(4):
        pk = prob_row[k]
        if pk <= 0.0:
            continue

        out_code = trans_codes[code, k]
        left_out = out_code // 4
        if left_out == left_neighbor:
            allowed[k] = 1

    qmix = _build_mixture_row(prob_row, allowed, alpha_t)
    branch = sample_branch_row(qmix)

    pk = prob_row[branch]
    qk = qmix[branch]
    if pk > 0.0 and qk > 0.0:
        log_wi += np.log(pk) - np.log(qk)

    if not is_commuting[code]:
        amp *= amp_factr
        sign *= branch_sign[code, branch]

    out_code = trans_codes[code, branch]
    pauli[q1] = out_code // 4
    pauli[q2] = out_code % 4

    return sign, amp, log_wi


@njit
def mixed_odd_layer_numba(pauli, odd_gates,
                          trans_codes, probs, is_commuting, branch_sign, amp_factr,
                          alpha_t,
                          sign, amp, log_wi):
    for g in range(odd_gates.shape[0]):
        q1 = odd_gates[g, 0]
        q2 = odd_gates[g, 1]
        sign, amp, log_wi = mixed_odd_gate_update(
            pauli, q1, q2,
            trans_codes, probs, is_commuting, branch_sign, amp_factr,
            alpha_t,
            sign, amp, log_wi
        )
    return sign, amp, log_wi


@njit
def evolve_one_triggered_mixture_layers(init_pauli,
                                        even_gates, odd_gates,
                                        trans_codes, probs, is_commuting, branch_sign,
                                        amp_factr,
                                        eta_xyz,
                                        alpha_schedule,
                                        n_steps):
    """
    Mixture proposal is used from the start with alpha_schedule[t].
    Once the trigger d == remaining_steps fires, all remaining layers use
    pure restricted sampling (alpha_t = 0.0).
    """
    pauli = init_pauli.copy()

    sign = np.int8(1)
    amp = 1.0
    damp = 1.0
    log_wi = 0.0

    triggered = False

    for t in range(n_steps):
        d = max_domain_size(pauli)
        R = n_steps - t

        if d == R:
            triggered = True

        if triggered:
            alpha_t = 0.0
        else:
            alpha_t = alpha_schedule[t]

        sign, amp, log_wi = mixed_even_layer_numba(
            pauli,
            trans_codes, probs, is_commuting, branch_sign, amp_factr,
            alpha_t,
            sign, amp, log_wi
        )

        damp = apply_noise(pauli, eta_xyz, damp)

        sign, amp, log_wi = mixed_odd_layer_numba(
            pauli, odd_gates,
            trans_codes, probs, is_commuting, branch_sign, amp_factr,
            alpha_t,
            sign, amp, log_wi
        )

        damp = apply_noise(pauli, eta_xyz, damp)

    wi = np.exp(log_wi)
    return pauli, sign, amp, wi, damp


@njit(parallel=True)
def evolve_many_triggered_mixture_layers(init_pauli,
                                         even_gates, odd_gates,
                                         trans_codes, probs, is_commuting, branch_sign,
                                         amp_factr,
                                         eta_xyz,
                                         alpha_schedule,
                                         n_steps,
                                         n_samples):
    n_qubits = init_pauli.shape[0]

    P_out = np.empty((n_samples, n_qubits), dtype=np.int8)
    s_out = np.empty(n_samples, dtype=np.int8)
    amp_out = np.empty(n_samples, dtype=np.float64)
    wi_out = np.empty(n_samples, dtype=np.float64)
    damp_out = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        p, s, a, w, d = evolve_one_triggered_mixture_layers(
            init_pauli,
            even_gates, odd_gates,
            trans_codes, probs, is_commuting, branch_sign,
            amp_factr,
            eta_xyz,
            alpha_schedule,
            n_steps
        )
        P_out[i] = p
        s_out[i] = s
        amp_out[i] = a
        wi_out[i] = w
        damp_out[i] = d

    return P_out, s_out, amp_out, wi_out, damp_out


@njit(parallel=True)
def bond_projector_contribs_is(P_out, s_out, amp_out, wi_out, damp_out):
    n_samples = P_out.shape[0]
    contribs = np.zeros(n_samples, dtype=np.float64)

    for k in prange(n_samples):
        val = obs_value(P_out[k])
        contribs[k] = s_out[k] * amp_out[k] * wi_out[k] * damp_out[k] * val

    return contribs


@njit(parallel=True)
def contrib_stats(x):
    total_1 = 0.0
    total_2 = 0.0
    n_samples = x.shape[0]
    for i in prange(n_samples):
        total_1 += x[i]
        total_2 += x[i] ** 2

    m1 = total_1 / n_samples
    m2 = total_2 / n_samples
    return m1, np.sqrt(m2 - m1 ** 2)


@njit
def count_pm_is(P_out, s_out):
    n = P_out.shape[0]
    n_p = 0
    n_m = 0

    for j in range(n):
        val = obs_value(P_out[j]) * s_out[j]
        if val == 1:
            n_p += 1
        elif val == -1:
            n_m += 1

    return n_p, n_m


@njit
def split_logs_pm_with_iw(P_out, s_out, amp_out, wi_out, damp_out):
    n_p, n_m = count_pm_is(P_out, s_out)

    amp_p = np.empty(n_p, dtype=np.float64)
    damp_p = np.empty(n_p, dtype=np.float64)
    amp_m = np.empty(n_m, dtype=np.float64)
    damp_m = np.empty(n_m, dtype=np.float64)

    ip = 0
    im = 0

    for j in range(P_out.shape[0]):
        val = obs_value(P_out[j]) * s_out[j]

        if val == 1:
            amp_p[ip] = np.log(amp_out[j]) + np.log(wi_out[j])
            damp_p[ip] = -np.log(damp_out[j])
            ip += 1
        elif val == -1:
            amp_m[im] = np.log(amp_out[j]) + np.log(wi_out[j])
            damp_m[im] = -np.log(damp_out[j])
            im += 1

    return amp_p, damp_p, amp_m, damp_m
