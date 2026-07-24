"""
Full-Trotter-layer Pauli path sampler for the 1D Heisenberg brickwall model.

This module replaces gate-by-gate unitary branch sampling with exact one-step
1-norm sampling over a full even+odd Trotter layer, under the working assumption
that, for a single incoming Pauli string, the one-step even+odd branch expansion
has no path recombination.

Noise model:
    diagonal one-qubit Pauli-basis noise after the even layer and after the odd
    layer. The unitary sampler samples only from the unitary 1-norm proposal;
    damping is tracked separately by multiplying the sampled even Pauli and final
    Pauli damping factors.

Encoding:
    0=I, 1=X, 2=Y, 3=Z.
    2q blocked code = 4*p_left + p_right.
"""

import numpy as np
import numba as nb
from numba import njit, prange

# -----------------------------------------------------------------------------
# Python-side table builders
# -----------------------------------------------------------------------------

CHAR_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
INT_TO_CHAR = np.array(["I", "X", "Y", "Z"])


def encode_2q(s: str) -> int:
    return 4 * CHAR_TO_INT[s[0]] + CHAR_TO_INT[s[1]]


def decode_2q(code: int) -> str:
    return INT_TO_CHAR[code // 4] + INT_TO_CHAR[code % 4]


P2 = {}
P2["II"] = ["II"]
P2["XX"] = ["XX"]
P2["YY"] = ["YY"]
P2["ZZ"] = ["ZZ"]
P2["XI"] = ["XI", "IX", "YZ", "ZY"]
P2["YI"] = ["YI", "IY", "ZX", "XZ"]
P2["ZI"] = ["ZI", "IZ", "XY", "YX"]
P2["IX"] = ["IX", "XI", "ZY", "YZ"]
P2["IY"] = ["IY", "YI", "XZ", "ZX"]
P2["IZ"] = ["IZ", "ZI", "YX", "XY"]
P2["XY"] = ["XY", "YX", "ZI", "IZ"]
P2["YX"] = ["YX", "XY", "IZ", "ZI"]
P2["ZX"] = ["ZX", "XZ", "YI", "IY"]
P2["XZ"] = ["XZ", "ZX", "IY", "YI"]
P2["YZ"] = ["YZ", "ZY", "XI", "IX"]
P2["ZY"] = ["ZY", "YZ", "XI", "IX"]

COM = ["II", "XX", "YY", "ZZ"]


def build_full_layer_tables(phi: float):
    """
    Build all 2-qubit transition tables needed by the full-layer sampler.

    Returns
    -------
    trans_codes : (16,4) int8
    signed_coeffs : (16,4) float64
        True signed Heisenberg coefficients for the local 2q update.
    abs_coeffs : (16,4) float64
        Absolute values of signed_coeffs.
    n_branches : (16,) int64
        1 for commuting codes, 4 otherwise.
    is_commuting : (16,) bool
    branch_sign : (16,4) int8
        Kept for compatibility/debugging.
    amp_factr : float
        1 + abs(sin(2phi)).
    """
    trans_codes = np.full((16, 4), -1, dtype=np.int8)
    signed_coeffs = np.zeros((16, 4), dtype=np.float64)
    abs_coeffs = np.zeros((16, 4), dtype=np.float64)
    n_branches = np.zeros(16, dtype=np.int64)
    is_commuting = np.zeros(16, dtype=np.bool_)
    branch_sign = np.ones((16, 4), dtype=np.int8)

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

    # Same sign convention as Pauli_path_Heis.py.
    neg_pos = ["XI", "YI", "ZI", "IX", "IY", "IZ", "ZY"]
    pos_neg = ["XY", "YX", "ZX", "XZ", "YZ"]

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

    for code in range(16):
        if is_commuting[code]:
            # Deterministic branch. trans_codes[code,0] should already be code.
            trans_codes[code, 0] = np.int8(code)
            signed_coeffs[code, 0] = 1.0
            abs_coeffs[code, 0] = 1.0
            n_branches[code] = 1
        else:
            signed_coeffs[code, 0] = c * c
            signed_coeffs[code, 1] = s * s
            signed_coeffs[code, 2] = 0.5 * abs_sin2 * int(branch_sign[code, 2])
            signed_coeffs[code, 3] = 0.5 * abs_sin2 * int(branch_sign[code, 3])
            for k in range(4):
                abs_coeffs[code, k] = abs(signed_coeffs[code, k])
            n_branches[code] = 4

    return trans_codes, signed_coeffs, abs_coeffs, n_branches, is_commuting, branch_sign, amp_factr


# -----------------------------------------------------------------------------
# Noise helper
# -----------------------------------------------------------------------------

@njit
def pauli_diag_factors_from_lambda(lam_xyz):
    """
    lam_xyz[q,0] = lambda_x on qubit q
    lam_xyz[q,1] = lambda_y on qubit q
    lam_xyz[q,2] = lambda_z on qubit q

    Returns eta_xyz[q,0:3] damping factors for X,Y,Z on qubit q.
    """
    n_qubits = lam_xyz.shape[0]
    eta_xyz = np.empty((n_qubits, 3), dtype=np.float64)

    for q in range(n_qubits):
        lx = lam_xyz[q, 0]
        ly = lam_xyz[q, 1]
        lz = lam_xyz[q, 2]

        ex = np.exp(-2.0 * lx)
        ey = np.exp(-2.0 * ly)
        ez = np.exp(-2.0 * lz)

        eta_xyz[q, 0] = ey + ez - 1.0  # X
        eta_xyz[q, 1] = ex + ez - 1.0  # Y
        eta_xyz[q, 2] = ex + ey - 1.0  # Z

    return eta_xyz


@njit
def pauli_noise_factor(P, eta_xyz):
    """Diagonal Pauli-basis one-qubit noise factor for a full Pauli string."""
    damp = 1.0
    for q in range(P.shape[0]):
        p = P[q]
        if p == 1:
            damp *= eta_xyz[q, 0]
        elif p == 2:
            damp *= eta_xyz[q, 1]
        elif p == 3:
            damp *= eta_xyz[q, 2]
    return damp


# -----------------------------------------------------------------------------
# Low-level full-layer 1-norm sampler
# -----------------------------------------------------------------------------



@njit
def _pauli_to_blocked_numba(P):
    B = np.empty(P.shape[0] // 2, dtype=np.int8)
    pauli_to_blocked_inplace(P, B)
    return B

@njit
def blocked_to_pauli_inplace(B, P):
    for j in range(B.shape[0]):
        code = int(B[j])
        P[2 * j] = np.int8(code // 4)
        P[2 * j + 1] = np.int8(code % 4)


@njit
def pauli_to_blocked_inplace(P, B):
    for j in range(B.shape[0]):
        B[j] = np.int8(4 * int(P[2 * j]) + int(P[2 * j + 1]))


@njit
def odd_pair_l1_factor(outL, outR, abs_coeffs, n_branches):
    """
    Sum of absolute odd-gate coefficients on the middle pair of neighboring
    blocked sites outL=(a,b), outR=(c,d).
    """
    b = int(outL) % 4
    c = int(outR) // 4
    mid = 4 * b + c

    if n_branches[mid] == 1:
        return abs_coeffs[mid, 0]
    return abs_coeffs[mid, 0] + abs_coeffs[mid, 1] + abs_coeffs[mid, 2] + abs_coeffs[mid, 3]


@njit
def build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts):
    """Fill even-layer branch tables for current input Pauli."""
    L = P_in.shape[0] // 2
    for j in range(L):
        code = 4 * int(P_in[2 * j]) + int(P_in[2 * j + 1])
        k_counts[j] = n_branches[code]
        for k in range(4):
            outs[j, k] = -1
            coeffs[j, k] = 0.0
            absw[j, k] = 0.0
        for k in range(n_branches[code]):
            outs[j, k] = trans_codes[code, k]
            coeffs[j, k] = signed_coeffs[code, k]
            absw[j, k] = abs_coeffs[code, k]


@njit
def build_right_messages_inplace(outs, absw, k_counts, abs_coeffs, n_branches, R):
    """Right messages for exact one-step 1-norm sampling over even branches."""
    L = k_counts.shape[0]
    for j in range(L):
        for k in range(4):
            R[j, k] = 0.0

    last = L - 1
    for r in range(k_counts[last]):
        R[last, r] = absw[last, r]

    for j in range(L - 2, -1, -1):
        for r_cur in range(k_counts[j]):
            total = 0.0
            for r_next in range(k_counts[j + 1]):
                f = odd_pair_l1_factor(outs[j, r_cur], outs[j + 1, r_next],
                                       abs_coeffs, n_branches)
                total += f * R[j + 1, r_next]
            R[j, r_cur] = absw[j, r_cur] * total


@njit
def sample_from_weights(weights, k):
    Z = 0.0
    for i in range(k):
        Z += weights[i]

    if Z <= 0.0:
        return 0, 0.0

    x = np.random.random() * Z
    c = 0.0
    for i in range(k):
        c += weights[i]
        if x < c:
            return i, weights[i] / Z

    return k - 1, weights[k - 1] / Z


@njit
def sample_odd_branch_for_gate(P, q, trans_codes, signed_coeffs, abs_coeffs, n_branches):
    """
    Sample one odd-gate branch from abs local coefficients, update P in-place,
    and return signed local coefficient and sample probability.
    """
    code = 4 * int(P[q]) + int(P[q + 1])
    k = n_branches[code]

    weights = np.zeros(4, dtype=np.float64)
    for r in range(k):
        weights[r] = abs_coeffs[code, r]

    branch, p_branch = sample_from_weights(weights, k)
    out_code = int(trans_codes[code, branch])
    P[q] = np.int8(out_code // 4)
    P[q + 1] = np.int8(out_code % 4)

    return signed_coeffs[code, branch], p_branch


@njit
def sample_one_full_trotter_layer(
    P_in,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
):
    """
    Sample one full even+odd Trotter layer from the exact unitary 1-norm
    proposal, under the no-recombination assumption.

    Parameters
    ----------
    P_in : int8[N_q]
        Input Pauli string. Not modified.

    Returns
    -------
    P_even : int8[N_q]
        Sampled Pauli after the even layer.
    P_out : int8[N_q]
        Sampled final Pauli after the odd layer.
    coeff_sign : int8
        Sign of the sampled full-step unitary coefficient.
    step_l1 : float64
        Exact unitary 1-norm of the full one-step expansion from this P_in.
    sample_prob : float64
        Probability of the sampled full branch under the proposal.
    coeff_even : float64
        Signed even-layer coefficient for diagnostics.
    coeff_odd : float64
        Signed odd-layer coefficient for diagnostics.
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2

    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    R = np.empty((L, 4), dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)
    build_right_messages_inplace(outs, absw, k_counts, abs_coeffs, n_branches, R)

    step_l1 = 0.0
    for r in range(k_counts[0]):
        step_l1 += R[0, r]

    branch_choices = np.empty(L, dtype=np.int64)
    weights = np.zeros(4, dtype=np.float64)
    sample_prob_even = 1.0

    # first even branch
    for r in range(4):
        weights[r] = 0.0
    for r in range(k_counts[0]):
        weights[r] = R[0, r]
    r0, p0 = sample_from_weights(weights, k_counts[0])
    branch_choices[0] = r0
    sample_prob_even *= p0

    # remaining even branches, conditioned on previous
    for j in range(1, L):
        r_prev = branch_choices[j - 1]
        for r in range(4):
            weights[r] = 0.0
        for r_cur in range(k_counts[j]):
            f = odd_pair_l1_factor(outs[j - 1, r_prev], outs[j, r_cur],
                                   abs_coeffs, n_branches)
            weights[r_cur] = f * R[j, r_cur]
        rj, pj = sample_from_weights(weights, k_counts[j])
        branch_choices[j] = rj
        sample_prob_even *= pj

    # Construct P_even and coeff_even.
    B_even = np.empty(L, dtype=np.int8)
    coeff_even = 1.0
    for j in range(L):
        r = branch_choices[j]
        B_even[j] = outs[j, r]
        coeff_even *= coeffs[j, r]

    P_even = np.empty(n_qubits, dtype=np.int8)
    blocked_to_pauli_inplace(B_even, P_even)

    # Sample the odd-layer local descendants from abs local coefficients.
    P_out = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P_out[q] = P_even[q]

    coeff_odd = 1.0
    sample_prob_odd = 1.0
    for q in range(1, n_qubits - 1, 2):
        c_odd, p_odd = sample_odd_branch_for_gate(
            P_out, q, trans_codes, signed_coeffs, abs_coeffs, n_branches
        )
        coeff_odd *= c_odd
        sample_prob_odd *= p_odd

    coeff_full = coeff_even * coeff_odd
    coeff_sign = np.int8(1)
    if coeff_full < 0.0:
        coeff_sign = np.int8(-1)

    sample_prob = sample_prob_even * sample_prob_odd
    return P_even, P_out, coeff_sign, step_l1, sample_prob, coeff_even, coeff_odd


@njit
def _active_block_window(P_in):
    """Return [start, end) blocked-site window padded for one odd-layer spread."""
    L = P_in.shape[0] // 2
    first = L
    last = -1
    for j in range(L):
        if P_in[2 * j] != 0 or P_in[2 * j + 1] != 0:
            if j < first:
                first = j
            if j > last:
                last = j

    if last < 0:
        return 0, 0

    start = first - 1
    if start < 0:
        start = 0
    end = last + 2
    if end > L:
        end = L
    return start, end


@njit
def sample_one_full_trotter_layer_lightcone(
    P_in,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
):
    """
    Sample one full layer on the active blocked light-cone window only.

    The current full-layer order is even gates followed by odd inter-block gates.
    Padding the non-identity blocked interval by one block on each side includes
    every odd-boundary gate through which support can spread during this layer.
    Identity-only tails contribute deterministic unit 1-norm and no noise.
    """
    n_qubits = P_in.shape[0]
    start_b, end_b = _active_block_window(P_in)

    P_even = np.empty(n_qubits, dtype=np.int8)
    P_out = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P_even[q] = P_in[q]
        P_out[q] = P_in[q]

    if end_b <= start_b:
        return P_even, P_out, np.int8(1), 1.0, 1.0, 1.0, 1.0

    start_q = 2 * start_b
    end_q = 2 * end_b
    n_local = end_q - start_q
    P_slice = np.empty(n_local, dtype=np.int8)
    for q in range(n_local):
        P_slice[q] = P_in[start_q + q]

    Pe_slice, Po_slice, step_sign, step_l1, sample_prob, coeff_even, coeff_odd = sample_one_full_trotter_layer(
        P_slice, trans_codes, signed_coeffs, abs_coeffs, n_branches
    )

    for q in range(n_local):
        P_even[start_q + q] = Pe_slice[q]
        P_out[start_q + q] = Po_slice[q]

    return P_even, P_out, step_sign, step_l1, sample_prob, coeff_even, coeff_odd


# -----------------------------------------------------------------------------
# Multi-step evolution
# -----------------------------------------------------------------------------

@njit
def evolve_one_full_layer_sampling(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
):
    """
    Evolve one path for n_steps. Each Trotter step samples from the exact
    one-step unitary 1-norm proposal and tracks damping separately.
    """
    n_qubits = init_pauli.shape[0]
    P = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P[q] = init_pauli[q]

    sign = np.int8(1)
    amp = 1.0
    damp = 1.0

    # Diagnostics from last step.
    P_even_last = np.empty(n_qubits, dtype=np.int8)

    for _ in range(n_steps):
        P_even, P_out, step_sign, step_l1, sample_prob, coeff_even, coeff_odd = sample_one_full_trotter_layer_lightcone(
            P, trans_codes, signed_coeffs, abs_coeffs, n_branches
        )

        # Noise after even layer and after odd layer.
        damp *= pauli_noise_factor(P_even, eta_xyz)
        damp *= pauli_noise_factor(P_out, eta_xyz)

        sign = np.int8(sign * step_sign)
        amp *= step_l1

        for q in range(n_qubits):
            P_even_last[q] = P_even[q]
            P[q] = P_out[q]

    return P, P_even_last, sign, amp, damp


@njit(parallel=True)
def evolve_many_full_layer_sampling(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
    n_samples,
):
    """
    Parallel many-path driver.

    Returns
    -------
    P_out : int8[n_samples, n_qubits]
    P_even_last : int8[n_samples, n_qubits]
        Even-layer Pauli from the last Trotter step, useful for diagnostics.
    s_out : int8[n_samples]
    amp_out : float64[n_samples]
    damp_out : float64[n_samples]
    """
    n_qubits = init_pauli.shape[0]
    P_out = np.empty((n_samples, n_qubits), dtype=np.int8)
    P_even_last = np.empty((n_samples, n_qubits), dtype=np.int8)
    s_out = np.empty(n_samples, dtype=np.int8)
    amp_out = np.empty(n_samples, dtype=np.float64)
    damp_out = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        P, Pe, s, a, d = evolve_one_full_layer_sampling(
            init_pauli, trans_codes, signed_coeffs, abs_coeffs, n_branches,
            eta_xyz, n_steps
        )
        for q in range(n_qubits):
            P_out[i, q] = P[q]
            P_even_last[i, q] = Pe[q]
        s_out[i] = s
        amp_out[i] = a
        damp_out[i] = d

    return P_out, P_even_last, s_out, amp_out, damp_out


# -----------------------------------------------------------------------------
# Observables and stats helpers
# -----------------------------------------------------------------------------

@njit
def obs_value(P):
    """Singlet-pair product observable on even bonds."""
    val = 1
    for q in range(0, P.shape[0], 2):
        a = P[q]
        b = P[q + 1]
        if a != b:
            return 0
        if a != 0:
            val = -val
    return val


@njit(parallel=True)
def bond_projector_contribs_full_layer(P_out, s_out, amp_out, damp_out):
    n = P_out.shape[0]
    contribs = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        contribs[i] = s_out[i] * amp_out[i] * damp_out[i] * obs_value(P_out[i])
    return contribs


@njit(parallel=True)
def contrib_stats(x):
    total_1 = 0.0
    total_2 = 0.0
    n = x.shape[0]
    for i in prange(n):
        total_1 += x[i]
        total_2 += x[i] * x[i]
    m1 = total_1 / n
    m2 = total_2 / n
    var = m2 - m1 * m1
    if var < 0.0:
        var = 0.0
    return m1, np.sqrt(var)


@njit
def max_domain_size(P):
    """Max number of good even bonds between neighboring bad even bonds."""
    n_bonds = P.shape[0] // 2
    last_bad = -1
    dmax = 0
    for b in range(n_bonds):
        if P[2*b] != P[2*b + 1]:
            if last_bad != -1:
                gap = b - last_bad - 1
                if gap > dmax:
                    dmax = gap
            last_bad = b
    return dmax

# =============================================================================
# Restricted full-layer mixture sampler with alpha_t schedule and trigger
# =============================================================================

@njit
def _triple_parity_signature_bad_block(P, b0, b1):
    """
    Parity signature for a contiguous bad block of even-bond indices [b0,b1].
    Current convention: XOR parity of X/Y/Z counts over all single-qubit Paulis
    in the block. Signature bits: X=1, Y=2, Z=4. Correct triple parity => 0.
    """
    sig = 0
    for b in range(b0, b1 + 1):
        p = P[2*b]
        q = P[2*b + 1]
        if p == 1:
            sig ^= 1
        elif p == 2:
            sig ^= 2
        elif p == 3:
            sig ^= 4
        if q == 1:
            sig ^= 1
        elif q == 2:
            sig ^= 2
        elif q == 3:
            sig ^= 4
    return sig


@njit
def find_active_zones_triple_parity(P, zone_starts, zone_ends, zone_domains):
    """
    Find active restricted zones on the even-bond lattice.

    A good even bond has P[2b] == P[2b+1]. Bad blocks are contiguous bad bonds.
    - If a bad block has zero triple-parity signature, it is its own active zone.
    - If it is parity-breaking, open a zone and include later bad blocks plus
      intervening good runs until the accumulated signature returns to zero.

    Returns n_zones. Arrays are filled for first n_zones entries.
    """
    n_bonds = P.shape[0] // 2
    n_blocks = 0
    starts = np.empty(n_bonds, dtype=np.int64)
    ends = np.empty(n_bonds, dtype=np.int64)
    sigs = np.empty(n_bonds, dtype=np.int64)

    b = 0
    while b < n_bonds:
        if P[2*b] == P[2*b + 1]:
            b += 1
            continue
        s = b
        while b + 1 < n_bonds and P[2*(b+1)] != P[2*(b+1) + 1]:
            b += 1
        e = b
        starts[n_blocks] = s
        ends[n_blocks] = e
        sigs[n_blocks] = _triple_parity_signature_bad_block(P, s, e)
        n_blocks += 1
        b += 1

    nz = 0
    i = 0
    while i < n_blocks:
        sig = sigs[i]
        if sig == 0:
            zone_starts[nz] = starts[i]
            zone_ends[nz] = ends[i]
            zone_domains[nz] = 0
            nz += 1
            i += 1
        else:
            zstart = starts[i]
            last_end = ends[i]
            acc = sig
            dmax = 0
            i += 1
            while i < n_blocks and acc != 0:
                gap = starts[i] - last_end - 1
                if gap > dmax:
                    dmax = gap
                acc ^= sigs[i]
                last_end = ends[i]
                i += 1
            zone_starts[nz] = zstart
            zone_ends[nz] = last_end
            zone_domains[nz] = dmax
            nz += 1
    return nz


@njit
def max_domain_size_triple_parity(P):
    n_bonds = P.shape[0] // 2
    zs = np.empty(n_bonds, dtype=np.int64)
    ze = np.empty(n_bonds, dtype=np.int64)
    zd = np.empty(n_bonds, dtype=np.int64)
    nz = find_active_zones_triple_parity(P, zs, ze, zd)
    dmax = 0
    for i in range(nz):
        if zd[i] > dmax:
            dmax = zd[i]
    return dmax


@njit
def fill_active_masks_from_zones(P, active_site, active_edge):
    """Fill active even-bond site mask and odd-edge mask from triple-parity zones."""
    L = P.shape[0] // 2
    for j in range(L):
        active_site[j] = False
    for j in range(L - 1):
        active_edge[j] = False

    zs = np.empty(L, dtype=np.int64)
    ze = np.empty(L, dtype=np.int64)
    zd = np.empty(L, dtype=np.int64)
    nz = find_active_zones_triple_parity(P, zs, ze, zd)
    for k in range(nz):
        s = zs[k]
        e = ze[k]
        for j in range(s, e + 1):
            active_site[j] = True
        for j in range(s, e):
            active_edge[j] = True


@njit
def odd_pair_l1_factor_masked(outL, outR, edge_active, abs_coeffs, n_branches):
    b = int(outL) % 4
    c = int(outR) // 4
    mid = 4 * b + c
    if edge_active:
        if n_branches[mid] == 1:
            return abs_coeffs[mid, 0]
        return abs_coeffs[mid, 0] + abs_coeffs[mid, 1] + abs_coeffs[mid, 2] + abs_coeffs[mid, 3]
    else:
        # Restricted inactive odd edge: freeze to branch 0 only.
        return abs_coeffs[mid, 0]


@njit
def build_right_messages_masked_inplace(outs, absw, k_counts, B_in,
                                        active_site, active_edge,
                                        abs_coeffs, n_branches, R):
    L = k_counts.shape[0]
    for j in range(L):
        for k in range(4):
            R[j, k] = 0.0

    last = L - 1
    for r in range(k_counts[last]):
        if active_site[last] or outs[last, r] == B_in[last]:
            R[last, r] = absw[last, r]

    for j in range(L - 2, -1, -1):
        for r_cur in range(k_counts[j]):
            if (not active_site[j]) and outs[j, r_cur] != B_in[j]:
                R[j, r_cur] = 0.0
                continue
            total = 0.0
            for r_next in range(k_counts[j + 1]):
                if (not active_site[j + 1]) and outs[j + 1, r_next] != B_in[j + 1]:
                    continue
                f = odd_pair_l1_factor_masked(outs[j, r_cur], outs[j + 1, r_next],
                                              active_edge[j], abs_coeffs, n_branches)
                total += f * R[j + 1, r_next]
            R[j, r_cur] = absw[j, r_cur] * total


@njit
def sample_odd_branch_for_gate_masked(P, q, edge_active, trans_codes,
                                      signed_coeffs, abs_coeffs, n_branches):
    code = 4 * int(P[q]) + int(P[q + 1])
    if not edge_active:
        out_code = int(trans_codes[code, 0])
        P[q] = np.int8(out_code // 4)
        P[q + 1] = np.int8(out_code % 4)
        return signed_coeffs[code, 0], 1.0, 0

    k = n_branches[code]
    weights = np.zeros(4, dtype=np.float64)
    for r in range(k):
        weights[r] = abs_coeffs[code, r]
    branch, p_branch = sample_from_weights(weights, k)
    out_code = int(trans_codes[code, branch])
    P[q] = np.int8(out_code // 4)
    P[q + 1] = np.int8(out_code % 4)
    return signed_coeffs[code, branch], p_branch, branch


@njit
def _branch_allowed_under_restriction(branch_choices, B_even, B_in, odd_branch_choices,
                                      active_site, active_edge):
    L = B_in.shape[0]
    for j in range(L):
        if (not active_site[j]) and B_even[j] != B_in[j]:
            return False
    for j in range(L - 1):
        if (not active_edge[j]) and odd_branch_choices[j] != 0:
            return False
    return True


@njit
def sample_one_full_trotter_layer_masked(
    P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches,
    active_site, active_edge,
):
    """Sample one layer from masked/restricted subset distribution."""
    n_qubits = P_in.shape[0]
    L = n_qubits // 2
    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    R = np.empty((L, 4), dtype=np.float64)
    B_in = np.empty(L, dtype=np.int8)
    pauli_to_blocked_inplace(P_in, B_in)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)
    build_right_messages_masked_inplace(outs, absw, k_counts, B_in, active_site,
                                        active_edge, abs_coeffs, n_branches, R)
    Z = 0.0
    for r in range(k_counts[0]):
        Z += R[0, r]

    branch_choices = np.empty(L, dtype=np.int64)
    weights = np.zeros(4, dtype=np.float64)
    sample_prob_even = 1.0

    for r in range(4):
        weights[r] = 0.0
    for r in range(k_counts[0]):
        weights[r] = R[0, r]
    r0, p0 = sample_from_weights(weights, k_counts[0])
    branch_choices[0] = r0
    sample_prob_even *= p0

    for j in range(1, L):
        r_prev = branch_choices[j - 1]
        for r in range(4):
            weights[r] = 0.0
        for r_cur in range(k_counts[j]):
            if R[j, r_cur] <= 0.0:
                weights[r_cur] = 0.0
            else:
                f = odd_pair_l1_factor_masked(outs[j - 1, r_prev], outs[j, r_cur],
                                              active_edge[j - 1], abs_coeffs, n_branches)
                weights[r_cur] = f * R[j, r_cur]
        rj, pj = sample_from_weights(weights, k_counts[j])
        branch_choices[j] = rj
        sample_prob_even *= pj

    B_even = np.empty(L, dtype=np.int8)
    coeff_even = 1.0
    for j in range(L):
        r = branch_choices[j]
        B_even[j] = outs[j, r]
        coeff_even *= coeffs[j, r]

    P_even = np.empty(n_qubits, dtype=np.int8)
    blocked_to_pauli_inplace(B_even, P_even)
    P_out = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P_out[q] = P_even[q]

    coeff_odd = 1.0
    sample_prob_odd = 1.0
    odd_branch_choices = np.zeros(L - 1, dtype=np.int64)
    edge = 0
    for q in range(1, n_qubits - 1, 2):
        c_odd, p_odd, br = sample_odd_branch_for_gate_masked(
            P_out, q, active_edge[edge], trans_codes, signed_coeffs, abs_coeffs, n_branches
        )
        coeff_odd *= c_odd
        sample_prob_odd *= p_odd
        odd_branch_choices[edge] = br
        edge += 1

    coeff_full = coeff_even * coeff_odd
    step_sign = np.int8(1)
    if coeff_full < 0.0:
        step_sign = np.int8(-1)
    return P_even, P_out, step_sign, Z, sample_prob_even * sample_prob_odd, coeff_even, coeff_odd, B_even, odd_branch_choices



@njit
def is_all_good_pauli(P):
    L = P.shape[0] // 2
    for j in range(L):
        if P[2*j] != P[2*j + 1]:
            return False
    return True


@njit
def _code_left(code):
    return int(code) // 4


@njit
def _code_right(code):
    return int(code) % 4


@njit
def build_all_good_right_messages(outs, absw, k_counts, trans_codes, abs_coeffs, n_branches, Rgood):
    """
    Rgood[j, r_j, left_val] = total absolute weight from site j onward,
    conditioned on choosing even branch r_j at site j and the final left
    qubit of blocked site j being left_val.

    This enforces that every final even-bond block is good: left == right.
    """
    L = k_counts.shape[0]
    for j in range(L):
        for r in range(4):
            for a in range(4):
                Rgood[j, r, a] = 0.0

    last = L - 1
    for r in range(k_counts[last]):
        code = int(outs[last, r])
        right = _code_right(code)
        for left in range(4):
            if left == right:
                Rgood[last, r, left] = absw[last, r]

    for j in range(L - 2, -1, -1):
        for r_cur in range(k_counts[j]):
            code_cur = int(outs[j, r_cur])
            b = _code_right(code_cur)
            for left_val in range(4):
                total = 0.0
                for r_next in range(k_counts[j + 1]):
                    code_next = int(outs[j + 1, r_next])
                    c = _code_left(code_next)
                    mid = 4 * b + c
                    kodd = n_branches[mid]
                    for ob in range(kodd):
                        out_mid = int(trans_codes[mid, ob])
                        u = _code_left(out_mid)   # final right qubit of site j
                        v = _code_right(out_mid)  # final left qubit of site j+1
                        if u == left_val:
                            total += abs_coeffs[mid, ob] * Rgood[j + 1, r_next, v]
                Rgood[j, r_cur, left_val] = absw[j, r_cur] * total


@njit
def sample_one_full_trotter_layer_all_good(
    P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches,
):
    """
    Exact restricted sampler for one full layer conditioned on final P_out
    being all good even bonds, i.e. P_out[2*j] == P_out[2*j+1] for every j.

    Returns same tuple shape as sample_one_full_trotter_layer_masked:
      P_even, P_out, step_sign, Z, sample_prob, coeff_even, coeff_odd,
      B_even, odd_branch_choices
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2
    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    Rgood = np.empty((L, 4, 4), dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)
    build_all_good_right_messages(outs, absw, k_counts, trans_codes, abs_coeffs,
                                  n_branches, Rgood)

    # Partition function. For the first site, final left qubit is the untouched
    # left qubit of its even-output block.
    Z = 0.0
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        Z += Rgood[0, r, a0]

    branch_choices = np.empty(L, dtype=np.int64)
    odd_branch_choices = np.zeros(max(L - 1, 1), dtype=np.int64)
    weights = np.zeros(16, dtype=np.float64)
    sample_prob = 1.0

    # If no all-good support, return deterministic no-op-ish sentinel with Z=0.
    if Z <= 0.0:
        B_even0 = np.empty(L, dtype=np.int8)
        pauli_to_blocked_inplace(P_in, B_even0)
        P_even0 = np.empty(n_qubits, dtype=np.int8)
        P_out0 = np.empty(n_qubits, dtype=np.int8)
        for q in range(n_qubits):
            P_even0[q] = P_in[q]
            P_out0[q] = P_in[q]
        return P_even0, P_out0, np.int8(1), 0.0, 1.0, 1.0, 1.0, B_even0, odd_branch_choices

    # Sample first even branch.
    for i in range(16):
        weights[i] = 0.0
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        weights[r] = Rgood[0, r, a0]
    r0, p0 = sample_from_weights(weights, k_counts[0])
    branch_choices[0] = r0
    sample_prob *= p0
    left_val = _code_left(int(outs[0, r0]))

    coeff_even = coeffs[0, r0]
    coeff_odd = 1.0

    # Sequentially sample odd edge j and next even branch r_{j+1}.
    for j in range(L - 1):
        code_cur = int(outs[j, branch_choices[j]])
        b = _code_right(code_cur)
        idx = 0
        for i in range(16):
            weights[i] = 0.0

        # Pack candidates as idx = r_next*4 + ob. Since both <=4, idx < 16.
        for r_next in range(k_counts[j + 1]):
            code_next = int(outs[j + 1, r_next])
            c = _code_left(code_next)
            mid = 4 * b + c
            for ob in range(n_branches[mid]):
                out_mid = int(trans_codes[mid, ob])
                u = _code_left(out_mid)
                v = _code_right(out_mid)
                if u == left_val:
                    weights[r_next * 4 + ob] = abs_coeffs[mid, ob] * Rgood[j + 1, r_next, v]

        choice, pchoice = sample_from_weights(weights, 16)
        sample_prob *= pchoice
        r_next = choice // 4
        ob = choice - 4 * r_next
        branch_choices[j + 1] = r_next
        odd_branch_choices[j] = ob

        code_next = int(outs[j + 1, r_next])
        mid = 4 * b + _code_left(code_next)
        out_mid = int(trans_codes[mid, ob])
        left_val = _code_right(out_mid)

        coeff_odd *= signed_coeffs[mid, ob]
        coeff_even *= coeffs[j + 1, r_next]

    # Build P_even.
    B_even = np.empty(L, dtype=np.int8)
    for j in range(L):
        B_even[j] = outs[j, branch_choices[j]]
    P_even = np.empty(n_qubits, dtype=np.int8)
    blocked_to_pauli_inplace(B_even, P_even)

    # Build P_out by applying sampled odd branches.
    P_out = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P_out[q] = P_even[q]
    for j in range(L - 1):
        q = 2 * j + 1
        code = 4 * int(P_out[q]) + int(P_out[q + 1])
        ob = odd_branch_choices[j]
        out_code = int(trans_codes[code, ob])
        P_out[q] = np.int8(out_code // 4)
        P_out[q + 1] = np.int8(out_code % 4)

    coeff_full = coeff_even * coeff_odd
    step_sign = np.int8(1)
    if coeff_full < 0.0:
        step_sign = np.int8(-1)

    return P_even, P_out, step_sign, Z, sample_prob, coeff_even, coeff_odd, B_even, odd_branch_choices


@njit
def _even_block_noise_factor(block_idx, code, eta_xyz):
    left = int(code) // 4
    right = int(code) % 4
    q0 = 2 * block_idx
    q1 = q0 + 1
    fac = 1.0
    if left > 0:
        fac *= eta_xyz[q0, left - 1]
    if right > 0:
        fac *= eta_xyz[q1, right - 1]
    return fac


@njit
def _final_good_block_factor(block_idx, val, eta_xyz):
    """
    Observable and final-noise factor for a final good block val-val.
    The singlet-pair observable contributes +1 for II and -1 for XX/YY/ZZ.
    """
    if val == 0:
        return 1.0
    q0 = 2 * block_idx
    q1 = q0 + 1
    return -eta_xyz[q0, val - 1] * eta_xyz[q1, val - 1]


@njit
def terminal_all_good_contribution(
    P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz
):
    """
    Exact signed/noisy contribution of the final even+odd layer from P_in.

    This sums all final-layer descendants with nonzero singlet-pair boundary
    value, i.e. final P_out[2*j] == P_out[2*j+1] for every blocked site j.
    It includes signed unitary coefficients, full-chain diagonal noise after
    the even layer, final full-chain diagonal noise after the odd layer, and
    the singlet-pair observable sign.
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2

    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    R = np.zeros((L, 4, 4), dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)

    last = L - 1
    for r in range(k_counts[last]):
        code = int(outs[last, r])
        right = _code_right(code)
        even_factor = coeffs[last, r] * _even_block_noise_factor(last, code, eta_xyz)
        for left_val in range(4):
            if left_val == right:
                R[last, r, left_val] = (
                    even_factor * _final_good_block_factor(last, left_val, eta_xyz)
                )

    for j in range(L - 2, -1, -1):
        for r_cur in range(k_counts[j]):
            code_cur = int(outs[j, r_cur])
            b = _code_right(code_cur)
            even_factor = coeffs[j, r_cur] * _even_block_noise_factor(j, code_cur, eta_xyz)
            for left_val in range(4):
                total = 0.0
                for r_next in range(k_counts[j + 1]):
                    code_next = int(outs[j + 1, r_next])
                    c = _code_left(code_next)
                    mid = 4 * b + c
                    for ob in range(n_branches[mid]):
                        out_mid = int(trans_codes[mid, ob])
                        u = _code_left(out_mid)
                        v = _code_right(out_mid)
                        if u == left_val:
                            total += signed_coeffs[mid, ob] * R[j + 1, r_next, v]
                R[j, r_cur, left_val] = (
                    even_factor * _final_good_block_factor(j, left_val, eta_xyz) * total
                )

    total = 0.0
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        total += R[0, r, a0]
    return total


@njit
def terminal_all_good_contribution_grid(
    P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz, s_grid
):
    """
    Exact signed contribution of the final even+odd layer for each s in s_grid.

    This is the grid-valued version of terminal_all_good_contribution. The
    unitary coefficients and singlet-pair observable signs are unchanged, while
    the even-layer and final-layer diagonal noise factors are raised to s.
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2
    ns = s_grid.shape[0]

    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    next_R = np.zeros((ns, 4, 4), dtype=np.float64)
    cur_R = np.zeros((ns, 4, 4), dtype=np.float64)
    local_pow = np.empty(ns, dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)

    last = L - 1
    for r in range(k_counts[last]):
        code = int(outs[last, r])
        right = _code_right(code)
        even_noise = _even_block_noise_factor(last, code, eta_xyz)
        c_even = coeffs[last, r]
        for left_val in range(4):
            if left_val == right:
                final_noise = 1.0
                if left_val > 0:
                    q0 = 2 * last
                    q1 = q0 + 1
                    final_noise = eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]
                obs = _good_block_obs_sign(left_val)
                local_noise = even_noise * final_noise
                for k in range(ns):
                    next_R[k, r, left_val] = c_even * obs * local_noise ** s_grid[k]

    for j in range(L - 2, -1, -1):
        for r in range(4):
            for left_val in range(4):
                for k in range(ns):
                    cur_R[k, r, left_val] = 0.0

        for r_cur in range(k_counts[j]):
            code_cur = int(outs[j, r_cur])
            b = _code_right(code_cur)
            even_noise = _even_block_noise_factor(j, code_cur, eta_xyz)
            c_even = coeffs[j, r_cur]
            for left_val in range(4):
                final_noise = 1.0
                if left_val > 0:
                    q0 = 2 * j
                    q1 = q0 + 1
                    final_noise = eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]
                obs = _good_block_obs_sign(left_val)
                local_noise = even_noise * final_noise
                for k in range(ns):
                    local_pow[k] = local_noise ** s_grid[k]
                for k in range(ns):
                    total = 0.0
                    for r_next in range(k_counts[j + 1]):
                        code_next = int(outs[j + 1, r_next])
                        c = _code_left(code_next)
                        mid = 4 * b + c
                        for ob in range(n_branches[mid]):
                            out_mid = int(trans_codes[mid, ob])
                            u = _code_left(out_mid)
                            v = _code_right(out_mid)
                            if u == left_val:
                                total += signed_coeffs[mid, ob] * next_R[k, r_next, v]
                    cur_R[k, r_cur, left_val] = (
                        c_even * obs * local_pow[k] * total
                    )

        tmp_R = next_R
        next_R = cur_R
        cur_R = tmp_R

    total_grid = np.zeros(ns, dtype=np.float64)
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        for k in range(ns):
            total_grid[k] += next_R[k, r, a0]
    return total_grid


@njit
def _coeff_sign(x):
    if x < 0.0:
        return -1
    return 1


@njit
def _good_block_obs_sign(val):
    if val == 0:
        return 1
    return -1


@njit
def terminal_all_good_qpm_sums(
    P_in, prefix_sign, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz
):
    """
    Exact final-layer terminal sums for Q^+(s=1), Q^-(s=1).

    Returns plus_num, plus_den, minus_num, minus_den for the final layer only.
    The denominator omits terminal noise; the numerator includes terminal noise.
    Both use absolute path coefficient mass, split by total sector
    prefix_sign * final_unitary_sign * obs(P_out).
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2

    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    num_p = np.zeros((L, 4, 4), dtype=np.float64)
    num_m = np.zeros((L, 4, 4), dtype=np.float64)
    den_p = np.zeros((L, 4, 4), dtype=np.float64)
    den_m = np.zeros((L, 4, 4), dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)

    last = L - 1
    for r in range(k_counts[last]):
        code = int(outs[last, r])
        right = _code_right(code)
        s_even = _coeff_sign(coeffs[last, r])
        even_noise = _even_block_noise_factor(last, code, eta_xyz)
        for left_val in range(4):
            if left_val == right:
                sector = s_even * _good_block_obs_sign(left_val)
                dval = absw[last, r]
                nval = dval * even_noise
                if left_val > 0:
                    q0 = 2 * last
                    q1 = q0 + 1
                    nval *= eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]
                if sector > 0:
                    den_p[last, r, left_val] = dval
                    num_p[last, r, left_val] = nval
                else:
                    den_m[last, r, left_val] = dval
                    num_m[last, r, left_val] = nval

    for j in range(L - 2, -1, -1):
        for r_cur in range(k_counts[j]):
            code_cur = int(outs[j, r_cur])
            b = _code_right(code_cur)
            s_even = _coeff_sign(coeffs[j, r_cur])
            even_noise = _even_block_noise_factor(j, code_cur, eta_xyz)
            for left_val in range(4):
                local_sector_no_odd = s_even * _good_block_obs_sign(left_val)
                den_local_base = absw[j, r_cur]
                num_local_base = den_local_base * even_noise
                if left_val > 0:
                    q0 = 2 * j
                    q1 = q0 + 1
                    num_local_base *= eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]

                np_acc = 0.0
                nm_acc = 0.0
                dp_acc = 0.0
                dm_acc = 0.0

                for r_next in range(k_counts[j + 1]):
                    code_next = int(outs[j + 1, r_next])
                    c = _code_left(code_next)
                    mid = 4 * b + c
                    for ob in range(n_branches[mid]):
                        out_mid = int(trans_codes[mid, ob])
                        u = _code_left(out_mid)
                        v = _code_right(out_mid)
                        if u == left_val:
                            odd_sector = _coeff_sign(signed_coeffs[mid, ob])
                            sector = local_sector_no_odd * odd_sector
                            dloc = den_local_base * abs_coeffs[mid, ob]
                            nloc = num_local_base * abs_coeffs[mid, ob]
                            if sector > 0:
                                np_acc += nloc * num_p[j + 1, r_next, v]
                                nm_acc += nloc * num_m[j + 1, r_next, v]
                                dp_acc += dloc * den_p[j + 1, r_next, v]
                                dm_acc += dloc * den_m[j + 1, r_next, v]
                            else:
                                np_acc += nloc * num_m[j + 1, r_next, v]
                                nm_acc += nloc * num_p[j + 1, r_next, v]
                                dp_acc += dloc * den_m[j + 1, r_next, v]
                                dm_acc += dloc * den_p[j + 1, r_next, v]

                num_p[j, r_cur, left_val] = np_acc
                num_m[j, r_cur, left_val] = nm_acc
                den_p[j, r_cur, left_val] = dp_acc
                den_m[j, r_cur, left_val] = dm_acc

    plus_num = 0.0
    plus_den = 0.0
    minus_num = 0.0
    minus_den = 0.0
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        plus_num += num_p[0, r, a0]
        plus_den += den_p[0, r, a0]
        minus_num += num_m[0, r, a0]
        minus_den += den_m[0, r, a0]

    if prefix_sign < 0:
        return minus_num, minus_den, plus_num, plus_den
    return plus_num, plus_den, minus_num, minus_den


@njit
def terminal_all_good_qpm_grid_sums(
    P_in, prefix_sign, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz,
    s_grid,
):
    """
    Exact final-layer terminal sums for Q^+(s), Q^-(s) over an s grid.

    Denominators omit terminal noise and are independent of s. Numerators use
    absolute path coefficient mass times terminal damping**s, split by total
    sector prefix_sign * final_unitary_sign * obs(P_out).
    """
    n_qubits = P_in.shape[0]
    L = n_qubits // 2
    ns = s_grid.shape[0]

    outs = np.empty((L, 4), dtype=np.int8)
    coeffs = np.empty((L, 4), dtype=np.float64)
    absw = np.empty((L, 4), dtype=np.float64)
    k_counts = np.empty(L, dtype=np.int64)
    next_num_p = np.zeros((ns, 4, 4), dtype=np.float64)
    next_num_m = np.zeros((ns, 4, 4), dtype=np.float64)
    cur_num_p = np.zeros((ns, 4, 4), dtype=np.float64)
    cur_num_m = np.zeros((ns, 4, 4), dtype=np.float64)
    next_den_p = np.zeros((4, 4), dtype=np.float64)
    next_den_m = np.zeros((4, 4), dtype=np.float64)
    cur_den_p = np.zeros((4, 4), dtype=np.float64)
    cur_den_m = np.zeros((4, 4), dtype=np.float64)
    local_pow = np.empty(ns, dtype=np.float64)

    build_even_branch_tables_inplace(P_in, trans_codes, signed_coeffs, abs_coeffs,
                                     n_branches, outs, coeffs, absw, k_counts)

    last = L - 1
    for r in range(k_counts[last]):
        code = int(outs[last, r])
        right = _code_right(code)
        s_even = _coeff_sign(coeffs[last, r])
        local_damp = _even_block_noise_factor(last, code, eta_xyz)
        for left_val in range(4):
            if left_val == right:
                sector = s_even * _good_block_obs_sign(left_val)
                dval = absw[last, r]
                term_damp = local_damp
                if left_val > 0:
                    q0 = 2 * last
                    q1 = q0 + 1
                    term_damp *= eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]
                if sector > 0:
                    next_den_p[r, left_val] = dval
                    for k in range(ns):
                        next_num_p[k, r, left_val] = dval * term_damp ** s_grid[k]
                else:
                    next_den_m[r, left_val] = dval
                    for k in range(ns):
                        next_num_m[k, r, left_val] = dval * term_damp ** s_grid[k]

    for j in range(L - 2, -1, -1):
        for r in range(4):
            for left_val in range(4):
                cur_den_p[r, left_val] = 0.0
                cur_den_m[r, left_val] = 0.0
                for k in range(ns):
                    cur_num_p[k, r, left_val] = 0.0
                    cur_num_m[k, r, left_val] = 0.0

        for r_cur in range(k_counts[j]):
            code_cur = int(outs[j, r_cur])
            b = _code_right(code_cur)
            s_even = _coeff_sign(coeffs[j, r_cur])
            even_noise = _even_block_noise_factor(j, code_cur, eta_xyz)
            for left_val in range(4):
                local_sector_no_odd = s_even * _good_block_obs_sign(left_val)
                den_local_base = absw[j, r_cur]
                local_damp = even_noise
                if left_val > 0:
                    q0 = 2 * j
                    q1 = q0 + 1
                    local_damp *= eta_xyz[q0, left_val - 1] * eta_xyz[q1, left_val - 1]
                for k in range(ns):
                    local_pow[k] = local_damp ** s_grid[k]

                for r_next in range(k_counts[j + 1]):
                    code_next = int(outs[j + 1, r_next])
                    c = _code_left(code_next)
                    mid = 4 * b + c
                    for ob in range(n_branches[mid]):
                        out_mid = int(trans_codes[mid, ob])
                        u = _code_left(out_mid)
                        v = _code_right(out_mid)
                        if u == left_val:
                            odd_sector = _coeff_sign(signed_coeffs[mid, ob])
                            sector = local_sector_no_odd * odd_sector
                            dloc = den_local_base * abs_coeffs[mid, ob]
                            local_abs_damp = abs_coeffs[mid, ob] * den_local_base
                            if sector > 0:
                                cur_den_p[r_cur, left_val] += dloc * next_den_p[r_next, v]
                                cur_den_m[r_cur, left_val] += dloc * next_den_m[r_next, v]
                                for k in range(ns):
                                    nloc = local_abs_damp * local_pow[k]
                                    cur_num_p[k, r_cur, left_val] += nloc * next_num_p[k, r_next, v]
                                    cur_num_m[k, r_cur, left_val] += nloc * next_num_m[k, r_next, v]
                            else:
                                cur_den_p[r_cur, left_val] += dloc * next_den_m[r_next, v]
                                cur_den_m[r_cur, left_val] += dloc * next_den_p[r_next, v]
                                for k in range(ns):
                                    nloc = local_abs_damp * local_pow[k]
                                    cur_num_p[k, r_cur, left_val] += nloc * next_num_m[k, r_next, v]
                                    cur_num_m[k, r_cur, left_val] += nloc * next_num_p[k, r_next, v]

        tmp_den = next_den_p
        next_den_p = cur_den_p
        cur_den_p = tmp_den
        tmp_den = next_den_m
        next_den_m = cur_den_m
        cur_den_m = tmp_den
        tmp_num = next_num_p
        next_num_p = cur_num_p
        cur_num_p = tmp_num
        tmp_num = next_num_m
        next_num_m = cur_num_m
        cur_num_m = tmp_num

    plus_num = np.zeros(ns, dtype=np.float64)
    minus_num = np.zeros(ns, dtype=np.float64)
    plus_den = 0.0
    minus_den = 0.0
    for r in range(k_counts[0]):
        a0 = _code_left(int(outs[0, r]))
        plus_den += next_den_p[r, a0]
        minus_den += next_den_m[r, a0]
        for k in range(ns):
            plus_num[k] += next_num_p[k, r, a0]
            minus_num[k] += next_num_m[k, r, a0]

    if prefix_sign < 0:
        return minus_num, minus_den, plus_num, plus_den
    return plus_num, plus_den, minus_num, minus_den


@njit
def sample_one_full_trotter_layer_mixture(
    P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches,
    alpha_free, use_restricted, force_restricted, final_step,
):
    """
    Mixture proposal: q = alpha_free*q_free + (1-alpha_free)*q_restricted.
    Returns step_l1_free and importance ratio p_free/q.
    """
    if (not use_restricted) or ((not force_restricted) and alpha_free >= 1.0):
        P_even, P_out, step_sign, Z_free, sample_prob, coeff_even, coeff_odd = sample_one_full_trotter_layer_lightcone(
            P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches
        )
        return P_even, P_out, step_sign, Z_free, 1.0, coeff_even, coeff_odd

    n_qubits = P_in.shape[0]
    L = n_qubits // 2

    # masks
    active_site = np.ones(L, dtype=np.bool_)
    active_edge = np.ones(max(L - 1, 1), dtype=np.bool_)
    if use_restricted:
        fill_active_masks_from_zones(P_in, active_site, active_edge)

    # Free Z
    all_site = np.ones(L, dtype=np.bool_)
    all_edge = np.ones(max(L - 1, 1), dtype=np.bool_)
    Pef, Pof, sf, Z_free, pf, cef, cof, Bef, obf = sample_one_full_trotter_layer_masked(
        P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches, all_site, all_edge
    )

    # Restricted Z/sample if needed.
    # On the final remaining layer, restriction means condition on all-good output.
    # Otherwise use the active-zone/domain-progress restricted sampler.
    if final_step:
        Per, Por, sr, Z_res, pr, cer, cor, Ber, obr = sample_one_full_trotter_layer_all_good(
            P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches
        )
    else:
        Per, Por, sr, Z_res, pr, cer, cor, Ber, obr = sample_one_full_trotter_layer_masked(
            P_in, trans_codes, signed_coeffs, abs_coeffs, n_branches, active_site, active_edge
        )

    if Z_res <= 0.0:
        # No restricted support.  If we are not forced restricted, fall back to
        # the free component of the mixture.  If trigger/force_restricted is on,
        # do NOT silently free-sample; return the restricted attempt with zero
        # importance ratio so the caller can see this path contributed zero.
        if force_restricted:
            return Per, Por, sr, Z_free, 0.0, cer, cor
        return Pef, Pof, sf, Z_free, 1.0, cef, cof

    a = alpha_free
    if force_restricted:
        a = 0.0

    use_res_mode = False
    if np.random.random() >= a:
        use_res_mode = True

    if use_res_mode:
        # sample is definitely allowed under restricted support.
        ratio = 1.0 / (a + (1.0 - a) * (Z_free / Z_res))
        return Per, Por, sr, Z_free, ratio, cer, cor
    else:
        # We already have a free sample. Check if it lies in restricted support.
        if final_step:
            allowed = is_all_good_pauli(Pof)
        else:
            allowed = _branch_allowed_under_restriction(np.empty(1, dtype=np.int64), Bef,
                                                        _pauli_to_blocked_numba(P_in), obf,
                                                        active_site, active_edge)
        if allowed:
            ratio = 1.0 / (a + (1.0 - a) * (Z_free / Z_res))
        else:
            ratio = 1.0 / a
        return Pef, Pof, sf, Z_free, ratio, cef, cof


@njit
def evolve_one_full_layer_sampling_mixture_restricted(
    init_pauli, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz,
    alpha_t, use_restricted, use_trigger,
):
    n_steps = alpha_t.shape[0]
    n_qubits = init_pauli.shape[0]
    P = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P[q] = init_pauli[q]

    sign = np.int8(1)
    amp = 1.0
    damp = 1.0
    P_even_last = np.empty(n_qubits, dtype=np.int8)
    triggered = False

    for t in range(n_steps):
        Rsteps = n_steps - t
        if use_trigger and use_restricted:
            d = max_domain_size_triple_parity(P)
            if d >= Rsteps:
                triggered = True
        force_restricted = triggered

        final_step = (Rsteps == 1)
        P_even, P_out, step_sign, Z_free, wi_ratio, ce, co = sample_one_full_trotter_layer_mixture(
            P, trans_codes, signed_coeffs, abs_coeffs, n_branches,
            alpha_t[t], use_restricted, force_restricted, final_step
        )

        damp *= pauli_noise_factor(P_even, eta_xyz)
        damp *= pauli_noise_factor(P_out, eta_xyz)
        sign = np.int8(sign * step_sign)
        amp *= Z_free * wi_ratio
        for q in range(n_qubits):
            P_even_last[q] = P_even[q]
            P[q] = P_out[q]

    return P, P_even_last, sign, amp, damp, triggered


@njit(parallel=True)
def evolve_many_full_layer_sampling_mixture_restricted(
    init_pauli, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz,
    alpha_t, n_samples, use_restricted=True, use_trigger=True,
):
    n_qubits = init_pauli.shape[0]
    P_out = np.empty((n_samples, n_qubits), dtype=np.int8)
    P_even_last = np.empty((n_samples, n_qubits), dtype=np.int8)
    s_out = np.empty(n_samples, dtype=np.int8)
    amp_out = np.empty(n_samples, dtype=np.float64)
    damp_out = np.empty(n_samples, dtype=np.float64)
    triggered_out = np.empty(n_samples, dtype=np.bool_)

    for i in prange(n_samples):
        P, Pe, s, a, d, tr = evolve_one_full_layer_sampling_mixture_restricted(
            init_pauli, trans_codes, signed_coeffs, abs_coeffs, n_branches,
            eta_xyz, alpha_t, use_restricted, use_trigger
        )
        for q in range(n_qubits):
            P_out[i, q] = P[q]
            P_even_last[i, q] = Pe[q]
        s_out[i] = s
        amp_out[i] = a
        damp_out[i] = d
        triggered_out[i] = tr
    return P_out, P_even_last, s_out, amp_out, damp_out, triggered_out
