import argparse
import time

import numpy as np
from numba import njit, prange

import Pauli_path_Heis_mixture_trigger as gate
import Pauli_path_Heis_full_layer_sampling_restricted as layer


def product_eta_from_lambda(lam_xyz):
    eta = np.empty_like(lam_xyz, dtype=np.float64)
    ex = np.exp(-2.0 * lam_xyz[:, 0])
    ey = np.exp(-2.0 * lam_xyz[:, 1])
    ez = np.exp(-2.0 * lam_xyz[:, 2])
    eta[:, 0] = ey * ez
    eta[:, 1] = ex * ez
    eta[:, 2] = ex * ey
    return eta


@njit
def gate_prefix_qpm_one(
    init_pauli,
    even_gates,
    odd_gates,
    trans_codes,
    probs,
    is_commuting,
    branch_sign,
    amp_factr,
    terminal_trans,
    terminal_signed,
    terminal_abs,
    terminal_n_branches,
    eta_xyz,
    n_steps,
):
    pauli = init_pauli.copy()
    sign = np.int8(1)
    amp = 1.0
    damp = 1.0
    log_wi = 0.0
    triggered = False

    for t in range(n_steps - 1):
        d = gate.max_domain_size(pauli)
        R = n_steps - t
        if d == R:
            triggered = True

        if triggered:
            sign, amp, log_wi = gate.mixed_even_layer_numba(
                pauli,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                0.0,
                sign,
                amp,
                log_wi,
            )
        else:
            sign, amp = gate.free_even_layer_numba(
                pauli,
                even_gates,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                sign,
                amp,
            )

        damp = gate.apply_noise(pauli, eta_xyz, damp)

        if triggered:
            sign, amp, log_wi = gate.mixed_odd_layer_numba(
                pauli,
                odd_gates,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                0.0,
                sign,
                amp,
                log_wi,
            )
        else:
            sign, amp = gate.free_odd_layer_numba(
                pauli,
                odd_gates,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                sign,
                amp,
            )

        damp = gate.apply_noise(pauli, eta_xyz, damp)

    tp_num, tp_den, tm_num, tm_den = layer.terminal_all_good_qpm_sums(
        pauli,
        sign,
        terminal_trans,
        terminal_signed,
        terminal_abs,
        terminal_n_branches,
        eta_xyz,
    )
    pref = amp * np.exp(log_wi)
    plus_num = pref * damp * tp_num
    plus_den = pref * tp_den
    minus_num = pref * damp * tm_num
    minus_den = pref * tm_den
    target = plus_num - minus_num
    return target, plus_num, plus_den, minus_num, minus_den, triggered


@njit(parallel=True)
def gate_prefix_qpm_aggregate(
    init_pauli,
    even_gates,
    odd_gates,
    trans_codes,
    probs,
    is_commuting,
    branch_sign,
    amp_factr,
    terminal_trans,
    terminal_signed,
    terminal_abs,
    terminal_n_branches,
    eta_xyz,
    n_steps,
    n_samples,
):
    sx = 0.0
    sx2 = 0.0
    spn = 0.0
    spd = 0.0
    smn = 0.0
    smd = 0.0
    ntr = 0

    for i in prange(n_samples):
        x, pn, pd, mn, md, tr = gate_prefix_qpm_one(
            init_pauli,
            even_gates,
            odd_gates,
            trans_codes,
            probs,
            is_commuting,
            branch_sign,
            amp_factr,
            terminal_trans,
            terminal_signed,
            terminal_abs,
            terminal_n_branches,
            eta_xyz,
            n_steps,
        )
        sx += x
        sx2 += x * x
        spn += pn
        spd += pd
        smn += mn
        smd += md
        if tr:
            ntr += 1

    return sx, sx2, spn, spd, smn, smd, ntr


@njit
def dp_prefix_qpm_one(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
):
    n_qubits = init_pauli.shape[0]
    P = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        P[q] = init_pauli[q]

    sign = np.int8(1)
    amp = 1.0
    damp = 1.0
    triggered = False

    for t in range(n_steps - 1):
        Rsteps = n_steps - t
        d = layer.max_domain_size_triple_parity(P)
        if d >= Rsteps:
            triggered = True

        if triggered:
            P_even, P_out, step_sign, Z_free, wi_ratio, _, _ = layer.sample_one_full_trotter_layer_mixture(
                P,
                trans_codes,
                signed_coeffs,
                abs_coeffs,
                n_branches,
                1.0,
                True,
                True,
                False,
            )
            amp *= Z_free * wi_ratio
        else:
            P_even, P_out, step_sign, step_l1, _, _, _ = layer.sample_one_full_trotter_layer_lightcone(
                P,
                trans_codes,
                signed_coeffs,
                abs_coeffs,
                n_branches,
            )
            amp *= step_l1

        damp *= layer.pauli_noise_factor(P_even, eta_xyz)
        damp *= layer.pauli_noise_factor(P_out, eta_xyz)
        sign = np.int8(sign * step_sign)
        for q in range(n_qubits):
            P[q] = P_out[q]

    tp_num, tp_den, tm_num, tm_den = layer.terminal_all_good_qpm_sums(
        P, sign, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz
    )
    plus_num = amp * damp * tp_num
    plus_den = amp * tp_den
    minus_num = amp * damp * tm_num
    minus_den = amp * tm_den
    target = plus_num - minus_num
    return target, plus_num, plus_den, minus_num, minus_den, triggered


@njit(parallel=True)
def dp_prefix_qpm_aggregate(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
    n_samples,
):
    sx = 0.0
    sx2 = 0.0
    spn = 0.0
    spd = 0.0
    smn = 0.0
    smd = 0.0
    ntr = 0

    for i in prange(n_samples):
        x, pn, pd, mn, md, tr = dp_prefix_qpm_one(
            init_pauli,
            trans_codes,
            signed_coeffs,
            abs_coeffs,
            n_branches,
            eta_xyz,
            n_steps,
        )
        sx += x
        sx2 += x * x
        spn += pn
        spd += pd
        smn += mn
        smd += md
        if tr:
            ntr += 1

    return sx, sx2, spn, spd, smn, smd, ntr


def row_from_aggregate(name, agg, n_samples, runtime):
    sx, sx2, spn, spd, smn, smd, ntr = agg
    mean = sx / n_samples
    var = (sx2 - sx * sx / n_samples) / (n_samples - 1)
    se = (var / n_samples) ** 0.5
    q_plus = spn / spd
    q_minus = smn / smd
    return {
        "name": name,
        "mean": float(mean),
        "var": float(var),
        "se": float(se),
        "q_plus": float(q_plus),
        "q_minus": float(q_minus),
        "runtime": float(runtime),
        "triggered_frac": float(ntr / n_samples),
    }


def print_summary(name, rows):
    means = np.array([r["mean"] for r in rows])
    ses = np.array([r["se"] for r in rows])
    q_plus = np.array([r["q_plus"] for r in rows])
    q_minus = np.array([r["q_minus"] for r in rows])
    runtimes = np.array([r["runtime"] for r in rows])

    print(f"\n{name}")
    print("-" * len(name))
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:2d}  mean {r['mean']:.8g}  SE {r['se']:.6g}  "
            f"Q+ {r['q_plus']:.12f}  Q- {r['q_minus']:.12f}  "
            f"time {r['runtime']:.3f}s"
        )
    print("summary")
    print(f"mean of means       {np.mean(means):.12g}")
    print(f"spread of means     {np.std(means, ddof=1):.6g}")
    print(f"mean reported SE    {np.mean(ses):.6g}")
    print(f"Q+ mean             {np.mean(q_plus):.12f}")
    print(f"Q+ spread std       {np.std(q_plus, ddof=1):.6g}")
    print(f"Q- mean             {np.mean(q_minus):.12f}")
    print(f"Q- spread std       {np.std(q_minus, ddof=1):.6g}")
    print(f"runtime mean        {np.mean(runtimes):.3f}s")
    return {
        "mean_spread": float(np.std(means, ddof=1)),
        "mean_se": float(np.mean(ses)),
        "q_plus_spread": float(np.std(q_plus, ddof=1)),
        "q_minus_spread": float(np.std(q_minus, ddof=1)),
        "runtime_mean": float(np.mean(runtimes)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=10_000_000)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.15)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    args = parser.parse_args()

    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[0] = 3
    init_pauli[1] = 3

    lam_xyz = np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)

    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    print("Repeated final-layer-DP Qpm benchmark", flush=True)
    print("-------------------------------------", flush=True)
    print(f"n_qubits       {args.n_qubits}", flush=True)
    print(f"n_steps        {args.n_steps}", flush=True)
    print(f"n_samples      {args.n_samples}", flush=True)
    print(f"n_repeats      {args.n_repeats}", flush=True)
    print("target         Z1Z2 as code qubits (0, 1)", flush=True)
    print("final step     exact all-good Qpm DP marginal", flush=True)
    print(f"phi            {args.phi}", flush=True)
    print(f"lambda_base    {args.lambda_base}", flush=True)
    print("noise_model    exact product eta", flush=True)

    print("\nWarming Numba kernels...", flush=True)
    gate_prefix_qpm_aggregate(
        init_pauli,
        even_gates,
        odd_gates,
        trans_g,
        probs_g,
        is_comm_g,
        sign_g,
        amp_factor_g,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        args.n_steps,
        8,
    )
    dp_prefix_qpm_aggregate(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        args.n_steps,
        8,
    )
    print("Warmup complete.", flush=True)

    gate_rows = []
    dp_rows = []
    for rep in range(args.n_repeats):
        t0 = time.perf_counter()
        agg_g = gate_prefix_qpm_aggregate(
            init_pauli,
            even_gates,
            odd_gates,
            trans_g,
            probs_g,
            is_comm_g,
            sign_g,
            amp_factor_g,
            trans_l,
            signed_l,
            abs_l,
            n_branches_l,
            eta_xyz,
            args.n_steps,
            args.n_samples,
        )
        gate_rows.append(row_from_aggregate("gate", agg_g, args.n_samples, time.perf_counter() - t0))
        print(f"completed gate repeat {rep + 1}/{args.n_repeats}", flush=True)

        t0 = time.perf_counter()
        agg_l = dp_prefix_qpm_aggregate(
            init_pauli,
            trans_l,
            signed_l,
            abs_l,
            n_branches_l,
            eta_xyz,
            args.n_steps,
            args.n_samples,
        )
        dp_rows.append(row_from_aggregate("dp", agg_l, args.n_samples, time.perf_counter() - t0))
        print(f"completed DP repeat {rep + 1}/{args.n_repeats}", flush=True)

    gs = print_summary("gate-wise + final-layer Qpm DP", gate_rows)
    ds = print_summary("full-layer DP + final-layer Qpm DP", dp_rows)

    print("\nRatios DP/gate, lower is better for spreads/time", flush=True)
    print("-----------------------------------------------", flush=True)
    print(f"mean spread      {ds['mean_spread'] / gs['mean_spread']:.6g}", flush=True)
    print(f"reported SE      {ds['mean_se'] / gs['mean_se']:.6g}", flush=True)
    print(f"Q+ spread        {ds['q_plus_spread'] / gs['q_plus_spread']:.6g}", flush=True)
    print(f"Q- spread        {ds['q_minus_spread'] / gs['q_minus_spread']:.6g}", flush=True)
    print(f"runtime          {ds['runtime_mean'] / gs['runtime_mean']:.6g}", flush=True)


if __name__ == "__main__":
    main()
