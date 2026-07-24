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
def evolve_one_gate_prefix_terminal(
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
            alpha_t = 0.0
            sign, amp, log_wi = gate.mixed_even_layer_numba(
                pauli,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                alpha_t,
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
            alpha_t = 0.0
            sign, amp, log_wi = gate.mixed_odd_layer_numba(
                pauli,
                odd_gates,
                trans_codes,
                probs,
                is_commuting,
                branch_sign,
                amp_factr,
                alpha_t,
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

    terminal = layer.terminal_all_good_contribution(
        pauli,
        terminal_trans,
        terminal_signed,
        terminal_abs,
        terminal_n_branches,
        eta_xyz,
    )
    return sign * amp * np.exp(log_wi) * damp * terminal, triggered


@njit(parallel=True)
def evolve_many_gate_prefix_terminal(
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
    contribs = np.empty(n_samples, dtype=np.float64)
    triggered = np.empty(n_samples, dtype=np.bool_)
    for i in prange(n_samples):
        c, tr = evolve_one_gate_prefix_terminal(
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
        contribs[i] = c
        triggered[i] = tr
    return contribs, triggered


@njit
def evolve_one_dp_prefix_terminal(
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

    terminal = layer.terminal_all_good_contribution(
        P, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz
    )
    return sign * amp * damp * terminal, triggered


@njit(parallel=True)
def evolve_many_dp_prefix_terminal(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
    n_samples,
):
    contribs = np.empty(n_samples, dtype=np.float64)
    triggered = np.empty(n_samples, dtype=np.bool_)
    for i in prange(n_samples):
        c, tr = evolve_one_dp_prefix_terminal(
            init_pauli,
            trans_codes,
            signed_coeffs,
            abs_coeffs,
            n_branches,
            eta_xyz,
            n_steps,
        )
        contribs[i] = c
        triggered[i] = tr
    return contribs, triggered


def summarize(name, contribs, triggered, runtime):
    n = contribs.shape[0]
    mean = float(np.mean(contribs))
    var = float(np.var(contribs, ddof=1))
    std = var**0.5
    se = std / n**0.5
    print(f"\n{name}")
    print("-" * len(name))
    print(f"runtime_s       {runtime:.6g}")
    print(f"target mean     {mean:.12g}")
    print(f"target var      {var:.12g}")
    print(f"target std      {std:.12g}")
    print(f"target SE       {se:.6g}")
    print(f"target var*time {var * runtime:.6g}")
    print(f"triggered_frac  {np.mean(triggered):.6g}")
    return {"mean": mean, "var": var, "se": se, "runtime": runtime, "var_time": var * runtime}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument("--phi", type=float, default=0.15)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    args = parser.parse_args()

    n_qubits = args.n_qubits
    n_steps = args.n_steps
    phi = args.phi
    n_samples = args.n_samples
    lambda_base = args.lambda_base

    init_pauli = np.zeros(n_qubits, dtype=np.int8)
    init_pauli[0] = 3
    init_pauli[1] = 3

    lam_xyz = np.full((n_qubits, 3), lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)

    even_gates, odd_gates = gate.make_even_odd_layers(n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(phi)

    print("Final-layer marginalized target benchmark")
    print("-----------------------------------------")
    print(f"n_qubits       {n_qubits}")
    print("target         Z1Z2 as code qubits (0, 1)")
    print(f"n_steps        {n_steps}")
    print(f"sampled steps  {n_steps - 1}")
    print("final step     exact all-good DP marginal")
    print(f"phi            {phi}")
    print(f"lambda_base    {lambda_base}")
    print("noise_model    exact product eta")
    print(f"n_samples      {n_samples}")
    print("trigger        enabled")

    print("\nWarming Numba kernels...")
    evolve_many_gate_prefix_terminal(
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
        n_steps,
        8,
    )
    evolve_many_dp_prefix_terminal(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        n_steps,
        8,
    )
    print("Warmup complete.")

    t0 = time.perf_counter()
    contrib_g, trig_g = evolve_many_gate_prefix_terminal(
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
        n_steps,
        n_samples,
    )
    row_g = summarize("gate-wise + final-layer DP", contrib_g, trig_g, time.perf_counter() - t0)

    t0 = time.perf_counter()
    contrib_l, trig_l = evolve_many_dp_prefix_terminal(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        n_steps,
        n_samples,
    )
    row_l = summarize("full-layer DP + final-layer DP", contrib_l, trig_l, time.perf_counter() - t0)

    diff = row_l["mean"] - row_g["mean"]
    combined_se = (row_g["se"] ** 2 + row_l["se"] ** 2) ** 0.5
    print("\nComparison")
    print("----------")
    print(f"mean diff DP-gate        {diff:.6g}")
    print(f"combined SE              {combined_se:.6g}")
    print(f"target var DP/gate       {row_l['var'] / row_g['var']:.6g}")
    print(f"target var*time DP/gate  {row_l['var_time'] / row_g['var_time']:.6g}")


if __name__ == "__main__":
    main()
