import argparse
import csv
import time

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

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


def parse_s_grid(text):
    return np.array([float(x) for x in text.split(",") if x.strip()], dtype=np.float64)


@njit
def gate_prefix_terminal_grid_one(
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
    eta_base,
    n_steps,
    s_grid,
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

        damp = gate.apply_noise(pauli, eta_base, damp)

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

        damp = gate.apply_noise(pauli, eta_base, damp)

    terminal_grid = layer.terminal_all_good_contribution_grid(
        pauli,
        terminal_trans,
        terminal_signed,
        terminal_abs,
        terminal_n_branches,
        eta_base,
        s_grid,
    )
    pref = float(sign) * amp * np.exp(log_wi)
    return pref, damp, terminal_grid, triggered


@njit
def dp_prefix_terminal_grid_one(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_base,
    n_steps,
    s_grid,
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

        damp *= layer.pauli_noise_factor(P_even, eta_base)
        damp *= layer.pauli_noise_factor(P_out, eta_base)
        sign = np.int8(sign * step_sign)
        for q in range(n_qubits):
            P[q] = P_out[q]

    terminal_grid = layer.terminal_all_good_contribution_grid(
        P,
        trans_codes,
        signed_coeffs,
        abs_coeffs,
        n_branches,
        eta_base,
        s_grid,
    )
    return float(sign) * amp, damp, terminal_grid, triggered


@njit(parallel=True)
def gate_grid_aggregate(
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
    eta_base,
    n_steps,
    n_samples,
    s_grid,
):
    ns = s_grid.shape[0]
    n_threads = get_num_threads()
    sx_t = np.zeros((n_threads, ns), dtype=np.float64)
    sx2_t = np.zeros((n_threads, ns), dtype=np.float64)
    ntr_t = np.zeros(n_threads, dtype=np.int64)

    for i in prange(n_samples):
        tid = get_thread_id()
        pref, damp, terminal_grid, triggered = gate_prefix_terminal_grid_one(
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
            eta_base,
            n_steps,
            s_grid,
        )
        log_damp = np.log(damp)
        for k in range(ns):
            x = pref * np.exp(log_damp * s_grid[k]) * terminal_grid[k]
            sx_t[tid, k] += x
            sx2_t[tid, k] += x * x
        if triggered:
            ntr_t[tid] += 1

    sx = np.zeros(ns, dtype=np.float64)
    sx2 = np.zeros(ns, dtype=np.float64)
    ntr = 0
    for tid in range(n_threads):
        ntr += ntr_t[tid]
        for k in range(ns):
            sx[k] += sx_t[tid, k]
            sx2[k] += sx2_t[tid, k]
    return sx, sx2, ntr


@njit(parallel=True)
def dp_grid_aggregate(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_base,
    n_steps,
    n_samples,
    s_grid,
):
    ns = s_grid.shape[0]
    n_threads = get_num_threads()
    sx_t = np.zeros((n_threads, ns), dtype=np.float64)
    sx2_t = np.zeros((n_threads, ns), dtype=np.float64)
    ntr_t = np.zeros(n_threads, dtype=np.int64)

    for i in prange(n_samples):
        tid = get_thread_id()
        pref, damp, terminal_grid, triggered = dp_prefix_terminal_grid_one(
            init_pauli,
            trans_codes,
            signed_coeffs,
            abs_coeffs,
            n_branches,
            eta_base,
            n_steps,
            s_grid,
        )
        log_damp = np.log(damp)
        for k in range(ns):
            x = pref * np.exp(log_damp * s_grid[k]) * terminal_grid[k]
            sx_t[tid, k] += x
            sx2_t[tid, k] += x * x
        if triggered:
            ntr_t[tid] += 1

    sx = np.zeros(ns, dtype=np.float64)
    sx2 = np.zeros(ns, dtype=np.float64)
    ntr = 0
    for tid in range(n_threads):
        ntr += ntr_t[tid]
        for k in range(ns):
            sx[k] += sx_t[tid, k]
            sx2[k] += sx2_t[tid, k]
    return sx, sx2, ntr


def rows_from_aggregate(method, agg, s_grid, n_samples, runtime, args):
    sx, sx2, ntr = agg
    rows = []
    for k, s in enumerate(s_grid):
        mean = sx[k] / n_samples
        var = (sx2[k] - sx[k] * sx[k] / n_samples) / (n_samples - 1)
        std = var**0.5
        se = std / n_samples**0.5
        rows.append(
            {
                "method": method,
                "s": float(s),
                "lambda": float(args.lambda_base * s),
                "mean": float(mean),
                "var": float(var),
                "std": float(std),
                "se": float(se),
                "runtime_s": float(runtime),
                "var_time": float(var * runtime),
                "triggered_frac": float(ntr / n_samples),
                "n_samples": int(n_samples),
                "n_qubits": int(args.n_qubits),
                "n_steps": int(args.n_steps),
                "phi": float(args.phi),
                "q1": int(args.q1),
                "q2": int(args.q2),
            }
        )
    return rows


def write_rows(path, rows):
    fieldnames = [
        "method",
        "s",
        "lambda",
        "mean",
        "var",
        "std",
        "se",
        "runtime_s",
        "var_time",
        "triggered_frac",
        "n_samples",
        "n_qubits",
        "n_steps",
        "phi",
        "q1",
        "q2",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(title, rows):
    print(f"\n{title}", flush=True)
    print("-" * len(title), flush=True)
    for row in rows:
        print(
            f"s={row['s']:5g} mean={row['mean']:.12g} "
            f"SE={row['se']:.4g} var={row['var']:.4g}",
            flush=True,
        )
    if rows:
        print(
            f"runtime={rows[0]['runtime_s']:.3f}s "
            f"triggered={rows[0]['triggered_frac']:.5g}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=36)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.3)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", type=str, default="0,0.5,1,2,4,8,16")
    parser.add_argument("--n-samples", type=int, default=10_000_000)
    parser.add_argument("--out-csv", type=str, default="mc_Z6Z10_final_layer_s_grid.csv")
    args = parser.parse_args()

    s_grid = parse_s_grid(args.s_grid)
    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[args.q1] = 3
    init_pauli[args.q2] = 3

    lam_base = np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    eta_base = product_eta_from_lambda(lam_base)

    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    print("Final-layer target s-grid MC benchmark", flush=True)
    print("--------------------------------------", flush=True)
    print(f"n_qubits       {args.n_qubits}", flush=True)
    print(f"n_steps        {args.n_steps}", flush=True)
    print(f"sampled steps  {args.n_steps - 1}", flush=True)
    print(f"target         Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"phi            {args.phi}", flush=True)
    print(f"lambda_base    {args.lambda_base}", flush=True)
    print(f"s_grid         {s_grid}", flush=True)
    print(f"n_samples      {args.n_samples}", flush=True)
    print("trigger        enabled", flush=True)
    print("final layer    exact all-good signed DP over s-grid", flush=True)
    print("noise_model    independent/product eta", flush=True)
    print("noise_scale    prefix damp^s and final DP damp^s", flush=True)

    print("\nWarming Numba kernels...", flush=True)
    gate_grid_aggregate(
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
        eta_base,
        args.n_steps,
        8,
        s_grid,
    )
    dp_grid_aggregate(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_base,
        args.n_steps,
        8,
        s_grid,
    )
    print("Warmup complete.", flush=True)

    t0 = time.perf_counter()
    agg_gate = gate_grid_aggregate(
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
        eta_base,
        args.n_steps,
        args.n_samples,
        s_grid,
    )
    gate_rows = rows_from_aggregate("gate", agg_gate, s_grid, args.n_samples, time.perf_counter() - t0, args)
    print_rows("gate-wise + final-layer DP", gate_rows)

    t0 = time.perf_counter()
    agg_dp = dp_grid_aggregate(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_base,
        args.n_steps,
        args.n_samples,
        s_grid,
    )
    dp_rows = rows_from_aggregate("layer", agg_dp, s_grid, args.n_samples, time.perf_counter() - t0, args)
    print_rows("layer-wise + final-layer DP", dp_rows)

    rows = gate_rows + dp_rows
    write_rows(args.out_csv, rows)
    print(f"\nwrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
