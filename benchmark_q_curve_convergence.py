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
    if ":" in text:
        a, b, n = text.split(":")
        return np.linspace(float(a), float(b), int(n), dtype=np.float64)
    return np.array([float(x) for x in text.split(",")], dtype=np.float64)


def find_s1_index(s_grid):
    idx = int(np.argmin(np.abs(s_grid - 1.0)))
    if abs(float(s_grid[idx]) - 1.0) > 1.0e-12:
        raise ValueError("s_grid must include s=1.0 so target observable can share the curve sums")
    return idx


@njit
def gate_prefix_q_curve_one(
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

    tp_num, tp_den, tm_num, tm_den = layer.terminal_all_good_qpm_grid_sums(
        pauli,
        sign,
        terminal_trans,
        terminal_signed,
        terminal_abs,
        terminal_n_branches,
        eta_xyz,
        s_grid,
    )
    pref = amp * np.exp(log_wi)
    return pref, damp, tp_num, tp_den, tm_num, tm_den, triggered


@njit
def dp_prefix_q_curve_one(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
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

        damp *= layer.pauli_noise_factor(P_even, eta_xyz)
        damp *= layer.pauli_noise_factor(P_out, eta_xyz)
        sign = np.int8(sign * step_sign)
        for q in range(n_qubits):
            P[q] = P_out[q]

    tp_num, tp_den, tm_num, tm_den = layer.terminal_all_good_qpm_grid_sums(
        P, sign, trans_codes, signed_coeffs, abs_coeffs, n_branches, eta_xyz, s_grid
    )
    return amp, damp, tp_num, tp_den, tm_num, tm_den, triggered


@njit(parallel=True)
def gate_q_curve_aggregate(
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
    s_grid,
    s1_idx,
):
    ns = s_grid.shape[0]
    n_threads = get_num_threads()
    sx_t = np.zeros(n_threads, dtype=np.float64)
    sx2_t = np.zeros(n_threads, dtype=np.float64)
    spd_t = np.zeros(n_threads, dtype=np.float64)
    smd_t = np.zeros(n_threads, dtype=np.float64)
    ntr_t = np.zeros(n_threads, dtype=np.int64)
    spn_t = np.zeros((n_threads, ns), dtype=np.float64)
    smn_t = np.zeros((n_threads, ns), dtype=np.float64)

    for i in prange(n_samples):
        tid = get_thread_id()
        pref, damp, tp_num, tp_den, tm_num, tm_den, tr = gate_prefix_q_curve_one(
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
            s_grid,
        )
        spd_t[tid] += pref * tp_den
        smd_t[tid] += pref * tm_den
        log_damp = np.log(damp)
        for k in range(ns):
            scale = pref * np.exp(log_damp * s_grid[k])
            pn = scale * tp_num[k]
            mn = scale * tm_num[k]
            spn_t[tid, k] += pn
            smn_t[tid, k] += mn
            if k == s1_idx:
                x = pn - mn
                sx_t[tid] += x
                sx2_t[tid] += x * x
        if tr:
            ntr_t[tid] += 1

    sx = 0.0
    sx2 = 0.0
    spd = 0.0
    smd = 0.0
    ntr = 0
    spn = np.zeros(ns, dtype=np.float64)
    smn = np.zeros(ns, dtype=np.float64)
    for tid in range(n_threads):
        sx += sx_t[tid]
        sx2 += sx2_t[tid]
        spd += spd_t[tid]
        smd += smd_t[tid]
        ntr += ntr_t[tid]
        for k in range(ns):
            spn[k] += spn_t[tid, k]
            smn[k] += smn_t[tid, k]
    return sx, sx2, spn, spd, smn, smd, ntr


@njit(parallel=True)
def dp_q_curve_aggregate(
    init_pauli,
    trans_codes,
    signed_coeffs,
    abs_coeffs,
    n_branches,
    eta_xyz,
    n_steps,
    n_samples,
    s_grid,
    s1_idx,
):
    ns = s_grid.shape[0]
    n_threads = get_num_threads()
    sx_t = np.zeros(n_threads, dtype=np.float64)
    sx2_t = np.zeros(n_threads, dtype=np.float64)
    spd_t = np.zeros(n_threads, dtype=np.float64)
    smd_t = np.zeros(n_threads, dtype=np.float64)
    ntr_t = np.zeros(n_threads, dtype=np.int64)
    spn_t = np.zeros((n_threads, ns), dtype=np.float64)
    smn_t = np.zeros((n_threads, ns), dtype=np.float64)

    for i in prange(n_samples):
        tid = get_thread_id()
        pref, damp, tp_num, tp_den, tm_num, tm_den, tr = dp_prefix_q_curve_one(
            init_pauli,
            trans_codes,
            signed_coeffs,
            abs_coeffs,
            n_branches,
            eta_xyz,
            n_steps,
            s_grid,
        )
        spd_t[tid] += pref * tp_den
        smd_t[tid] += pref * tm_den
        log_damp = np.log(damp)
        for k in range(ns):
            scale = pref * np.exp(log_damp * s_grid[k])
            pn = scale * tp_num[k]
            mn = scale * tm_num[k]
            spn_t[tid, k] += pn
            smn_t[tid, k] += mn
            if k == s1_idx:
                x = pn - mn
                sx_t[tid] += x
                sx2_t[tid] += x * x
        if tr:
            ntr_t[tid] += 1

    sx = 0.0
    sx2 = 0.0
    spd = 0.0
    smd = 0.0
    ntr = 0
    spn = np.zeros(ns, dtype=np.float64)
    smn = np.zeros(ns, dtype=np.float64)
    for tid in range(n_threads):
        sx += sx_t[tid]
        sx2 += sx2_t[tid]
        spd += spd_t[tid]
        smd += smd_t[tid]
        ntr += ntr_t[tid]
        for k in range(ns):
            spn[k] += spn_t[tid, k]
            smn[k] += smn_t[tid, k]
    return sx, sx2, spn, spd, smn, smd, ntr


def row_from_aggregate(agg, n_samples, runtime):
    sx, sx2, spn, spd, smn, smd, ntr = agg
    mean = sx / n_samples
    var = (sx2 - sx * sx / n_samples) / (n_samples - 1)
    se = (var / n_samples) ** 0.5
    return {
        "mean": float(mean),
        "var": float(var),
        "se": float(se),
        "q_plus": np.asarray(spn / spd, dtype=np.float64),
        "q_minus": np.asarray(smn / smd, dtype=np.float64),
        "plus_num": np.asarray(spn, dtype=np.float64),
        "plus_den": float(spd),
        "minus_num": np.asarray(smn, dtype=np.float64),
        "minus_den": float(smd),
        "runtime": float(runtime),
        "triggered_frac": float(ntr / n_samples),
    }


def summarize_curve(name, rows, s_grid):
    means = np.array([r["mean"] for r in rows])
    ses = np.array([r["se"] for r in rows])
    runtimes = np.array([r["runtime"] for r in rows])
    q_plus = np.stack([r["q_plus"] for r in rows])
    q_minus = np.stack([r["q_minus"] for r in rows])

    plus_num = np.sum(np.stack([r["plus_num"] for r in rows]), axis=0)
    plus_den = np.sum(np.array([r["plus_den"] for r in rows]))
    minus_num = np.sum(np.stack([r["minus_num"] for r in rows]), axis=0)
    minus_den = np.sum(np.array([r["minus_den"] for r in rows]))
    q_plus_ref = plus_num / plus_den
    q_minus_ref = minus_num / minus_den

    q_plus_std = np.std(q_plus, axis=0, ddof=1)
    q_minus_std = np.std(q_minus, axis=0, ddof=1)
    q_plus_rms_err = np.sqrt(np.mean((q_plus - q_plus_ref[None, :]) ** 2, axis=1))
    q_minus_rms_err = np.sqrt(np.mean((q_minus - q_minus_ref[None, :]) ** 2, axis=1))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"target mean of batch means   {np.mean(means):.12g}")
    print(f"target batch spread std      {np.std(means, ddof=1):.6g}")
    print(f"target mean reported SE      {np.mean(ses):.6g}")
    print(f"target spread / SE           {np.std(means, ddof=1) / np.mean(ses):.6g}")
    print(f"runtime mean                 {np.mean(runtimes):.3f}s")
    print(f"triggered frac mean          {np.mean([r['triggered_frac'] for r in rows]):.6g}")
    print(f"Q+ mean RMS batch error      {np.mean(q_plus_rms_err):.6g}")
    print(f"Q- mean RMS batch error      {np.mean(q_minus_rms_err):.6g}")
    print("s-grid curve spread")
    for k, s in enumerate(s_grid):
        print(
            f"  s={s:5.2f}  "
            f"Q+ ref {q_plus_ref[k]:.10f} std {q_plus_std[k]:.4g}  "
            f"Q- ref {q_minus_ref[k]:.10f} std {q_minus_std[k]:.4g}"
        )

    return {
        "target_spread": float(np.std(means, ddof=1)),
        "target_se": float(np.mean(ses)),
        "q_plus_rms": float(np.mean(q_plus_rms_err)),
        "q_minus_rms": float(np.mean(q_minus_rms_err)),
        "q_plus_std_mean": float(np.mean(q_plus_std)),
        "q_minus_std_mean": float(np.mean(q_minus_std)),
        "runtime": float(np.mean(runtimes)),
    }


def write_repeat_rows(path, gate_rows, dp_rows, s_grid):
    fieldnames = [
        "method",
        "repeat",
        "s",
        "target_mean_s1",
        "target_se_s1",
        "q_plus",
        "q_minus",
        "runtime_s",
        "triggered_frac",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in (("gate", gate_rows), ("layer", dp_rows)):
            for rep, row in enumerate(rows, start=1):
                for k, s in enumerate(s_grid):
                    writer.writerow(
                        {
                            "method": method,
                            "repeat": rep,
                            "s": float(s),
                            "target_mean_s1": row["mean"],
                            "target_se_s1": row["se"],
                            "q_plus": float(row["q_plus"][k]),
                            "q_minus": float(row["q_minus"][k]),
                            "runtime_s": row["runtime"],
                            "triggered_frac": row["triggered_frac"],
                        }
                    )


def safe_ratio(num, den):
    if den == 0.0:
        if num == 0.0:
            return np.nan
        return np.inf
    return num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--q1", type=int, default=0)
    parser.add_argument("--q2", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-repeats", type=int, default=4)
    parser.add_argument("--phi", type=float, default=0.15)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", type=str, default="0,0.5,1,1.5,2")
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    s_grid = parse_s_grid(args.s_grid)
    s1_idx = find_s1_index(s_grid)

    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[args.q1] = 3
    init_pauli[args.q2] = 3

    lam_xyz = np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    eta_xyz = product_eta_from_lambda(lam_xyz)

    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    print("Q-curve convergence benchmark", flush=True)
    print("-----------------------------", flush=True)
    print(f"n_qubits       {args.n_qubits}", flush=True)
    print(f"n_steps        {args.n_steps}", flush=True)
    print(f"n_samples      {args.n_samples}", flush=True)
    print(f"n_repeats      {args.n_repeats}", flush=True)
    print(f"s_grid         {s_grid}", flush=True)
    print(f"target         Z_{args.q1} Z_{args.q2}", flush=True)
    print("final step     exact all-good Qpm curve DP marginal", flush=True)
    print(f"phi            {args.phi}", flush=True)
    print(f"lambda_base    {args.lambda_base}", flush=True)
    print("noise_model    exact product eta", flush=True)

    print("\nWarming Numba kernels...", flush=True)
    gate_q_curve_aggregate(
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
        s_grid,
        s1_idx,
    )
    dp_q_curve_aggregate(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        eta_xyz,
        args.n_steps,
        8,
        s_grid,
        s1_idx,
    )
    print("Warmup complete.", flush=True)

    gate_rows = []
    dp_rows = []
    for rep in range(args.n_repeats):
        t0 = time.perf_counter()
        agg_g = gate_q_curve_aggregate(
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
            s_grid,
            s1_idx,
        )
        gate_rows.append(row_from_aggregate(agg_g, args.n_samples, time.perf_counter() - t0))
        print(f"completed gate repeat {rep + 1}/{args.n_repeats}", flush=True)

        t0 = time.perf_counter()
        agg_l = dp_q_curve_aggregate(
            init_pauli,
            trans_l,
            signed_l,
            abs_l,
            n_branches_l,
            eta_xyz,
            args.n_steps,
            args.n_samples,
            s_grid,
            s1_idx,
        )
        dp_rows.append(row_from_aggregate(agg_l, args.n_samples, time.perf_counter() - t0))
        print(f"completed DP repeat {rep + 1}/{args.n_repeats}", flush=True)

    gate_summary = summarize_curve("gate-wise + final-layer Q-curve DP", gate_rows, s_grid)
    dp_summary = summarize_curve("full-layer DP + final-layer Q-curve DP", dp_rows, s_grid)

    if args.out_csv:
        write_repeat_rows(args.out_csv, gate_rows, dp_rows, s_grid)
        print(f"\nwrote {args.out_csv}", flush=True)

    print("\nRatios DP/gate, lower is better for spreads/time", flush=True)
    print("-----------------------------------------------", flush=True)
    print(f"target spread       {safe_ratio(dp_summary['target_spread'], gate_summary['target_spread']):.6g}", flush=True)
    print(f"target reported SE  {safe_ratio(dp_summary['target_se'], gate_summary['target_se']):.6g}", flush=True)
    print(f"Q+ RMS curve error  {safe_ratio(dp_summary['q_plus_rms'], gate_summary['q_plus_rms']):.6g}", flush=True)
    print(f"Q- RMS curve error  {safe_ratio(dp_summary['q_minus_rms'], gate_summary['q_minus_rms']):.6g}", flush=True)
    print(f"Q+ mean grid std    {safe_ratio(dp_summary['q_plus_std_mean'], gate_summary['q_plus_std_mean']):.6g}", flush=True)
    print(f"Q- mean grid std    {safe_ratio(dp_summary['q_minus_std_mean'], gate_summary['q_minus_std_mean']):.6g}", flush=True)
    print(f"runtime             {safe_ratio(dp_summary['runtime'], gate_summary['runtime']):.6g}", flush=True)


if __name__ == "__main__":
    main()
