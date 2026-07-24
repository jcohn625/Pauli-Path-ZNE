import argparse
import csv
import os
import time

import numpy as np

import Pauli_path_Heis_mixture_trigger as gate
import Pauli_path_Heis_full_layer_sampling_restricted as layer
from benchmark_final_layer_marginal import (
    evolve_many_dp_prefix_terminal,
    evolve_many_gate_prefix_terminal,
    product_eta_from_lambda,
)


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def read_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((float(row["s"]), row["method"]))
    return done


def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        fieldnames = [
            "s",
            "lambda",
            "method",
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
            "noise_model",
            "noise_placement",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(contribs, triggered, runtime):
    n = contribs.shape[0]
    mean = float(np.mean(contribs))
    var = float(np.var(contribs, ddof=1))
    std = var**0.5
    se = std / n**0.5
    return {
        "mean": mean,
        "var": var,
        "std": std,
        "se": se,
        "runtime_s": float(runtime),
        "var_time": float(var * runtime),
        "triggered_frac": float(np.mean(triggered)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=36)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.3)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-values", type=str, default="0,1,2,4,16")
    parser.add_argument("--n-samples", type=int, default=10_000_000)
    parser.add_argument("--out-csv", type=str, default="mc_Z6Z10_noise_sweep_final_layer.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    s_values = parse_float_list(args.s_values)
    done = read_done(args.out_csv) if args.resume else set()

    init_pauli = np.zeros(args.n_qubits, dtype=np.int8)
    init_pauli[args.q1] = 3
    init_pauli[args.q2] = 3

    even_gates, odd_gates = gate.make_even_odd_layers(args.n_qubits)
    trans_g, probs_g, is_comm_g, sign_g, amp_factor_g = gate.build_transition_tables(args.phi)
    trans_l, signed_l, abs_l, n_branches_l, *_ = layer.build_full_layer_tables(args.phi)

    print("Final-layer marginalized MC noise sweep", flush=True)
    print("---------------------------------------", flush=True)
    print(f"n_qubits        {args.n_qubits}", flush=True)
    print(f"n_steps         {args.n_steps}", flush=True)
    print(f"sampled steps   {args.n_steps - 1}", flush=True)
    print(f"target          Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"phi             {args.phi}", flush=True)
    print(f"lambda_base     {args.lambda_base}", flush=True)
    print(f"s_values        {s_values}", flush=True)
    print(f"n_samples       {args.n_samples}", flush=True)
    print("trigger         enabled", flush=True)
    print("final layer     exact all-good DP marginal", flush=True)
    print("noise_model     independent/product eta", flush=True)
    print("noise_placement full layer after each half-layer", flush=True)
    print(f"out_csv         {args.out_csv}", flush=True)

    warm_eta = product_eta_from_lambda(
        np.full((args.n_qubits, 3), args.lambda_base, dtype=np.float64)
    )
    print("\nWarming Numba kernels...", flush=True)
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
        warm_eta,
        args.n_steps,
        8,
    )
    evolve_many_dp_prefix_terminal(
        init_pauli,
        trans_l,
        signed_l,
        abs_l,
        n_branches_l,
        warm_eta,
        args.n_steps,
        8,
    )
    print("Warmup complete.", flush=True)

    for s in s_values:
        lam = np.full((args.n_qubits, 3), args.lambda_base * s, dtype=np.float64)
        eta_xyz = product_eta_from_lambda(lam)

        if (float(s), "gate") not in done:
            t0 = time.perf_counter()
            contribs, triggered = evolve_many_gate_prefix_terminal(
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
            row = summarize(contribs, triggered, time.perf_counter() - t0)
            row.update(
                {
                    "s": float(s),
                    "lambda": float(args.lambda_base * s),
                    "method": "gate",
                    "n_samples": args.n_samples,
                    "n_qubits": args.n_qubits,
                    "n_steps": args.n_steps,
                    "phi": args.phi,
                    "q1": args.q1,
                    "q2": args.q2,
                    "noise_model": "independent",
                    "noise_placement": "layer",
                }
            )
            append_row(args.out_csv, row)
            print(
                f"s={s:g} gate mean={row['mean']:.12g} SE={row['se']:.4g} "
                f"var={row['var']:.4g} time={row['runtime_s']:.2f}s "
                f"trigger={row['triggered_frac']:.4g}",
                flush=True,
            )
            del contribs, triggered
        else:
            print(f"skip s={s:g} gate already in {args.out_csv}", flush=True)

        if (float(s), "layer") not in done:
            t0 = time.perf_counter()
            contribs, triggered = evolve_many_dp_prefix_terminal(
                init_pauli,
                trans_l,
                signed_l,
                abs_l,
                n_branches_l,
                eta_xyz,
                args.n_steps,
                args.n_samples,
            )
            row = summarize(contribs, triggered, time.perf_counter() - t0)
            row.update(
                {
                    "s": float(s),
                    "lambda": float(args.lambda_base * s),
                    "method": "layer",
                    "n_samples": args.n_samples,
                    "n_qubits": args.n_qubits,
                    "n_steps": args.n_steps,
                    "phi": args.phi,
                    "q1": args.q1,
                    "q2": args.q2,
                    "noise_model": "independent",
                    "noise_placement": "layer",
                }
            )
            append_row(args.out_csv, row)
            print(
                f"s={s:g} layer mean={row['mean']:.12g} SE={row['se']:.4g} "
                f"var={row['var']:.4g} time={row['runtime_s']:.2f}s "
                f"trigger={row['triggered_frac']:.4g}",
                flush=True,
            )
            del contribs, triggered
        else:
            print(f"skip s={s:g} layer already in {args.out_csv}", flush=True)

    print(f"\nwrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
