import argparse
import csv
import os
import time

import numpy as np

from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def read_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((float(row["s"]), int(row["chi_max"])))
    return done


def append_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        fieldnames = [
            "s",
            "lambda",
            "chi_max",
            "mps_value",
            "runtime_s",
            "max_bond",
            "max_discarded_weight",
            "svd_method",
            "noise_model",
            "noise_placement",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=36)
    parser.add_argument("--n-steps", type=int, default=12)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.3)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-values", type=str, default="0,1,2,4,16")
    parser.add_argument("--chi-values", type=str, default="128,256,350,512")
    parser.add_argument("--svd-method", type=str, default="auto")
    parser.add_argument("--noise-model", type=str, default="legacy_sum")
    parser.add_argument("--noise-placement", type=str, default="gate")
    parser.add_argument("--out-csv", type=str, default="mps_Z6Z10_chi_sweep.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    s_values = parse_float_list(args.s_values)
    chi_values = parse_int_list(args.chi_values)
    target = pauli_zz(args.n_qubits, args.q1, args.q2)
    done = read_done(args.out_csv) if args.resume else set()

    print("MPS chi convergence sweep", flush=True)
    print("-------------------------", flush=True)
    print(f"n_qubits        {args.n_qubits}", flush=True)
    print(f"n_steps         {args.n_steps}", flush=True)
    print(f"target          Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"phi             {args.phi}", flush=True)
    print(f"lambda_base     {args.lambda_base}", flush=True)
    print(f"s_values        {s_values}", flush=True)
    print(f"chi_values      {chi_values}", flush=True)
    print(f"svd_method      {args.svd_method}", flush=True)
    print(f"noise_model     {args.noise_model}", flush=True)
    print(f"noise_placement {args.noise_placement}", flush=True)
    print(f"out_csv         {args.out_csv}", flush=True)

    for s in s_values:
        lam = np.full((args.n_qubits, 3), args.lambda_base * s, dtype=np.float64)
        for chi in chi_values:
            key = (float(s), int(chi))
            if key in done:
                print(f"skip s={s:g} chi={chi} already in {args.out_csv}", flush=True)
                continue

            t0 = time.perf_counter()
            value, _, info = evolve_observable_backward_mps(
                target,
                n_qubits=args.n_qubits,
                phi=args.phi,
                lam_xyz=lam,
                n_steps=args.n_steps,
                chi_max=chi,
                svd_method=args.svd_method,
                oversample=24,
                n_iter=1,
                return_mps=True,
                noise_model=args.noise_model,
                noise_placement=args.noise_placement,
            )
            runtime = time.perf_counter() - t0
            max_bond = int(np.max(info["bond_dims"]))
            max_discarded = float(np.max(info["discarded_by_backward_step"]))
            row = {
                "s": float(s),
                "lambda": float(args.lambda_base * s),
                "chi_max": int(chi),
                "mps_value": float(value),
                "runtime_s": float(runtime),
                "max_bond": max_bond,
                "max_discarded_weight": max_discarded,
                "svd_method": args.svd_method,
                "noise_model": args.noise_model,
                "noise_placement": args.noise_placement,
            }
            append_row(args.out_csv, row)
            print(
                f"s={s:g} chi={chi:4d} value={value:.12g} "
                f"time={runtime:.2f}s max_bond={max_bond} "
                f"max_discard={max_discarded:.3g}",
                flush=True,
            )

    print(f"\nwrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
