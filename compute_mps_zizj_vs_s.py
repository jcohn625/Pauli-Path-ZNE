import argparse
import csv
import time

import numpy as np

from pauli_mps_solver import evolve_observable_backward_mps, pauli_zz


def parse_s_grid(text):
    if ":" in text:
        start, stop, count = text.split(":")
        return np.linspace(float(start), float(stop), int(count), dtype=np.float64)
    return np.array([float(x) for x in text.split(",")], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--q1", type=int, default=6)
    parser.add_argument("--q2", type=int, default=10)
    parser.add_argument("--phi", type=float, default=0.15)
    parser.add_argument("--lambda-base", type=float, default=1.0e-3)
    parser.add_argument("--s-grid", type=str, default="0:3:13")
    parser.add_argument("--chi-max", type=int, default=128)
    parser.add_argument("--svd-method", type=str, default="randomized")
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--out-csv", type=str, default="mps_Z6Z10_vs_s.csv")
    args = parser.parse_args()

    s_grid = parse_s_grid(args.s_grid)
    target = pauli_zz(args.n_qubits, args.q1, args.q2)

    rows = []
    print("MPS Z_i Z_j vs s", flush=True)
    print("----------------", flush=True)
    print(f"n_qubits        {args.n_qubits}", flush=True)
    print(f"n_steps         {args.n_steps}", flush=True)
    print(f"target          Z_{args.q1} Z_{args.q2}", flush=True)
    print(f"phi             {args.phi}", flush=True)
    print(f"lambda_base     {args.lambda_base}", flush=True)
    print(f"s_grid          {s_grid}", flush=True)
    print("noise_model     independent", flush=True)
    print("noise_placement layer", flush=True)
    print(f"chi_max         {args.chi_max}", flush=True)
    print(f"cutoff          {args.cutoff}", flush=True)
    print(f"svd_method      {args.svd_method}", flush=True)

    for s in s_grid:
        lam = np.full((args.n_qubits, 3), args.lambda_base * s, dtype=np.float64)
        t0 = time.perf_counter()
        value, _, info = evolve_observable_backward_mps(
            target,
            n_qubits=args.n_qubits,
            phi=args.phi,
            lam_xyz=lam,
            n_steps=args.n_steps,
            chi_max=args.chi_max,
            cutoff=args.cutoff,
            svd_method=args.svd_method,
            noise_model="independent",
            noise_placement="layer",
            use_lightcone=True,
            return_mps=True,
        )
        runtime = time.perf_counter() - t0
        max_bond = int(np.max(info["bond_dims"]))
        max_discard = float(np.max(info["discarded_by_backward_step"]))
        rows.append((float(s), float(value), runtime, max_bond, max_discard))
        print(
            f"s={s:.6g}  value={value:.12g}  "
            f"time={runtime:.3f}s  max_bond={max_bond}  max_discard={max_discard:.3g}",
            flush=True,
        )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["s", "mps_value", "runtime_s", "max_bond", "max_discarded_weight"])
        writer.writerows(rows)
    print(f"\nwrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
