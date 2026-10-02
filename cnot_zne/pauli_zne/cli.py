import argparse
import json
import sys
from pathlib import Path
from .config import DEFAULT
from .pipeline import run, refit


def main():
    parser = argparse.ArgumentParser(
        description="CNOT-resolved Heisenberg MPS and ZNE benchmark"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write an editable default JSON config")
    init.add_argument("path")
    calc = sub.add_parser("run", help="Compute curves, references and fits")
    calc.add_argument("--config")
    calc.add_argument("--output", required=True)
    calc.add_argument("--N", type=int)
    calc.add_argument("--L", type=int)
    calc.add_argument("--phi", type=float)
    calc.add_argument("--O", help="N-character Pauli string, qubit0 first")
    calc.add_argument("--s-grid", help="Comma-separated positive scales")
    fit = sub.add_parser("fit", help="Refit saved curves without simulating again")
    fit.add_argument("--run", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--fit-config", help="JSON containing fit settings only")
    fit.add_argument(
        "--observations", help="Optional CSV with s,O means on the saved fit grid"
    )
    args = parser.parse_args()
    try:
        if args.command == "init":
            with Path(args.path).open("x") as f:
                json.dump(DEFAULT, f, indent=2)
                f.write("\n")
            print(f"Wrote {args.path}")
            return
        if args.command == "run":
            circuit = {
                k: getattr(args, k)
                for k in ("N", "L", "phi", "O")
                if getattr(args, k) is not None
            }
            overrides = {"circuit": circuit}
            if args.s_grid is not None:
                overrides["s_grid"] = [float(x) for x in args.s_grid.split(",")]
            out = run(
                args.config,
                args.output,
                overrides,
                progress=lambda s: print(s, flush=True),
            )
        else:
            out = refit(
                args.run,
                args.output,
                args.fit_config,
                args.observations,
                progress=lambda s: print(s, flush=True),
            )
        print(f"Completed: {out}/report.md")
    except (ValueError, OSError, FloatingPointError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(2)
