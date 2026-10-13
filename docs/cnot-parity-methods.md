# Per-CNOT-layer noise with parity-resolved MPS and PEPS

Use `--noise-placement after_cnot` for the sparse Pauli-Lindblad model discussed
by van den Berg and collaborators. After **each complete CNOT layer**, one noise
channel acts on the whole hardware graph, including idle qubits and correlations
between different active gate pairs. A nonempty Heisenberg matching contains
three such insertions, with rotations between them. Empty matchings are skipped,
including their noise, as in the existing `cnot_zne` implementation.

The implementation supports all three single-qubit generators, all nine Pauli
products on each supplied edge, and edges beyond nearest neighbors. Rates can
vary by step, matching, and each of the three CNOT occurrences. The sparse
Lindbladian is exponentiated exactly at each insertion; there is no noise
Trotter approximation. Rates must be finite and nonnegative.

## Installation and quick runs

Use the existing [installation instructions](tensor-network-methods.md).
No additional dependencies are required. The CPU environment is Python 3.12
with `requirements-mps.txt` for MPS, or `requirements-peps.txt` for PEPS.

```bash
# Single-qubit rates lambda_X/Y/Z plus all nine products per adjacent edge.
python tensor_zne.py mps --noise-placement after_cnot \
  --n 6 --steps 2 --sites 2 3 --lam .001 .002 .003 --pair-rate .0001 \
  --cap 16 --guide-cap 64 --scales 0 1 2 4 8 --mode both \
  --out results/cnot_mps

python tensor_zne.py peps --noise-placement after_cnot \
  --nx 2 --ny 3 --steps 1 --sites 0 1 --lam .001 .002 .003 --pair-rate .0001 \
  --cap 8 --boundary-cap 4 --environment-cap 16 \
  --scales 0 1 2 4 8 --mode both --out results/cnot_peps
```

`--pair-rate .0001` assigns .0001 to **each** of XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
on every noise edge; it is not the total pair-noise rate. The default is zero.
Default noise edges are the union of the lattice's nearest-neighbor gate edges.
`--no-residual` disables scalar residual correction. The existing Heisenberg-noise
methods remain the default when `--noise-placement` is omitted.

Optional Metal choices remain `--mps-backend metal`, `--gate-backend metal`, and
`--boundary-backend metal-jacobi`. In the CNOT PEPS path, two-site updates use the
selected gate engine; single-site transfers and bookkeeping remain on CPU.
There is no claim of speedup for these small, sequential elementary updates.

## Why the parity symmetry survives

The complete matching preserves the usual Z2 x Z2 labels. Our earlier exact
four-qubit experiments showed that pushing each noise insertion through its
remaining block-local unitary preserves its product-of-factors representation.

The implementation uses an equivalent **transported-frame** representation to
avoid even the local non-diagonal expansion. During this specific three-CNOT
Heisenberg decomposition, the global symmetry generators, transported by the
ideal prefix, remain products of Paulis at every elementary-gate boundary.
Their local Pauli commutation signs assign a four-entry charge map to each site.
An elementary gate maps its incoming charge convention to its outgoing one.
The compiler verifies this covariance for every gate, including the routing
SWAPs, and verifies restoration of the standard labels after every matching.

This is not projection onto the wrong, fixed intermediate sectors: physical
charge maps actually change during the decomposition. All virtual charge
arithmetic and truncation use the current maps. Standard physical labels are
restored at full-Heisenberg-matching boundaries.

Consequently, the original noise stays **diagonal at its actual insertion time**.
Single-qubit factors are applied locally; the nine commuting generators on a
noise edge are combined into one 4-by-4 diagonal damping table and applied as a
two-site update. This grouping is exact. A single generator has operator rank at
most 2 across the edge; grouping the nine can give rank up to 4. No global PTM or
dense four-qubit effective gate is constructed.

This also preserves the **elementary-path Q definition**: unsigned evolution
uses the absolute transfer of each original rotation and CNOT, with the same
noise factors. No absolute value of a composite signed channel, matrix inverse,
or similarity transform of a potentially singular absolute transfer is used.
The tests include phi=pi/4, where such inverse-based constructions can fail.

## Noise files and API

`--noise-file rates.npz` accepts numeric arrays named `singles`, `pairs`, and
`edges`. Unknown arrays are rejected and pickle loading is disabled.

| Array | Full shape |
|---|---|
| `singles` | `(steps, matchings, 3, N, 3)` |
| `pairs` | `(steps, matchings, 3, E, 3, 3)` |
| `edges` | `(E, 2)` |

The first `3` is chronological CNOT occurrence, not CNOT direction. Axis order
in the final dimensions is X,Y,Z. Step and matching axes are in **forward circuit
time**, even though observable evolution runs backwards. Any trailing shape
that broadcasts to the full shape is accepted, e.g. `(N,3)` or `(E,3,3)`.
Use explicit leading singleton dimensions when omitting an interior axis.

For MPS there are two matchings: even, odd. For rectangular PEPS there are four:
horizontal-even, horizontal-odd, vertical-even, vertical-odd. Physical site
labels are `y*nx+x`. Edges must be unique integer pairs `0 <= i < j < N`; pair
axes refer to i and j in that order. An `edges` array replaces the default graph.
An empty edge list disables pair noise. Missing `singles` uses `--lam`, and
missing `pairs` uses `--pair-rate`. A supplied array replaces the corresponding
CLI default; it is not added to it. JSON output records the noise-file path and
SHA-256 digest. Retain the NPZ alongside results for reproduction.

Example with a longer-range term:

```python
import numpy as np

edges = np.array([(0,1), (1,2), (2,3), (3,4), (4,5), (0,5)])
pairs = np.zeros((len(edges), 3, 3))
pairs[:, 2, 2] = .0002       # ZZ on each listed edge
pairs[-1, 0, 1] = .00005    # additional X_0 Y_5 generator
singles = np.tile([.001, .002, .003], (6,1))
np.savez('rates.npz', singles=singles, pairs=pairs, edges=edges)
```

```bash
python tensor_zne.py mps --noise-placement after_cnot --noise-file rates.npz \
  --n 6 --steps 2 --sites 2 3 --cap 16 --guide-cap 64 \
  --mode both --out results/cnot_graph
```

Python API:

```python
from tensor_zne import evaluate_mps, evaluate_peps, curves

result = evaluate_mps(
    n=6, steps=2, sites=(2,3), cap=16, guide_cap=64,
    noise_placement='after_cnot',
    single_rates=singles, pair_rates=pairs, noise_edges=edges,
)
```

Both APIs accept those three noise arguments. `single_rates=None` uses `lam`.
They are rejected by the historical Heisenberg-noise paths rather than silently
ignored. For PEPS use a graph with site labels valid for its rectangle.

## Conventions, routing and residuals

The new MPS mode follows forward even-then-odd matchings, as in `cnot_zne`.
This differs from the older blocked-MPS backward-sweep convention; do not compare
noise-free outputs between modes without matching the physical circuit ordering.
Initial MPS singlets remain (0,1),(2,3),... . PEPS singlets follow its existing
snake order, documented in the main guide.

MPS now uses physical dimension 4, one Pauli index per qubit, with site-dependent
charge maps. It moves a canonical center rather than canonicalizing the entire
chain for every update. CNOT/SWAP transfers use signed permutations, and noise
uses direct diagonal multiplication. Every SVD ranks singular values globally across sectors
and applies one total bond cap. Optional Metal products use the existing
randomized SVD implementation when a block is large enough.

Nonadjacent noise factors are routed using Pauli-index SWAPs and then swapped
back. MPS uses the chain path; PEPS uses a Manhattan path on the rectangle. These
are computational routing operations, not extra physical noisy gates, and the
charge maps move with them. They introduce extra finite-bond truncations, so
longer-range edges increase cost and require convergence checks.

MPS residual correction observes every two-site truncation, including correlated
noise and routing, and uses the corresponding adjoint of the **same routed
schedule**. Exact guides give the scalar telescoping identity; finite guide
bonds approximate it. Guide checkpoints plus a regenerated chunk use roughly
O(sqrt(number of operations)) snapshots instead of retaining a guide at every
operation. `--terminal` applies only to the older Heisenberg-noise mode: the
CNOT mode evolves the full circuit and has no special terminal split.

PEPS uses its existing spatial simple update and streamed boundary residual
correction. The latter corrects only the contraction of the already-evolved
PEPS; it does not correct gate, correlated-noise, or routing truncations during
spatial evolution. Converge spatial, boundary and environment caps separately.

## Ratio and Q output

The output file schemas are unchanged. The Q-group method is `cnot`, so pass
`methods="cnot"` to `fit_zne_compare.run_zne_fit_compare`. For example:

```python
from fit_zne_compare import run_zne_fit_compare
run_zne_fit_compare(
    q_groups='results/cnot_mps_q_groups.csv',
    observables='results/cnot_mps_observables.csv',
    methods='cnot', s_grid=[1,2,4,8], s_fit_q=[1,2,4,8], s_fit_o=[1,2,4,8],
    q_summary='pointwise_mom', n_trials=100, out_prefix='results/cnot_fit',
)
```

This mode's Q is defined by the elementary CX/rotation paths and generally
differs from the older full-Heisenberg-gate Q. Ratio and Q denominator guards
remain in effect. Undefined Q sectors are reported, not clipped into fit inputs.

## Validation

`tests/test_cnot_parity.py` compares signed and unsigned evolution with direct
dense elementary-circuit calculations, including all weight-1/2 generators,
independently scheduled rates, cross-pair and longer-range noise, idle qubits,
zero noise, and phi=pi/4. It also checks agreement with the existing independent
CNOT-resolved density-matrix solver. Truncated MPS residuals are checked against
exact values; PEPS boundary residuals are checked against exact contraction of
the same evolved network. Metal checks are opt-in with `TEST_PEPS_METAL=1`.
These are correctness checks, not large-system accuracy or speed guarantees.

Reference: [van den Berg and Wocjan, Section 2](https://arxiv.org/html/2311.15408v2#S2).
