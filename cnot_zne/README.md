# CNOT-resolved MPS zero-noise extrapolation

A standalone, configurable benchmark for a 1-D open-chain Heisenberg circuit. It computes classical MPS estimates of **O(s), R(s)=O(s)/O(0), Q+ and Q−**, then compares ratio, direct-Q, single-exponential, and dual-exponential extrapolation on the same noise scales. Shot counts can differ at every point.

This folder is self-contained: no imports from a personal checkout, no absolute paths, no credentials, and no dependence on the earlier scratch scripts. It can be copied into the original repository or installed on its own. Nothing is automatically committed or pushed.

## Quick start

From this folder, with Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
python -m pauli_zne run --config examples/small.json --output runs/small
python -m pytest -q
```

You can also run `python -m pauli_zne` directly from this folder if NumPy and SciPy are already installed. `pauli-zne` is the equivalent installed command.

The small example runs N=6, L=2, three MPS bond dimensions, an independent exact reference, and three paired shot scenarios. Start there before the more expensive example:

```bash
python -m pauli_zne run --config examples/n32.json --output runs/n32
```

The N32 example is a configuration, **not a claim that the previous coarse-noise N32 results carry over**. This model resolves noise at every CNOT layer. The current implementation sweeps the full chain and does not yet implement the old 10+2 contraction optimization. N32/L12/high-bond runs can be expensive. Reference calculations stop at the configured bond budget and report when convergence has not been established.

On macOS with Accelerate, `VECLIB_MAXIMUM_THREADS=1` controls the backend's threading. On OpenBLAS use `OPENBLAS_NUM_THREADS=1`. SciPy's `gesvd` driver is used, with reconstruction checks before truncation; randomized range finding includes QR stabilization and configurable power iterations.

## Interactive notebook

Open [ZNE_workflow.ipynb](ZNE_workflow.ipynb) for the mathematical walkthrough and four-cell workflow:

1. Circuit, observable, four channel models, scale grid and budgets.
2. MPS signed/unsigned curves, ratios, Q modes and independent references.
3. All four noiseless fits and paired per-point shot scenarios.
4. Error/convergence tables, six diagnostic plots and exports under `runs/`.

Install notebook dependencies with `python -m pip install -e '.[notebook]'`, then launch `jupyter lab ZNE_workflow.ipynb`. The notebook starts with N6/L2 and can be edited for larger systems. It separates simulation from fitting, so shot budgets can be changed without recalculating MPS. Saved outputs show the default example; rerun all four cells after changing circuit or noise inputs.

## High-level Python functions

The notebook delegates simulation, fitting, tables and plotting to `pauli_zne.workflow`:

```python
from pauli_zne.workflow import (
    define_experiment, calculate_curves, fit_curves, analyze_results,
)

experiment = define_experiment(N=6, L=2, O="IIZIZI")
calculation = calculate_curves(experiment, s_grid=[1, 2, 4, 8], bond_dims=[16, 32, 64])
fits = fit_curves(calculation, shots={"1m_each": 1_000_000}, trials=200)
analysis = analyze_results(fits, save=True, output_root="runs")
```

- `define_experiment`: circuit and noise definitions, with optional `mps`, `reference_options`, and `rates_directory` overrides. O defaults to centered ZZ when omitted.
- `calculate_curves`: MPS and reference calculations. Returns `calculation.curves`, `.reference`, `.contractions`, `.curve_rows`, and `.warnings`. It copies inputs so later edits cannot silently change existing results.
- `fit_curves`: all four fits, optionally with custom `shots`, `trials`, `seed`, `rate_bounds`, `amplitude_bound`, or `observations`. Returns `.noiseless`, `.shot_results`, and `.exponential_trials`. It performs no MPS operations; call it repeatedly to explore shot budgets.
- `analyze_results`: diagnostic tables, six plots, and optional exports. Returns `.figure`, `.fit_table`, `.shot_table`, `.curve_table`, `.error_table`, and `.output_dir`. Use `save=False` for in-memory analysis, or `show=False` in scripts.
- `save_results`: export a `FitResults` object without plotting. Each export gets a unique directory.

`show=False, progress=None` keeps calculation/fitting calls quiet. Simulation and fitting alone do not import the notebook plotting dependencies. All these functions use the same underlying model and fitting implementation as the CLI.

## Circuit inputs

Generate the full default configuration:

```bash
python -m pauli_zne init config.json
python -m pauli_zne run --config config.json --output runs/custom
```

`init` and `run` refuse to overwrite existing files/nonempty result directories.

```json
{
  "circuit": {
    "N": 6,
    "L": 2,
    "phi": 0.19634954084936207,
    "O": "IIZIZI"
  },
  "s_grid": [1, 2, 4, 8]
}
```

- `N`: even number of qubits, at least 2.
- `L`: number of full brickwall steps. Each step applies the even bonds and then the odd bonds in forward state evolution.
- `phi`: radians, with two-qubit unitary `exp[-i phi (XX+YY+ZZ)/2]`.
- `O`: an N-character Pauli string, qubit 0 first. The example is Z2 Z4. Identity factors are `I`. The current interface supports a single Pauli string, not an arbitrary weighted observable sum.
- Initial state: a product of singlets on (0,1), (2,3), … . This is fixed in this version.
- `s_grid`: distinct positive finite noise scales. Default `[1,2,4,8]`. The code calculates s=0 automatically for normalization and validation and **never fits that point**. Scales are sorted. Shot lists use this sorted order; maps keyed by s avoid ambiguity.

Circuit and grid overrides are also available:

```bash
python -m pauli_zne run --config config.json --N 4 --L 3 --phi 0.2 --O IZZI --s-grid 1,2,3,4,6,8 --output runs/override
```

When changing N, supply an O of matching length. JSON is strict: misspelled configuration keys fail validation.

## Four CNOT-layer noise models

Each Heisenberg gate is decomposed into three CNOTs and four blocks of one-qubit rotations. The CNOT directions are right→left, left→right, right→left. Combined with bond parity there are four **types**, but six CNOT-layer occurrences per full even/odd step:

|Name|Bonds|Control direction|Occurrences per step|
|:--|:--|:--|--:|
|`even_lr`|(0,1), (2,3), …|left→right|1|
|`even_rl`|(0,1), (2,3), …|right→left|2|
|`odd_lr`|(1,2), (3,4), …|left→right|1|
|`odd_rl`|(1,2), (3,4), …|right→left|2|

The two occurrences of a given type share that type's rates within a step. Step-dependent arrays can change them across L. An empty matching (the odd layer for N=2) applies neither gates nor noise.

The model is a global Pauli-Lindblad channel:

`E_s = exp[s * sum_P lambda_P (P rho P - rho)]`.

It supports all on-site X/Y/Z generators and all nine nearest-neighbor products XX, XY, …, ZZ. A Pauli observable is multiplied by `exp[-2 s sum(lambda_P for anticommuting P)]`. A channel acts once per CNOT layer on the **whole chain**, including idle qubits and nearest-neighbor generators across matching boundaries; it is not separately repeated for every CNOT gate. Rates must be finite and nonnegative.

```json
{
  "noise": {
    "placement": "before_cnot",
    "single": [0.001, 0.001, 0.0001],
    "pair": 0.0,
    "channels": {
      "even_lr": {"single": [0.001, 0.002, 0.0001]},
      "even_rl": {"single": [0.002, 0.001, 0.0001]},
      "odd_lr":  {"single": [0.001, 0.001, 0.0002]},
      "odd_rl":  {"single": [0.003, 0.001, 0.0001]}
    }
  }
}
```

`single` axes are X,Y,Z. `pair` axes are the left and right X,Y,Z generators. Scalars fill every entry; `[lambda_X,lambda_Y,lambda_Z]` fills all sites. The default shared arrays broadcast to:

- singles: `(L,4,N,3)`;
- pairs: `(L,4,N-1,3,3)`.

Thus `(N,3)` shares a site-dependent model across all four types, `(4,N,3)` separates the types, and `(L,4,N,3)` also separates steps. Pair rates follow the analogous shapes. Per-channel overrides broadcast to `(L,N,3)` and `(L,N-1,3,3)`.

`placement` is explicitly in **forward state order**. `before_cnot` matches the existing CNOT-resolved solver. Choose `after_cnot` for noise after every CNOT layer. Backward observable evolution reverses this ordering. These generally give different noisy circuits.

For larger vectors, supply an NPZ with `single_rates` and `pair_rates`:

```python
import numpy as np
N, L = 32, 12
single = np.broadcast_to([.001, .001, .0001], (L, 4, N, 3)).copy()
pair = np.zeros((L, 4, N-1, 3, 3))
single[:, 3, :, 0] *= 2  # stronger X generator in odd_rl
np.savez('rates.npz', single_rates=single, pair_rates=pair)
```

Set `"noise": {"rates_file": "rates.npz"}`. Shared defaults are replaced by the NPZ, then any named channel overrides are applied. Alternatively, a channel can use `{"file":"odd_rates.npz"}` with that channel's two arrays. All input paths resolve relative to the configuration file. Resolved arrays are saved in each run's `inputs.npz`.

This version supports Pauli-Lindblad noise, not arbitrary non-Pauli channels such as amplitude damping.

## More complex noise models

The current backend supports **Pauli-Lindblad channels** built from any combination of on-site X/Y/Z generators and nearest-neighbor XX, XY, …, ZZ generators. Their rates can vary by qubit, bond, CNOT-layer type, and full circuit step. All rates must be **real, finite and nonnegative**.

The examples below are alternatives for the **configuration cell**, not extra computation cells. After changing the physical noise model, rerun the MPS, fitting and analysis cells. Changing shot counts alone still requires only fitting and analysis.

### Defaults and channel overrides

`noise["single"]` and `noise["pair"]` supply shared defaults. A named channel's `"single"` or `"pair"` **replaces that entire field for that channel**; it is not added to the default. An omitted field inherits the default. For example, overriding `"single"` alone leaves the default pair noise in place.

Channel order for arrays is **even_lr, even_rl, odd_lr, odd_rl**. The same model is reused for repeated occurrences of a type within a full step. Every selected channel acts globally, once per matching CNOT layer; these arrays do not automatically restrict noise to the active CNOT bonds.

### Example: sampled site disorder plus correlated pair noise

This samples independent on-site rates for each channel and qubit, then adds selected pair generators. Sampling happens **once**, with a fixed seed. All s points use the same sampled arrays; s multiplies the rates. Do not resample the physical model at each s or in each measurement-shot trial.

```python
import numpy as np
from pauli_zne.workflow import define_experiment

N, L = 6, 2
rng = np.random.default_rng(7)
channel = {"even_lr": 0, "even_rl": 1, "odd_lr": 2, "odd_rl": 3}

# Axes: [channel, qubit, X/Y/Z].
mean = np.array([0.001, 0.001, 0.0001])
single = np.maximum(
    rng.normal(mean, 0.10 * mean, size=(4, N, 3)),
    0.0,  # Gaussian draws are clipped to preserve nonnegative rates.
)

# Axes: [channel, bond-left-qubit, left X/Y/Z, right X/Y/Z].
pair = np.zeros((4, N - 1, 3, 3))
pair[:, :, 2, 2] = 0.0002                    # ZZ noise on every adjacent pair
pair[channel["odd_rl"], 1, 0, 1] = 0.00005    # X_1 Y_2, only in odd_rl

# Optional: sample every pair-generator rate independently instead.
# pair = np.maximum(rng.normal(0.0001, 0.00001, (4, N-1, 3, 3)), 0.0)

experiment = define_experiment(
    N=N, L=L, phi=np.pi / 16, O=None,  # centered ZZ observable
    noise={
        "placement": "before_cnot",  # or "after_cnot"
        "single": single.tolist(),
        "pair": pair.tolist(),
        "channels": {},              # no overrides replacing these samples
    },
)
```

`pair[c,q,a,b]` is the rate of the dissipator generated by $\sigma_a^{(q)}\sigma_b^{(q+1)}$. Thus the nonzero XY entry above describes **correlated Pauli noise**, not a coherent XY gate. The 3×3 array is a table of nine generator rates, not a covariance matrix: XY and YX are different generators, and the table need not be symmetric.

Clipping a Gaussian changes its distribution near zero. A positive distribution such as a lognormal is another choice; an untruncated Gaussian can produce invalid negative rates.

### Array shapes and step-dependent noise

| Scope | `single` shape | `pair` shape |
|:--|:--|:--|
| Same spatial model for every channel and step | `(N,3)` | `(N-1,3,3)` |
| Different model for each channel type | `(4,N,3)` | `(4,N-1,3,3)` |
| Also different for each full circuit step | `(L,4,N,3)` | `(L,4,N-1,3,3)` |

A scalar broadcasts to every rate in its field. A single `[lambda_X,lambda_Y,lambda_Z]` vector broadcasts to all qubits and channels. For a **named channel override**, omit the channel axis: use `(N,3)` or `(L,N,3)` for singles and `(N-1,3,3)` or `(L,N-1,3,3)` for pairs.

For example, starting with the sampled arrays above, make the noise grow by 5% per full forward circuit step:

```python
step_multiplier = 1.0 + 0.05 * np.arange(L)
single_by_step = single[None, ...] * step_multiplier[:, None, None, None]
pair_by_step = pair[None, ...] * step_multiplier[:, None, None, None, None]

experiment = define_experiment(
    N=N, L=L, phi=np.pi / 16, O=None,
    noise={
        "single": single_by_step.tolist(),
        "pair": pair_by_step.tolist(),
        "channels": {},
    },
)
```

Step 0 is the first full step in forward state evolution. The same step-dependent schedule is held fixed across the s grid. This interface does not separately parameterize the two occurrences of the same CNOT-layer type within one step.

### Load calibrated or pre-sampled rates from a file

Save arrays with the same shapes under the keys `single_rates` and `pair_rates`:

```python
np.savez("noise_rates.npz", single_rates=single_by_step, pair_rates=pair_by_step)
experiment = define_experiment(
    N=N, L=L, phi=np.pi / 16, O=None,
    noise={"rates_file": "noise_rates.npz", "channels": {}},
    rates_directory=".",
)
```

With the Python API, `rates_directory` specifies the base directory for noise files; it defaults to the working directory. JSON configurations resolve paths relative to the configuration file. A whole-model NPZ replaces the shared defaults, and any named channel overrides are applied afterward. A named channel can instead use `{"file": "odd_rl_rates.npz"}` containing just its two rate arrays, without the channel axis. Each exported run saves the fully resolved arrays in `inputs.npz`.

### Channels that need a backend extension

Arbitrary Kraus channels, amplitude damping, coherent over-rotations, non-nearest-neighbor/multi-qubit generators, and cross terms between different Lindblad generators are **not supported by these rate fields**. They cannot be added by supplying complex-valued or negative lambda entries. They require extending both the MPS propagation and the independent reference calculation; the unsigned path/Q definition must also be reviewed for that extension.


## MPS curves and Q definitions

Configure the requested curves separately from the reference sweep:

```json
{
  "mps": {
    "bond_dims": [32, 64, 128, 256],
    "cutoff": 1e-12,
    "svd": "randomized",
    "power_iterations": 2,
    "oversample": 24,
    "seed": 7
  }
}
```

`svd` can also be `full`. The cutoff is relative to the largest singular value at a split. Bond dimensions refer to the new **one-qubit, physical-dimension-4 MPS**. They are not directly equivalent in cost or approximation to the earlier two-qubit-blocked MPS caps. The summed discarded-relative-weight diagnostic is not a bound on observable error.

For each K and scale, compute signed path sum S(s)=O_K(s) and unsigned path sum A(s). The unsigned computation takes absolute values **at each elementary rotation/CNOT transfer and at the singlet terminal functional**, before paths are added. Taking the absolute value of a contracted Heisenberg gate would give a different Q definition.

- `R_K(s) = S(s)/S(0)`.
- `Q+(s) = [A(s)+S(s)]/[A(0)+S(0)]`.
- `Q−(s) = [A(s)−S(s)]/[A(0)−S(0)]`.

Outputs include Q average, Q difference, and the stable contrast `S(s)−[A(s)/A(0)] S(0)`, proportional to Q difference. The Q fit rescales this contrast **before** solving, avoiding loss of the informative difference when Q+ and Q− round to nearly the same value. Large unsigned sums are stored logarithmically. No interpolation or polynomial approximation is used.

Near-zero S(0) makes the ratio undefined. Missing/zero unsigned mass, a missing sign sector, or a rank-deficient Q basis is explicitly marked rather than divided by zero. An insufficient bond approximation can also produce invalid unsigned mass. Numerical failures are recorded in `failure.json` and abort that run; partial contraction checkpoints remain available.

## Reference selection and convergence

```json
{
  "reference": {
    "statevector_max_n": 20,
    "density_max_n": 8,
    "memory_mb": 512,
    "bond_dims": [128, 256, 512],
    "atol": 1e-8,
    "rtol": 1e-4,
    "consecutive": 2
  }
}
```

- O(0): independent forward statevector if N and the estimated array memory fit the limits; otherwise the selected signed MPS reference.
- O(s>0): exact forward **density matrix** for small systems. A single statevector cannot represent general noisy mixed states exactly.
- Otherwise: signed MPS at increasing reference bonds. Every scale, including zero, must satisfy `abs(new-old) <= atol + rtol*abs(new)` for the requested number of consecutive increments. With `consecutive=2`, at least three caps are needed.
- Randomized references are also rerun with a different seed and an additional power iteration; this must agree within the same tolerance.
- If the budget is exhausted, the largest computed bond is retained and marked **not converged**. Agreement is an empirical criterion, not an exactness certificate. No statevector error claim is made for a finite-bond reference.

The memory setting guards exact reference selection, not total process memory. Forcing large bond caps can still consume substantial RAM. Saved reference checks expose every change; a large relative error near an O(s) zero should be read together with the absolute error.

## Fitting and per-point shot noise

All methods use the same observed s points and equal least-squares weights:

|Method|Fit|Minimum points|
|:--|:--|--:|
|Ratio|one amplitude × R_K(s)|1|
|Direct Q|two amplitudes × Q average and rescaled contrast|2, with full rank|
|Single exponential|a exp(−b s)|2|
|Dual exponential|a1 exp(−b1 s)+a2 exp(−b2 s)|4|

The extrapolated value is the intercept at s=0. Q curve values are fixed classical inputs. Exponential fits use continuous variable projection, multiple starting points, and coarse rate search for the dual fit. Default rates are bounded to `[0.001,5]` and signed component amplitudes to `[-1,1]`; both bounds are configurable. The sum of dual amplitudes is not clipped to the physical range. Fits at bounds and optimizer non-success are saved. Dual exponential is underdetermined with fewer than four points and is marked unavailable.

```json
{
  "fit": {
    "rate_bounds": [0.001, 5.0],
    "amplitude_bound": 1.0,
    "trials": 1000,
    "seed": 7,
    "shot_scenarios": {
      "equal": 1000000,
      "unequal": [10000, 100000, 1000000, 10000000],
      "only_1_2": {"1": 100000, "2": 1000000, "4": null, "8": null}
    }
  }
}
```

Every run includes a noiseless fit. `shot_scenarios: {}` disables Monte Carlo. A scalar uses the same number of shots at every point; a list follows the sorted grid; a map must cover each scale exactly once. `null` keeps that mean exact. Exact means are still equal-weight observations, **not hard constraints**.

Shot noise is ordinary independent Pauli ±1 binomial noise added only to O(s), never to the classical Q or ratio. All methods share the same draws in a scenario. This is not a probability-error-amplification protocol. Reports include RMSE, median absolute relative error, total shots, parameter-boundary fractions and optimizer failures. For a near-zero target, relative errors are null and absolute errors remain available.

Refit saved curves with new shot budgets without repeating MPS calculations:

```bash
python -m pauli_zne fit --run runs/small --fit-config examples/refit.json --output runs/refit
```

`--fit-config` contains only the contents of the `fit` object. Optionally provide `--observations measured.csv` with columns `s,O`, exactly one row per saved positive scale. Comparisons then use the saved simulation target, not experimentally known truth. Any additional simulated shot noise treats supplied means as plug-in probabilities.

## Outputs and reproducibility

Full simulation runs contain resolved config, normalized noise arrays, Python/library versions and source hashes, raw MPS contractions, per-bond O/R/Q curves, reference selection/checks, noiseless fits, shot summaries, per-trial exponential diagnostics, and a Markdown report. Intermediate contractions and completed curve caps are checkpointed. A run does not automatically resume or mix old files. Refit folders contain new fit results and a `refit_inputs.json` pointing to the original curve/input artifacts; they do not duplicate those artifacts.

`runs/`, caches, build outputs and virtual environments are ignored by Git. Keep simulation outputs outside version control unless a particular validated example is deliberately selected.

Tests cover the KAK unitary, both noise orderings against independent dense evolution, site/pair and four-channel rate schedules, elementary unsigned paths against a dense Pauli calculation, ratio/Q edge cases, fitting recovery, selective shot noise, and complete run/refit workflows. Large-N convergence must still be established for each circuit.
