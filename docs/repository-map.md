# Repository Map

[Back to README](../README.md)

## Core Samplers

```text
Pauli_path_Heis_mixture_trigger.py
Pauli_path_Heis_full_layer_sampling_restricted.py
benchmark_q_curve_convergence.py
final_dp_mom_convergence_sweep.py
qpm_numba_utils.py
```

## MPS Comparison

```text
pauli_mps_solver.py
compute_mps_zizj_vs_s.py
sweep_mps_zizj_chi_vs_s.py
pauli_mps_solver_user_guide.md
```

## Final Fitting

```text
fit_zne_compare.py
```

## Diagnostics

```text
diagnose_q_batch_heavytails.py
compare_final_dp_vs_single_pauli_q.py
sweep_final_dp_vs_single_convergence.py
benchmark_q_batchsize_convergence.py
```

## Notebook Tutorial

```text
notebooks/Final_Layer_DP_Usage.ipynb
```

## Current Preferred Defaults

```text
noise_model       independent
noise_placement   layer
q_summary         pointwise_mom
s_grid            1,2,4,8
fit models        dual_exp, gate_qmom, layer_qmom
```

## Symmetry-resolved tensor methods

- `tensor_zne.py`: shared MPS/PEPS ratio and Q CLI/API, residual correction and CSV exports.
- `parity_pauli_mps.py`, `u1_pauli_mps.py`: Z2 × Z2 implementation and shared charge-block operations.
- `mps_residual.py`: discarded-evolution residual telescope and backward guides.
- `optimized_pauli_svd.py`, `contiguous_pauli_kernel.py`: CPU/Metal SVD products and Numba transfer.
- `pauli_peps.py`, `pauli_peps_symmetry.py`: Pauli-basis quimb PEPS and symmray blocks.
- `peps_boundary.py`, `peps_boundary_residual.py`: CPU boundary contractions and streamed residuals.
- `peps_metal_layer.py`, `peps_gpu_boundary.py`, `peps_gpu_decomp.py`: optional MLX/Metal operations.
- `lattice_circuit.py`, `causal_pauli_network.py`: circuit definitions and independent exact-network reference.
- `requirements-*.txt`, `tests/`: install profiles and regression tests.

See [installation and usage](tensor-network-methods.md).

- `cnot_parity.py`: graph-wide CNOT-layer noise compiler and transported charge maps.
- `cnot_parity_mps.py`: canonical site-wise parity MPS and checkpointed residuals.
- `cnot_parity_peps.py`: symmetry-resolved elementary PEPS updates and graph routing.
- [CNOT noise usage](cnot-parity-methods.md): schedules, longer-range edges, and Q compatibility.

## CPU and Apple Silicon optimizations

- `cpu_numba_kernels.py`: strict CPU sector gathering, selection, factor reconstruction,
  and small-block PEPS scaling/packing; `PAULI_CPU_NUMBA=0` disables the new policy.
- `mps_contractions.py`, `mps_qr.py`: CPU matrix-product routing and cached QR workspaces.
- `peps_hybrid_decomp.py`: selective CPU LAPACK factors within a Metal boundary contraction.
- `benchmarks/`, `docs/apple-silicon/`: benchmark drivers and intentionally published measurements.
- `apple_*experiment.py`, `peps_*experiment.py`, `peps_message_refinement.py`,
  `peps_split_layout.py`, and `apple_metal_ragged.py`: research prototypes; not automatic defaults.

See the [CPU Numba report](cpu-numba-results.md) and
[hybrid results](apple-hybrid-refinement-results.md) for enabled policies and limitations.
