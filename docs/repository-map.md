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
