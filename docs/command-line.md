# Command-Line Appendix

[Back to README](../README.md)

The notebook API is the recommended interface for exploration. The scripts can
also be run directly for long resumable jobs.

## Grouped Q Data

```bash
MPLCONFIGDIR=.mplconfig python final_dp_mom_convergence_sweep.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-grid 1,2,4,8 \
  --total-samples 1e6,1e7 \
  --n-groups 16 \
  --methods gate,layer \
  --out-prefix final_dp_mom_convergence_N20_L10_Z6Z10_phi02
```

## MPS Observable Data

```bash
python sweep_mps_zizj_chi_vs_s.py \
  --n-qubits 20 \
  --n-steps 10 \
  --q1 6 \
  --q2 10 \
  --phi 0.2 \
  --s-values 0,1,2,4,8 \
  --chi-values 350 \
  --noise-model independent \
  --noise-placement layer \
  --out-csv mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --resume
```

## Final ZNE Comparison

```bash
MPLCONFIGDIR=.mplconfig python fit_zne_compare.py \
  --q-groups final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --observables mps_N20_L10_Z6Z10_phi02_chi350.csv \
  --s-grid 1,2,4,8 \
  --s-fit-q 1,2,4,8 \
  --s-fit-o 1,2,4,8 \
  --shots 100000 \
  --n-trials 500 \
  --q-summary pointwise_mom \
  --out-prefix zne_fit_N20_L10_Z6Z10_phi02_100k
```

## Heavy-Tail Diagnostic

```bash
MPLCONFIGDIR=.mplconfig python diagnose_q_batch_heavytails.py \
  --batches final_dp_mom_convergence_N20_L10_Z6Z10_phi02_groups.csv \
  --samples-per-batch 625000 \
  --out-prefix taildiag_N20_L10_Z6Z10_phi02
```
