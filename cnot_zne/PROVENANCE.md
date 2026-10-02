# Provenance and scope

This package was prepared as a clean standalone implementation for the Pauli-Path-ZNE project. It follows the circuit convention and specialized three-CNOT decomposition in the local project's `cnot_lindblad_numba.py` and the gate-resolved ordering in `pauli_mps_solver.py`. Those original files were inspected but not modified or copied wholesale.

The new MPS implementation uses one physical-dimension-4 site per qubit, stabilized randomized range finding, SciPy gesvd, explicit unsigned elementary path transfers, and independent forward exact references. It does not depend on the original repository checkout. The prior scratch benchmark's blocked-MPS bond dimensions, coarse noise placement and numerical results should not be conflated with this implementation.

The package itself performs no Git operations; preparing or publishing a branch is a separate step. Repository ownership and licensing should follow the destination project's policy; this folder does not introduce a new license grant.
