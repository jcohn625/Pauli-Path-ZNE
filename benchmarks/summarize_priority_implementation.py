"""Publish this priority pass's raw measurements and final source provenance."""
import hashlib,json,platform,sys
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
root=Path(__file__).resolve().parents[1]
raw=root/'results/priority_implementation'
files={
 'row':'row.json','guided_projector_initial':'guided.json',
 'guided_projector_and_rtm':'guided_rtm.json','ready_batches_warm':'ready_batches_warm.json',
 'blocked_locally_sorted_initial':'blocked.json','blocked_unsorted':'blocked_unsorted.json',
 'blocked_refined':'blocked_refined.json',
}
sources=[
 'peps_boundary_residual.py','tensor_zne.py','peps_environment_experiment.py',
 'apple_jacobi_experiment.py','apple_blocked_jacobi_experiment.py',
 'benchmarks/run_boundary_priorities.py','benchmarks/run_blocked_jacobi.py',
 'benchmarks/profile_ready_boundary.py','benchmarks/summarize_priority_implementation.py',
 'tests/test_peps_row_residual.py','tests/test_peps_environment_experiment.py',
 'tests/test_apple_blocked_jacobi_experiment.py',
]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data=dict(schema_version=1,archived_at_utc=datetime.now(timezone.utc).isoformat(),
 hardware='Apple M4 Pro, 24 GiB unified memory',python=sys.version,platform=platform.platform(),
 packages={name:version(name) for name in ('numpy','scipy','numba','quimb','symmray','mlx')},
 blas_threads=1,timed_repetitions=3,
 notes=[
  'No concurrent numerical benchmarks; one warm-up before three timing samples.',
  'PEPS exact references contract the fixed spatially truncated PEPS, not the exact circuit.',
  'Actual-ready batch profile is warmed and explicitly synchronized; not an uninstrumented speed comparison.',
  'SVD batch-16 repeats up to eight captured matrices and is not a measured ready solver batch.',
  'Final source hashes describe current scripts. The initial locally-sorted GPU run predates the unsorted local rotation correction.',
  'The initial projector run predates inclusion of RTM in the guided benchmark.',
 ],
 final_source_sha256={s:digest(root/s) for s in sources},
 captured_matrices_sha256=digest(root/'results/exploration_20261013/adaptive_real.npz'),
 data={k:json.loads((raw/v).read_text()) for k,v in files.items()})
out=root/'docs/apple-silicon/priority-implementation-20261013.json'
out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
print(out)
