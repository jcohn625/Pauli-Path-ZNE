"""Publish final benchmark data; retain pre-fix numerical evidence separately."""
import hashlib,json,platform,sys
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
root=Path(__file__).resolve().parents[1];raw=root/'results/refinement_hybrid'
source_files=[
 'peps_gpu_decomp.py','peps_hybrid_decomp.py','peps_gpu_boundary.py','pauli_peps.py',
 'tensor_zne.py','peps_boundary_residual.py','peps_environment_experiment.py',
 'peps_message_refinement.py','benchmarks/run_hybrid_boundary.py',
 'benchmarks/run_hybrid_workflow.py','benchmarks/run_captured_decompositions.py',
 'benchmarks/run_message_refinement.py','benchmarks/summarize_hybrid_refinement.py',
 'tests/test_peps_hybrid_decomp.py','tests/test_peps_message_refinement.py']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((raw/name).read_text())
data=dict(schema_version=1,archived_at_utc=datetime.now(timezone.utc).isoformat(),
 hardware='Apple M4 Pro, 24 GiB unified memory',python=sys.version,platform=platform.platform(),
 packages={p:version(p) for p in ('numpy','scipy','numba','quimb','symmray','mlx')},
 source_sha256={s:digest(root/s) for s in source_files},
 captured_input_sha256=digest(raw/'boundary.npz'),
 notes=[
  'Final numerical timings use serial benchmark runs, one warm-up and three timed repetitions, one CPU BLAS thread.',
  'The final large-case boundary timings replace the overlapping preliminary large-case/workflow measurements.',
  'Boundary timing reuses an evolved PEPS and resident GPU input; workflow timing includes evolution and conversion.',
  'Pre-fix QR replay records demonstrate the defect; final source hashes describe the corrected implementation.',
  'The captured QR/SVD inputs came from the pre-fix Metal boundary trajectory, rather than a synthetic shape substitution.',
  'Message references are exact contractions of the spatially truncated PEPS, not exact circuit expectations.',
 ],
 boundary=read('boundary_stable.json')[:3]+read('boundary_large_final.json'),
 workflow=read('workflow_final.json'),decompositions_before_fix=read('decompositions.json'),
 decompositions_after_fix=read('decompositions_stable.json'),messages=read('messages.json'))
out=root/'docs/apple-silicon/hybrid-refinement-20261013.json'
out.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');print(out)
