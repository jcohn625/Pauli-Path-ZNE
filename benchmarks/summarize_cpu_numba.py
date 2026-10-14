"""Archive measured CPU Numba timings and source hashes."""
import hashlib,json,platform
from importlib.metadata import version
from pathlib import Path
root=Path(__file__).resolve().parents[1]
files={'micro':'micro_final.json','mps_ablation':'mps_ablation.json','mps_policy':'mps_policy.json',
       'peps_initial':'peps.json','peps_prepared':'peps_prepared.json','peps_policy':'peps_policy.json','peps_quimb':'peps_quimb.json'}
data={k:json.loads((root/'results/cpu_numba'/v).read_text()) for k,v in files.items()}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=['cpu_numba_kernels.py','cnot_parity_mps.py','peps_metal_layer.py',
         'tests/test_cpu_numba_kernels.py','tests/test_peps_scaling.py','benchmarks/benchmark_cpu_numba.py']
result=dict(date='2026-10-14',machine='Apple M4 Pro, 24 GiB',platform=platform.platform(),
            versions={n:version(n) for n in ['numpy','scipy','numba','quimb','symmray']},
            numerical_threads=1,policy=['scale','gather','select','scatter'],
            source_sha256={p:digest(root/p) for p in sources},
            baseline_sha256={p:digest(Path('/tmp/pauli-numba-baseline')/p)
                             for p in ['cnot_parity_mps.py','peps_metal_layer.py']},results=data)
path=root/'docs/apple-silicon/cpu-numba-20261014.json'
path.write_text(json.dumps(result,indent=2)+'\n')
for section,rows in data.items():
 for r in rows:print(section,r['label'],r['seconds'])
print(path)
