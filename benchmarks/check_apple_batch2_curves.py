"""Compare O, U, ratio, Q and residuals to the pre-batch-2 module snapshot."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
import cnot_parity_mps as mps
import peps_metal_layer as peps
from tensor_zne import curves, evaluate_mps, evaluate_peps


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, args.baseline/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old_mps = load('batch2_old_mps', 'cnot_parity_mps.py')
old_peps = load('batch2_old_peps', 'peps_metal_layer.py')
targets = [(mps.FrameMPS, old_mps.FrameMPS, name) for name in ('_left_qr', '_right_qr')]
targets.append((peps.LayerEngine, old_peps.LayerEngine, 'scale_many'))
current = [getattr(obj, name) for obj, _, name in targets]


def select(before):
    for (obj, old, name), new in zip(targets, current):
        setattr(obj, name, getattr(old, name) if before else new)


results = []
try:
    for method, evaluator, settings in [
        ('mps', evaluate_mps, dict(n=6, steps=2, sites=(1, 3), cap=4, guide_cap=8)),
        ('peps', evaluate_peps, dict(nx=2, ny=3, steps=2, sites=(1, 3), cap=4,
                                   boundary_cap=4, environment_cap=8,
                                   gate_backend='batched-cpu'))]:
        settings.update(noise_placement='after_cnot', pair_rates=.001, residual=True)
        values = []
        for before in (True, False):
            select(before)
            values.append(curves(evaluator, [0., 1., 3.], **settings))
        differences = []
        for old, new in zip(values[0]['rows'], values[1]['rows']):
            for key in old:
                if old[key] is None or new[key] is None:
                    assert old[key] is new[key], (method, key)
                else:
                    differences.append(abs(old[key]-new[key]))
        residual_differences = []
        for old, new in zip(values[0]['raw'], values[1]['raw']):
            for kind in ('signed', 'unsigned'):
                residual_differences.append(abs(old[kind]['correction']-new[kind]['correction']))
                details = new[kind]['details']
                if method == 'mps':
                    assert details['charge_leakage'] == 0.
        maximum = max(differences + residual_differences)
        assert maximum < 1e-11, (method, maximum)
        results.append(dict(method=method, settings=settings, max_abs_difference=maximum,
                            max_residual_difference=max(residual_differences),
                            before=values[0]['rows'], after=values[1]['rows']))
        print(method, maximum, flush=True)
finally:
    select(False)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2)+'\n')
