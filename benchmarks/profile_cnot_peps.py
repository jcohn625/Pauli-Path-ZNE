"""Profile a warmed correlated-noise PEPS evaluation (CPU)."""
import cProfile, pstats, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from tensor_zne import evaluate_peps
settings=dict(nx=4,ny=4,steps=1,sites=(5,10),cap=4,boundary_cap=4,
              environment_cap=8,noise_placement='after_cnot',pair_rates=.0001)
evaluate_peps(**settings)
p=cProfile.Profile();p.enable()
for _ in range(3):evaluate_peps(**settings)
p.disable();out=root/'results';out.mkdir(exist_ok=True)
p.dump_stats(str(out/'cnot_peps_profile.prof'))
with (out/'cnot_peps_profile.txt').open('w') as f:
 s=pstats.Stats(p,stream=f);s.strip_dirs().sort_stats('cumulative').print_stats(35)
 s.sort_stats('tottime').print_stats(25)
print((out/'cnot_peps_profile.txt').read_text())
