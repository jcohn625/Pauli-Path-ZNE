import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/pauli_zne_numba_cache')
import numpy as np
import pytest
from lattice_circuit import square
from pauli_peps import evaluate_peps


@pytest.mark.parametrize('unsigned',[False,True])
@pytest.mark.parametrize('scale',[0.,1.])
def test_cpu_batched_matches_sequential_without_truncation(unsigned,scale):
    kw=dict(nx=2,ny=2,pairs=[(0,1),(2,3)],n_steps=2,phi=.19,
            lam_xyz=(.002,.004,.008),symmetry='z2z2',bond_cap=256,
            boundary_cap=1024,unsigned=unsigned,scale=scale,cutoff=0.)
    a=evaluate_peps('ZZII',square(2,2),gate_backend='quimb',**kw)
    b=evaluate_peps('ZZII',square(2,2),gate_backend='batched-cpu',**kw)
    assert b['value']==pytest.approx(a['value'],rel=2e-11,abs=2e-11)
    assert b['layer_stats']['gates']==a['included_gates']
    assert b['symmetry_stats']['tensor_charge_checks_passed']


def test_batched_real_rectangle_and_capture():
    captures=[]
    kw=dict(nx=2,ny=3,pairs=[(0,1),(2,3),(4,5)],n_steps=2,phi=.19,
            lam_xyz=(.002,.004,.008),symmetry='z2z2',bond_cap=8,boundary_cap=16)
    a=evaluate_peps('IIZZII',square(2,3),gate_backend='quimb',**kw)
    b=evaluate_peps('IIZZII',square(2,3),gate_backend='batched-cpu',
                    capture_layer=lambda *args:captures.append(len(args[2])),**kw)
    assert b['value']==pytest.approx(a['value'],rel=2e-10,abs=2e-10)
    assert captures==[3]
    c=evaluate_peps('IIZZII',square(2,3),gate_backend='quimb',
                    capture_layer=lambda *args:None,**kw)
    assert c['value']==pytest.approx(a['value'],rel=2e-10,abs=2e-10)


@pytest.mark.skipif(os.environ.get('TEST_PEPS_METAL')!='1',reason='requires explicit Metal access')
def test_metal_exact_small_observable():
    kw=dict(nx=2,ny=2,pairs=[(0,1),(2,3)],n_steps=1,phi=.19,
            lam_xyz=(.002,.004,.008),symmetry='z2z2',bond_cap=16,boundary_cap=32)
    a=evaluate_peps('ZZII',square(2,2),gate_backend='quimb',**kw)['value']
    b=evaluate_peps('ZZII',square(2,2),gate_backend='metal',**kw)['value']
    assert b==pytest.approx(a,rel=1e-5,abs=1e-7)
