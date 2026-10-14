"""Portable ratio/Q workflows for symmetry-resolved Pauli MPS and PEPS."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import numpy as np


@contextmanager
def peps_contractions():
    from autoray import get_lib_fn, register_function
    from cotengra.contract import default_implementation
    from pauli_peps import _safe_einsum, _safe_tensordot
    from peps_boundary import boundary_backend
    original = get_lib_fn('numpy', 'tensordot')
    register_function('numpy', 'tensordot', _safe_tensordot)
    try:
        with default_implementation((_safe_einsum, _safe_tensordot)), boundary_backend('blas'):
            yield
    finally:
        register_function('numpy', 'tensordot', original)


def evaluate_mps(*, n=6, steps=3, sites=(2, 4), scale=1., lam=(.001, .002, .003),
                 phi=np.pi/16, cap=16, guide_cap=32, terminal=2, residual=True,
                 unsigned=False, backend='cpu', noise_placement='heisenberg',
                 single_rates=None, pair_rates=0., noise_edges=None,
                 guide_cache_mb=None,store_contributions=True,contribution_callback=None,
                 noise_strategy='local',noise_mpo_max_bond=64):
    """1D chain: historical Heisenberg noise or full after-CX-layer noise.

    after_cnot uses a single-site parity MPS and corrects every routed truncation;
    terminal is only used by the historical Heisenberg-noise path.
    """
    if noise_placement not in ('heisenberg','after_cnot'):
        raise ValueError('MPS noise_placement must be heisenberg or after_cnot')
    if noise_placement == 'after_cnot':
        from cnot_parity_mps import evaluate
        return evaluate(n=n,steps=steps,sites=sites,phi=phi,scale=scale,
            singles=lam if single_rates is None else single_rates,pair_rates=pair_rates,
            edges=noise_edges,cap=cap,guide_cap=guide_cap,residual=residual,
            unsigned=unsigned,backend=backend,guide_cache_mb=guide_cache_mb,
            store_contributions=store_contributions,contribution_callback=contribution_callback,
            noise_strategy=noise_strategy,noise_mpo_max_bond=noise_mpo_max_bond)
    if guide_cache_mb is not None or not store_contributions or contribution_callback is not None or noise_strategy!='local':
        raise ValueError('guide memory/diagnostic options require after_cnot')
    if single_rates is not None or noise_edges is not None or np.any(pair_rates):
        raise ValueError('correlated/scheduled rates require noise_placement="after_cnot"')
    import pauli_mps_solver as solver
    from parity_pauli_mps import run
    rates = np.asarray(lam, dtype=float)
    if rates.shape != (3,) or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError('lam must contain three finite nonnegative rates')
    if not np.isfinite(scale) or scale < 0 or steps < 0 or not np.isfinite(phi):
        raise ValueError('scale and steps must be nonnegative; scale and phi must be finite')
    eta = solver.pauli_diag_factors_from_lambda(np.tile(rates * scale, (n, 1)), 'independent')
    kw = dict(terminal=terminal, phi=phi, unsigned=unsigned, backend=backend)
    if residual:
        from mps_residual import run as corrected
        result = corrected(solver, n, steps, *sites, eta, cap, guide_cap, **kw)
        return dict(approximate=result['base'], correction=result['delta'],
                    corrected=result['corrected'], details=result)
    value, stats = run(solver, n, steps, *sites, eta, cap, **kw)
    return dict(approximate=value, correction=0., corrected=value, details=stats)


def evaluate_peps(*, nx=2, ny=3, steps=2, sites=(0, 1), scale=1.,
                  lam=(.001, .002, .003), phi=np.pi/16, cap=8, boundary_cap=8,
                  environment_cap=16, residual=True, unsigned=False, residual_granularity='row',
                  gate_backend='quimb', boundary_backend='blas', jacobi_sweeps=32,
                  noise_placement='gate', single_rates=None, pair_rates=0., noise_edges=None):
    """Open rectangle; residuals correct boundary truncation of the evolved PEPS.

    metal-hybrid is opt-in: Metal FP32 contractions with selective CPU LAPACK
    QR/SVD. The default blas path retains CPU FP64.
    """
    if residual_granularity not in ('row','bond'):
        raise ValueError('residual_granularity must be row or bond')
    from lattice_circuit import square
    from pauli_peps import evaluate_peps as evolve
    if nx < 2 or ny < 2 or nx * ny % 2:
        raise ValueError('use a rectangle at least 2x2 with an even number of sites')
    if len(sites) != 2 or len(set(sites)) != 2 or any(q < 0 or q >= nx * ny for q in sites):
        raise ValueError('use two distinct in-range observable sites')
    lattice = square(nx, ny)
    observable = ['I'] * lattice.n_sites
    for q in sites:
        observable[q] = 'Z'
    if noise_placement not in ('gate','layer','after_cnot'):
        raise ValueError('PEPS noise_placement must be gate, layer, or after_cnot')
    cnot_noise=None
    if noise_placement=='after_cnot':
        cnot_noise=dict(singles=lam if single_rates is None else single_rates,
                        pair_rates=pair_rates,edges=noise_edges)
    elif single_rates is not None or noise_edges is not None or np.any(pair_rates):
        raise ValueError('correlated/scheduled rates require noise_placement="after_cnot"')
    kw = dict(nx=nx, ny=ny, pairs=list(zip(lattice.order[::2], lattice.order[1::2])),
              n_steps=steps, phi=phi, lam_xyz=lam, scale=scale, unsigned=unsigned,
              bond_cap=cap, boundary_cap=boundary_cap, symmetry='z2z2',
              gate_backend=gate_backend, boundary_backend=boundary_backend,
              jacobi_sweeps=jacobi_sweeps,
              noise_placement='gate' if cnot_noise is not None else noise_placement,
              cnot_noise=cnot_noise)
    if not residual:
        result = evolve(observable, lattice, **kw)
        return dict(approximate=result['value'], correction=0., corrected=result['value'], details=result)
    from peps_boundary_residual import streaming_residuals
    saved = []
    evolution_stats=[]
    class Captured(Exception):
        pass
    def capture(tn, log_scale):
        saved.append((tn.copy(), log_scale))
        evolution_stats.append(getattr(tn,"_cnot_evolution_stats",None))
        raise Captured
    try:
        evolve(observable, lattice, capture_boundary=capture, **kw)
    except Captured:
        pass
    if not saved:
        raise RuntimeError('PEPS boundary was not captured')
    tn, log_scale = saved[0]
    with peps_contractions():
        if boundary_backend in ('metal-jacobi','metal-hybrid'):
            from peps_gpu_boundary import gpu_boundary, to_gpu
            with gpu_boundary(jacobi_sweeps, decomposition_backend=
                              "hybrid" if boundary_backend=="metal-hybrid" else "metal") as engine:
                gpu = to_gpu(tn, engine.mx)
                def materialize(net):
                    engine.mx.eval([b for t in net for b in t.data.blocks.values()])
                result = streaming_residuals(gpu, nx, ny, boundary_cap, environment_cap,
                                             materialize=materialize, granularity=residual_granularity)
                result["boundary_gpu_stats"]=engine.stats
        else:
            result = streaming_residuals(tn, nx, ny, boundary_cap, environment_cap,
                                         granularity=residual_granularity)
    result["cnot_evolution_stats"]=evolution_stats[0]
    factor = np.exp(log_scale)
    for key in ('approximate', 'corrected', 'correction', 'absolute_correction_sum'):
        result[key] *= factor
    for item in result['contributions']:
        for key in ('before', 'after', 'correction'):
            item[key] *= factor
    return {**{k: result[k] for k in ('approximate', 'correction', 'corrected')}, 'details': result}


def curves(evaluator, scales, *, mode='both', denominator_tol=1e-12, **kwargs):
    """Compute R=O(s)/O(0), Q±=(U(s)±O(s))/(U(0)±O(0)).

    A missing/ill-conditioned ratio or sign sector is returned as None, never
    clipped. The Q tolerance is relative to the zero-noise mass; the ratio uses an
    absolute denominator tolerance.
    """
    if mode not in ('ratio', 'q', 'both'):
        raise ValueError('mode must be ratio, q, or both')
    if not np.isfinite(denominator_tol) or denominator_tol < 0:
        raise ValueError('denominator_tol must be finite and nonnegative')
    scales = sorted(set([0.] + [float(s) for s in scales]))
    if any(not np.isfinite(s) or s < 0 for s in scales):
        raise ValueError('scales must be finite and nonnegative')
    raw = []
    for s in scales:
        signed = evaluator(scale=s, unsigned=False, **kwargs)
        unsigned = evaluator(scale=s, unsigned=True, **kwargs) if mode != 'ratio' else None
        for result in (signed, unsigned):
            if result and not all(np.isfinite(result[k]) for k in ('approximate', 'correction', 'corrected')):
                raise FloatingPointError('nonfinite tensor contraction')
        raw.append(dict(s=s, signed=signed, unsigned=unsigned))
    rows = []
    for item in raw:
        row = dict(s=item['s'])
        for label, key in (('base', 'approximate'), ('corrected', 'corrected')):
            o = item['signed'][key]; o0 = raw[0]['signed'][key]
            row[f'observable_{label}'] = o
            row[f'ratio_{label}'] = o/o0 if abs(o0) > denominator_tol else None
            if mode != 'ratio':
                u = item['unsigned'][key]; u0 = raw[0]['unsigned'][key]
                row[f'unsigned_{label}'] = u
                threshold = denominator_tol * max(abs(u0), abs(o0), np.finfo(float).tiny)
                for sector, sign in (('plus', 1), ('minus', -1)):
                    num = (u + sign*o)/2; den = (u0 + sign*o0)/2
                    # A negative mass signals unresolved truncation/cancellation.
                    row[f'{sector}_num_{label}'] = num
                    row[f'{sector}_den_{label}'] = den
                    row[f'q_{sector}_{label}'] = num/den if den > threshold and num >= 0 else None
            else:
                row[f'unsigned_{label}'] = None
        rows.append(row)
    return dict(rows=rows, raw=raw)


def write_outputs(prefix, result, config, method):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    def write(suffix, rows):
        with Path(str(prefix) + suffix).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write('_curves.csv', result['rows'])
    write('_observables.csv', [dict(case='tensor_zne', s=r['s'], observable=r['observable_corrected'])
                                for r in result['rows']])
    if config['mode'] != 'ratio':
        qrows = [dict(case='tensor_zne', method=method, group=1, s=r['s'],
                      q_plus=r['q_plus_corrected'], q_minus=r['q_minus_corrected'],
                      plus_num=r['plus_num_corrected'], plus_den=r['plus_den_corrected'],
                      minus_num=r['minus_num_corrected'], minus_den=r['minus_den_corrected'])
                 for r in result['rows']]
        # Do not send undefined or nonpositive log-fit inputs into the existing fitter.
        if all(r['q_plus'] is not None and r['q_minus'] is not None and
               r['q_plus'] > 0 and r['q_minus'] > 0 for r in qrows):
            write('_q_groups.csv', qrows)
        else:
            Path(str(prefix) + '_q_groups.csv').unlink(missing_ok=True)
            print('Q fit export omitted: a sign sector is absent, nonpositive, or unresolved; inspect curves.')
    else:
        Path(str(prefix) + '_q_groups.csv').unlink(missing_ok=True)
    Path(str(prefix) + '.json').write_text(json.dumps(dict(config=config, **result), indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('engine', choices=['mps', 'peps'])
    p.add_argument('--mode', choices=['ratio', 'q', 'both'], default='both')
    p.add_argument('--scales', nargs='+', type=float, default=[0, 1, 2, 4])
    p.add_argument('--sites', nargs=2, type=int, default=[0, 1])
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--phi', type=float, default=float(np.pi/16))
    p.add_argument('--lam', nargs=3, type=float, default=[.001, .002, .003])
    p.add_argument('--cap', type=int, default=8)
    p.add_argument('--no-residual', action='store_true')
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--terminal', type=int, default=2)
    p.add_argument('--guide-cap', type=int, default=32)
    p.add_argument('--noise-strategy',choices=['local','mpo'],default='local',help='experimental MPS diagonal-noise MPO strategy')
    p.add_argument('--noise-mpo-max-bond',type=int,default=64)
    p.add_argument('--guide-cache-mb',type=float,help='after_cnot MPS retained guide snapshot budget; 0 recomputes without caching')
    p.add_argument('--summary-only',action='store_true',help='omit per-operation MPS residual diagnostics')
    p.add_argument('--mps-backend', choices=['cpu', 'metal'], default='cpu')
    p.add_argument('--nx', type=int, default=2)
    p.add_argument('--ny', type=int, default=3)
    p.add_argument('--boundary-cap', type=int, default=8)
    p.add_argument('--environment-cap', type=int, default=16)
    p.add_argument('--residual-granularity', choices=['row','bond'], default='row',
                   help='PEPS residual diagnostics; row avoids intermediate overlap contractions')
    p.add_argument('--gate-backend', choices=['quimb', 'batched-cpu', 'metal'], default='quimb')
    p.add_argument('--boundary-backend', choices=['blas', 'metal-jacobi', 'metal-hybrid'], default='blas')
    p.add_argument('--jacobi-sweeps', type=int, default=32)
    p.add_argument('--peps-noise-placement', choices=['gate', 'layer'], default='gate')
    p.add_argument('--noise-placement', choices=['heisenberg','after_cnot'], default='heisenberg',
                   help='after_cnot applies a full graph-wide Pauli-Lindblad channel after each CX layer')
    p.add_argument('--pair-rate', type=float, default=0., help='rate of EACH of the nine Pauli pair generators on every noise edge')
    p.add_argument('--noise-file', help='NPZ with optional singles, pairs, edges; allow_pickle=False')
    p.add_argument('--out', required=True, help='output prefix')
    a = p.parse_args()
    extra={}
    if (a.guide_cache_mb is not None or a.summary_only or a.noise_strategy!='local') and (a.engine!='mps' or a.noise_placement!='after_cnot'):
        p.error('--guide-cache-mb/--summary-only/--noise-strategy mpo require MPS --noise-placement after_cnot')
    if a.noise_file or a.pair_rate:
        if a.noise_placement!='after_cnot':p.error('--noise-file/--pair-rate require --noise-placement after_cnot')
    if a.noise_placement=='after_cnot':
        extra['pair_rates']=a.pair_rate
        if a.noise_file:
            with np.load(a.noise_file,allow_pickle=False) as data:
                unknown=set(data.files)-{'singles','pairs','edges'}
                if unknown:p.error(f'unknown noise arrays: {sorted(unknown)}')
                for old,new in [('singles','single_rates'),('pairs','pair_rates'),('edges','noise_edges')]:
                    if old in data:extra[new]=data[old].copy()

    kw = dict(steps=a.steps, sites=a.sites, phi=a.phi, lam=a.lam, cap=a.cap, residual=not a.no_residual)
    if a.engine == 'mps':
        fn = evaluate_mps
        kw.update(n=a.n, terminal=a.terminal, guide_cap=a.guide_cap, backend=a.mps_backend,noise_placement=a.noise_placement,guide_cache_mb=a.guide_cache_mb,store_contributions=not a.summary_only,noise_strategy=a.noise_strategy,noise_mpo_max_bond=a.noise_mpo_max_bond,**extra)
        method = 'cnot' if a.noise_placement=='after_cnot' else 'layer'
    else:
        fn = evaluate_peps
        kw.update(nx=a.nx, ny=a.ny, boundary_cap=a.boundary_cap, environment_cap=a.environment_cap,
                  gate_backend=a.gate_backend, boundary_backend=a.boundary_backend,
                  residual_granularity=a.residual_granularity,
                  jacobi_sweeps=a.jacobi_sweeps, noise_placement=
                  "after_cnot" if a.noise_placement=="after_cnot" else a.peps_noise_placement,**extra)
        method = "cnot" if a.noise_placement=="after_cnot" else a.peps_noise_placement
    result = curves(fn, a.scales, mode=a.mode, **kw)
    if a.noise_file:
        import hashlib
        a.noise_file_sha256=hashlib.sha256(Path(a.noise_file).read_bytes()).hexdigest()
    write_outputs(a.out, result, vars(a), method)
    print(f'Wrote {a.out}_curves.csv, {a.out}_observables.csv, and {a.out}.json')


if __name__ == '__main__':
    main()
