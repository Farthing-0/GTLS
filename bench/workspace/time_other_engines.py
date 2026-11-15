"""Time native TLS2/CETRA on the exact heterogeneous-dy GTLS workspace input.

Each invocation runs one engine, a small compilation warmup, one full warmup,
and the requested number of measured complete searches. Never run competing
benchmarks concurrently. CETRA retains its native duration and period grids.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parents[1]
EXPERIMENT = ROOT / 'GTLS-Compare/tls2-gtls-delta001-20261114v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=['tls2', 'cetra'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--cetra-stride', type=float, default=.01)
    parser.add_argument('--device', default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(out / 'result.json')
    os.environ.update(
        CUDA_VISIBLE_DEVICES='' if args.engine == 'tls2' else args.device,
        CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_DEVICE='0', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='14' if args.engine == 'tls2' else '1',
    )
    package = 'transitleastsquares' if args.engine == 'tls2' else 'cetra'
    source = (EXPERIMENT / 'sources/tls2/transitleastsquares' if args.engine == 'tls2'
              else ROOT / 'cetra/src/cetra')
    snapshot = out / 'source' / package
    shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path[:0] = [str(snapshot.parent), str(ROOT / 'GTLS-Compare/.deps')]
    import numpy as np

    protocol = json.loads((EXPERIMENT / 'protocol.json').read_text())
    case = next(c for c in protocol['cases'] if c['id'] == '0730d_100s')
    record = dict(engine=args.engine, phase='initializing', case=case,
                  pattern='mixed', python=sys.version, executable=sys.executable,
                  runner_sha256=digest(__file__),
                  source_sha256={str(p.relative_to(snapshot)): digest(p)
                                 for p in sorted(snapshot.rglob('*')) if p.is_file()},
                  cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])

    def save(**updates):
        record.update(updates)
        temp = out / 'result.json.tmp'
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temp.replace(out / 'result.json')

    def hashes():
        return {k: hashlib.sha256(v.tobytes()).hexdigest()
                for k, v in zip(('t', 'y', 'dy'), (t, y, dy))}

    save()
    try:
        assert digest(EXPERIMENT / case['input']) == case['input_sha256']
        with np.load(EXPERIMENT / case['input']) as data:
            t, y, dy = (data[k][:case['n_points']].copy() for k in ('t', 'y', 'dy'))
        assert hashes() == case['arrays_sha256']
        rng = np.random.default_rng(20261114)
        dy *= np.exp(rng.normal(0, .6, len(t))) * (1 + .4 * np.sin(np.arange(len(t)) * .003))
        gtls = json.loads((ROOT / 'optimization-20261115/local-initial/result.json').read_text())
        assert hashes() == gtls['input_sha256'], 'Must use exactly the measured GTLS arrays'
        expected_grid = np.sort(np.load(EXPERIMENT / case['grid']))
        save(input_sha256=hashes(), dy_min=float(dy.min()), dy_max=float(dy.max()),
             input_matches_gtls=True)
        if args.engine == 'tls2':
            import transitleastsquares as module
            from transitleastsquares.version import TLS_VERSIONING
            settings = protocol['settings']['tls2'].copy()
            sync = lambda: None
            save(version=TLS_VERSIONING, settings=settings, cpu_threads=14)

            def call(tt, yy, dd, small=False):
                kw = settings.copy()
                if small:
                    kw.update(period_min=1., period_max=2., oversampling_factor=1)
                answer = module.transitleastsquares(tt, yy, dd, verbose=False).power(**kw)
                pp = np.asarray(answer.periods)
                score = np.ma.filled(answer.power, np.nan)
                candidate = {k: float(getattr(answer, attr)) for k, attr in
                             [('period', 'period'), ('t0', 'T0'), ('duration', 'duration'),
                              ('depth', 'depth'), ('snr', 'snr'), ('sde', 'SDE')]}
                return pp, score, candidate, {}
        else:
            import cetra as module
            import pycuda.driver as cuda
            from cetra.cetra import _ensure_kernels
            _ensure_kernels()
            sync = cuda.Context.synchronize
            settings = dict(min_duration=.02, max_duration=.5, duration_log_step=1.1,
                            t0_stride_fraction=args.cetra_stride)
            periodic = dict(min_period=1., max_period=12., n_transits_min=2,
                            pgrid_R_star=1., pgrid_M_star=1., pgrid_oversample=3,
                            random_order=False, verbose=False)
            save(version=module.__version__, settings=settings, periodic_settings=periodic,
                 gpu=cuda.Context.get_device().name(),
                 gpu_pci_bus=cuda.Context.get_device().pci_bus_id(),
                 comparison_note='Same t/y/dy and period bounds; native CETRA grids, '
                                 'transit templates and phase-search rules differ from TLS.')

            def call(tt, yy, dd, small=False):
                sync()
                start = time.perf_counter()
                detector = module.TransitDetector(module.LightCurve(tt, yy, dd, verbose=False),
                                                  **settings, verbose=False)
                sync()
                setup = time.perf_counter()
                detector.linear_search(verbose=False)
                sync()
                linear = time.perf_counter()
                kw = periodic.copy()
                if small:
                    kw.update(max_period=2., pgrid_oversample=1)
                answer = detector.period_search(**kw)
                candidate = detector.get_max_likelihood_periodic_transit()
                sync()
                stop = time.perf_counter()
                details = dict(setup_seconds=setup-start, linear_seconds=linear-setup,
                               periodic_seconds=stop-linear, duration_count=detector.duration_count,
                               t0_count=detector.num_t0_strides)
                values = {k: float(getattr(candidate, k)) for k in ('period', 't0', 'duration', 'depth')}
                return (np.asarray(answer.period_array), np.asarray(answer.like_ratio_array),
                        values, details)

        assert Path(module.__file__).resolve().is_relative_to(snapshot)
        small = t < t[0] + 30
        start = time.perf_counter()
        call(t[small], y[small], dy[small], small=True)
        sync()
        save(initialization_seconds=time.perf_counter()-start, phase='full_warmup')
        print(f'{args.engine}: small warmup complete; full warmup starting', flush=True)
        start = time.perf_counter()
        call(t, y, dy)
        sync()
        save(full_warmup_seconds=time.perf_counter()-start, phase='measuring')
        print(f"{args.engine}: full warmup {record['full_warmup_seconds']:.3f}s", flush=True)
        times, stages, candidates = [], [], []
        for repeat in range(args.repeats):
            sync()
            start = time.perf_counter()
            pp, score, candidate, details = call(t, y, dy)
            sync()
            elapsed = time.perf_counter()-start
            assert pp.shape == score.shape and np.isfinite(score).all()
            assert all(np.isfinite(v) for v in candidate.values())
            if args.engine == 'tls2':
                assert np.array_equal(np.sort(pp), expected_grid)
            else:
                from cetra.cetra import period_grid
                native = period_grid(float(np.ptp(t)), min_period=1., max_period=12.,
                                     R_star=1., M_star=1., oversampling_factor=3, n_transits_min=2)
                assert np.array_equal(np.sort(pp), np.sort(native))
            assert hashes() == record['input_sha256'], 'Engine changed the input arrays'
            times.append(elapsed)
            stages.append(details)
            candidates.append(candidate)
            save(times_seconds=times, stages=stages, candidates=candidates)
            print(f'{args.engine}: repeat {repeat+1} {elapsed:.6f}s; P={candidate["period"]:.9f}', flush=True)
        np.savez_compressed(out / 'periodogram.npz', periods=pp, power=score)
        same = np.array_equal(np.sort(pp), expected_grid)
        save(ok=True, phase='complete', median_seconds=statistics.median(times),
             period_count=len(pp), grid_exactly_matches_gtls=same,
             maximum_grid_difference_days=(float(np.max(np.abs(np.sort(pp)-expected_grid)))
                                           if len(pp) == len(expected_grid) else None),
             recovered=all(abs(c['period']/protocol['injection']['period_days']-1)<.01
                           for c in candidates), periodogram_sha256=digest(out / 'periodogram.npz'))
        return 0
    except Exception:
        save(ok=False, phase='error', error=traceback.format_exc())
        print(record['error'], flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
