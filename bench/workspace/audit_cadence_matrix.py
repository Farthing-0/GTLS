"""Audit a downloaded native-grid matrix and summarize separate nsys captures."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def profile_summary(path):
    result = {}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        names = dict(con.execute('SELECT id,value FROM StringIds'))
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for kind, table, field in [
            ('kernels', 'CUPTI_ACTIVITY_KIND_KERNEL', 'demangledName'),
            ('runtime', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'nameId'),
            ('driver', 'CUPTI_ACTIVITY_KIND_DRIVER', 'nameId'),
        ]:
            if table not in tables:
                continue
            rows = list(con.execute(
                f'SELECT {field},COUNT(*),SUM(end-start) FROM {table} '
                f'GROUP BY {field} ORDER BY SUM(end-start) DESC'))
            total = sum(r[2] for r in rows)
            result[kind] = [dict(name=names.get(name, str(name)), calls=count,
                                 seconds=ns/1e9, percent=100*ns/total if total else 0)
                            for name, count, ns in rows]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    root = args.directory.resolve()
    protocol = read(root/'protocol.json')
    prepared = args.workspace/'a100-cetra-matrix-20261115/data/protocol.json'
    assert digest(prepared) == digest(root/'protocol.json')
    assert len(protocol['cases']) == 30 and protocol['dy_pattern'] == 'mixed'
    assert protocol['gtls_branch'] == 'main' and protocol['versions']['gtls'] == '2.2.1'
    assert (root/'branch.txt').read_text().strip() == 'main'
    assert (root/'commit.txt').read_text().strip() == protocol['gtls_commit']
    summary = read(root/'summary.json')
    assert summary['complete'] and summary['completed_jobs'] == 60
    missed, hot, variability = [], [], []
    for case in protocol['cases']:
        for engine in ('gtls', 'cetra'):
            path = root/'measurements'/case['id']
            record = read(path/(engine+'.json'))
            assert record['ok'] and record['status'] == 'complete'
            assert record['version'] == protocol['versions'][engine]
            assert record['protocol_sha256'] == digest(root/'protocol.json')
            assert record['case']['arrays_sha256'] == case['arrays_sha256']
            times = record['timings_seconds']
            assert len(times) == 3 and all(t > 0 for t in times)
            assert statistics.median(times) == record['median_seconds']
            with np.load(path/(engine+'_periodogram.npz')) as data:
                periods, power = data['periods'], data['power']
                assert periods.shape == power.shape and np.isfinite(power).all()
                assert len(periods) == record['grid']['count']
                assert hashlib.sha256(np.sort(periods).astype('<f8').tobytes()).hexdigest() == record['grid']['sha256']
            if not record['all_periods_recovered']:
                missed.append(dict(case=case['id'], engine=engine, candidates=record['candidates']))
            if any(t['thermal_throttled'] for t in record['telemetry']):
                hot.append(dict(case=case['id'], engine=engine))
            variability.append(dict(case=case['id'], engine=engine,
                                    coefficient_of_variation=float(np.std(times, ddof=1)/np.mean(times))))
    cross = {}
    for engine, local in [('gtls', 'local-initial/fields.npz'),
                          ('cetra', 'local-cetra-mixed/periodogram.npz')]:
        with np.load(args.workspace/'optimization-20261115'/local) as a, np.load(
            root/'measurements/0730d_100s'/(engine+'_periodogram.npz')) as b:
            ia, ib = np.argsort(a['periods']), np.argsort(b['periods'])
            assert np.array_equal(a['periods'][ia], b['periods'][ib])
            x, y = a['power'][ia], b['power'][ib]
            error = np.abs(x-y)
            cross[engine] = dict(period_grid_exact=True, power_exact=np.array_equal(x, y),
                                 power_allclose_1e_5=bool(np.allclose(x, y, rtol=1e-5, atol=1e-5)),
                                 max_abs_power_error=float(error.max()),
                                 max_error_over_peak=float(error.max()/max(float(np.max(np.abs(x))), 1e-30)))
    profiles = {p.stem: profile_summary(p) for p in sorted((root/'nsys').glob('*.sqlite'))}
    assert len(profiles) == 4, f'Expected four nsys captures, found {len(profiles)}'
    ratios = [r['cetra_over_gtls'] for r in summary['rows']]
    audit = dict(complete=True, source_commit=protocol['gtls_commit'], cases=30, engine_cases=60,
                 repetitions=180, input_protocol_verified=True, missed_recoveries=missed,
                 thermal_flags=hot, cross_gpu_730d_100s=cross,
                 gtls_faster_cases=sum(r > 1 for r in ratios), cetra_faster_cases=sum(r < 1 for r in ratios),
                 speed_ratio_range=[min(ratios), max(ratios)],
                 maximum_coefficient_of_variation=max(variability, key=lambda r:r['coefficient_of_variation']),
                 profiles=profiles)
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'profiles'}, indent=2))


if __name__ == '__main__':
    main()
