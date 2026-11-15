"""Summarize frozen 3090/A100 runs and independently compare all saved fields."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('experiment', type=Path)
args = parser.parse_args()
root = args.experiment.resolve()
remote = root/'a100/results'
paths = {
    '3090_workspace_mixed': root/'workspace-730-mixed',
    '3090_workspace_constant': root/'workspace-730-constant',
    '3090_cheap_mixed': root/'cheap-screen-730-mixed',
    '3090_scan1024_mixed': root/'scan1024-730-mixed',
    'A100_workspace_mixed': remote/'workspace-mixed',
    'A100_workspace_constant': remote/'workspace-constant',
    'A100_cheap_mixed': remote/'cheap-screen-mixed',
    'A100_cap629_mixed': remote/'workspace-mixed-cap629',
    'A100_reference_mixed': remote/'reference-mixed',
    'A100_device_bound_mixed': remote/'device-bound-mixed',
    'A100_device_bound_constant': remote/'device-bound-constant',
    '3090_device_bound_mixed': root/'device-bound-730-mixed',
}
full = {name: json.loads((path/'result.json').read_text()) for name, path in paths.items()}
for name, result in full.items():
    assert result['ok'] and result['phase'] == 'complete', name
runs = {}
for name, result in full.items():
    runs[name] = {key: result.get(key) for key in (
        'median_seconds', 'times_seconds', 'warmup_seconds', 'thermal_throttled',
        'peak_device_bytes', 'batch_cap', 'commit', 'input_sha256', 'source_sha256')}
    telemetry = json.loads((paths[name]/'telemetry.json').read_text())
    runs[name]['sm_mhz_median'] = float(np.median([x['sm_mhz'] for x in telemetry]))
    runs[name]['temperature_c_max'] = max(x['temperature_c'] for x in telemetry)
pairs = [
    ('3090_workspace_mixed', 'A100_workspace_mixed'),
    ('3090_workspace_constant', 'A100_workspace_constant'),
    ('3090_cheap_mixed', 'A100_cheap_mixed'),
    ('A100_workspace_mixed', 'A100_cheap_mixed'),
    ('A100_cap629_mixed', 'A100_workspace_mixed'),
    ('A100_reference_mixed', 'A100_workspace_mixed'),
    ('3090_workspace_mixed', '3090_scan1024_mixed'),
    ('3090_workspace_mixed', '3090_cheap_mixed'),
    ('A100_workspace_mixed', 'A100_device_bound_mixed'),
    ('A100_cheap_mixed', 'A100_device_bound_mixed'),
    ('A100_workspace_constant', 'A100_device_bound_constant'),
    ('3090_workspace_mixed', '3090_device_bound_mixed'),
    ('3090_device_bound_mixed', 'A100_device_bound_mixed'),
    ('3090_workspace_constant', 'A100_device_bound_constant'),
]
comparisons = []
for first, second in pairs:
    a, b = full[first], full[second]
    assert a['input_sha256'] == b['input_sha256'] and a['settings'] == b['settings']
    with np.load(paths[first]/'fields.npz') as fa, np.load(paths[second]/'fields.npz') as fb:
        assert set(fa.files) == set(fb.files)
        fields = {name: bool(fa[name].shape == fb[name].shape and fa[name].dtype == fb[name].dtype
                            and fa[name].tobytes() == fb[name].tobytes()) for name in fa.files}
    comparisons.append(dict(reference=first, candidate=second,
        speedup=a['median_seconds']/b['median_seconds'],
        all_fields_bitwise_equal=all(fields.values()), fields=fields,
        identical_source=a['source_sha256'] == b['source_sha256']))
    assert all(fields.values()), (first, second, fields)
profiles = {}
for name, path in [('3090', root/'nsys-workspace-730-mixed'),
                   ('A100', remote/'nsys-workspace-mixed')]:
    profile = json.loads((path/'summary.json').read_text())
    profiles[name] = profile['kernels']
report = dict(case='730 days / 100 seconds, 630720 samples, 48935 periods',
    timing='3 warmed synchronized API calls, except 3090_device_bound_mixed is a single verification run; profiler timing excluded',
    limitations=['Different host CPUs and GPU drivers; this is an end-to-end system comparison.',
                 '3090 varying-dy repeats recorded brief thermal flags (2/114 samples). The final single verification run had none. User requested ending further timing reruns.'],
    runs=runs, comparisons=comparisons, kernels=profiles)
(root/'a100/comparison.json').write_text(json.dumps(report, indent=2)+'\n')
with (root/'a100/timings.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['run','median_seconds','thermal_throttled','peak_GiB'])
    for name, result in runs.items():
        writer.writerow([name,result['median_seconds'],result['thermal_throttled'],result['peak_device_bytes']/2**30])
for item in comparisons:
    print(f"{item['reference']} -> {item['candidate']}: {item['speedup']:.5f}x; all 24 fields bitwise equal")
