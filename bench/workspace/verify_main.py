"""Freeze both revisions, then compare all native fields on short light curves."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='1')
parser.add_argument('--reference', default='main')
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
revisions = {}
for name, ref in [('main', args.reference), ('optimized', 'HEAD')]:
    sha = subprocess.check_output(['git', 'rev-parse', ref], cwd=REPO, text=True).strip()
    revisions[name] = sha
    for path in subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', sha,
                                         'src/gputls'], cwd=REPO, text=True).splitlines():
        dest = out / 'sources' / name / Path(path).relative_to('src')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(['git', 'show', sha + ':' + path], cwd=REPO))

rows = []
for case, pattern in [('0030d_600s', 'constant'), ('0030d_600s', 'random'),
                      ('0030d_100s', 'mixed'), ('0030d_100s', 'spikes')]:
    paths = []
    for name in revisions:
        dest = out / (case + '-' + pattern) / name
        paths.append(dest)
        subprocess.run([sys.executable, str(Path(__file__).with_name('run.py')),
                        '--case', case, '--pattern', pattern, '--device', args.device,
                        '--source', str(out / 'sources' / name / 'gputls'),
                        '--repeats', '1', '--output', str(dest)], check=True)
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('compare.py')),
                             *map(str, paths)], capture_output=True, text=True)
    comparison = json.loads((paths[1] / 'comparison.json').read_text())
    rows.append(dict(case=case, pattern=pattern, **comparison))
    summary = dict(revisions=revisions, cases=rows,
                   all_fields_bitwise_equal=all(r['all_fields_bitwise_equal'] for r in rows))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(case, pattern, comparison['all_fields_bitwise_equal'], flush=True)
    if result.returncode:
        print(result.stdout, result.stderr, flush=True)
raise SystemExit(0 if summary['all_fields_bitwise_equal'] else 1)
