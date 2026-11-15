"""Measure power-of-two buckets against the current exact sort pipeline."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='1')
parser.add_argument('--rows', type=int, default=512)
parser.add_argument('--launch-sweep', action='store_true')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
repo = Path(__file__).resolve().parents[2]
root = repo.parents[1] / 'GTLS-Compare/tls2-gtls-delta001-20261114v2'
sys.path.insert(0, str(repo / 'src'))
import cupy as cp
import numpy as np
from gputls import core

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
source = core.KERNEL_SOURCE.read_text()
power2 = source.replace(
    '    if (key != 0xffffffffu) {',
    '    return __umulhi(key, (unsigned int)buckets);\n    if (key != 0xffffffffu) {', 1)
power2 = power2.replace('int bucket = min((int)(pointPhase * nBuckets), nBuckets - 1);',
                        'int bucket = __umulhi(phaseKey(pointPhase), (unsigned int)nBuckets);')
(out / 'power2.cu').write_text(power2)
mods = {k: cp.RawModule(code=s, options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',))
        for k, s in [('original', source), ('power2', power2)]}
names = ['foldBucketsPackedPhase', 'scanCounts', 'bucketScatterPackedPhase', 'sortBucketsPackedPhase']
funcs = {k: [m.get_function(n) for n in names] for k, m in mods.items()}
protocol = json.loads((root / 'protocol.json').read_text())
case = next(c for c in protocol['cases'] if c['id'] == '0730d_100s')
with np.load(root / case['input']) as data:
    t = data['t'][:case['n_points']]
periods = np.sort(np.load(root / case['grid']))
n, rows = len(t), args.rows
gput = cp.asarray(t)
phase = cp.empty((rows, n), cp.uint32)
index = cp.empty((rows, n), cp.int32)
reference = cp.empty_like(index)
b = (n+2)//3
base_power = 1 << (b.bit_length()-1)
variants = [('original', b, 256), ('power2', base_power, 256), ('power2', 2*base_power, 256)]
if args.launch_sweep:
    variants = [('original', b, threads) for threads in [64, 128, 256, 512, 1024]]
counts_storage = cp.empty(rows*max(x[1] for x in variants), cp.int32)
cursor_storage = cp.empty_like(counts_storage)
results = []
for offset in [0, 16000, 40000]:
    inverse = cp.asarray(1 / periods[offset:offset+rows])
    def pipeline(kind, buckets, threads=256):
        counts = counts_storage[:rows*buckets].reshape(rows, buckets)
        cursor = cursor_storage[:rows*buckets].reshape(rows, buckets)
        fold, scan, scatter, sort = funcs[kind]
        events = [cp.cuda.Event() for _ in range(6)]
        events[0].record()
        counts.fill(0)
        events[1].record()
        fold(((n+255)//256, rows), (256,), (gput, inverse, phase, counts, np.int32(n), np.int32(buckets)))
        events[2].record()
        scan((rows,), (512,), (counts, cursor, np.int32(buckets)))
        events[3].record()
        scatter(((n+threads-1)//threads, rows), (threads,),
                (phase, gput, inverse, cursor, index, np.int32(n), np.int32(buckets)))
        events[4].record()
        sort(((buckets+255)//256, rows), (256,),
             (phase, gput, inverse, counts, cursor, index, np.int32(n), np.int32(buckets)))
        events[5].record()
        events[-1].synchronize()
        return {k: cp.cuda.get_elapsed_time(events[i], events[i+1])
                for i, k in enumerate(['zero', 'fold', 'scan', 'scatter', 'sort'])}
    pipeline('original', b)
    cp.copyto(reference, index)
    for kind, buckets, threads in variants:
        pipeline(kind, buckets, threads)
        assert bool(cp.all(reference == index)), (offset, kind, buckets)
    samples = {v: [] for v in variants}
    for repeat in range(4):
        for variant in (variants if repeat%2 == 0 else variants[::-1]):
            samples[variant].append(pipeline(*variant))
    for (kind, buckets, threads), values in samples.items():
        medians = {k: statistics.median(r[k] for r in values) for k in values[0]}
        result = dict(offset=offset, kind=kind, buckets=buckets, threads=threads, median_ms=medians,
                      total_ms=sum(medians.values()), indices_bitwise_equal=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2)+'\n')
