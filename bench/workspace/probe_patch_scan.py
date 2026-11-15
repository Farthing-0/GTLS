"""Choose patch/scan launch geometry on genuine folded 730-day inputs."""
import argparse, json, os, sys, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
exp = repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=a.device, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMBA_NUM_THREADS='1', CUPY_CACHE_DIR=str(exp/'cache/cupy'),
                  NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot = a.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','21fc9ae','src/gputls'], cwd=repo, text=True).splitlines():
    target = snapshot/Path(name).relative_to('src')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(subprocess.check_output(['git','show','21fc9ae:'+name], cwd=repo))
sys.path.insert(0, str(snapshot))
import cupy as cp
import numpy as np
from gputls import core, gtls

source = (repo/'src/gputls/kernels.cu').read_text()
(a.output/'candidate.cu').write_text(source)
candidate = cp.RawModule(code=source,
                         options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',))
loader = core.load_module
original = loader()
active = None
results = []
class Done(Exception): pass

def benchmark(values):
    s = active
    index, residual, dy, patched, weighted, n, size = values
    cumulative = s.workspace.view('cumulative', (752, int(size)), cp.int64)
    prepared = cp.empty((int(n), 2), cp.float32)
    candidate.get_function('prepareWeights')(((int(n)+255)//256,), (256,),
                                             (residual, dy, prepared, n))
    for rows in (1, 16, 32, 64, 82, 128, 256, 512, 752):
        def separate():
            original.get_function('patchWeighted')(((int(size)+255)//256, rows), (256,), values)
            original.get_function('scanResidual')((rows,), (512,),
                (patched, cumulative, s.fixedScale, size))
        variants = {'baseline': separate}
        for threads in (128, 256, 512, 1024):
            variants['fused'+str(threads)] = lambda threads=threads: candidate.get_function('patchScan')(
                (rows,), (threads,), (index, prepared, weighted, cumulative, s.fixedScale, n, size))
        separate()
        take = [0, rows-1]
        reference_w, reference_c = weighted[take].get(), cumulative[take].get()
        measurements = {name: [] for name in variants}
        for repeat in range(4):
            for name in np.random.default_rng(900+repeat).permutation(list(variants)):
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record(); variants[name](); end.record(); end.synchronize()
                if repeat:
                    measurements[name].append(cp.cuda.get_elapsed_time(start, end))
                else:
                    assert weighted[take].get().tobytes() == reference_w.tobytes(), (rows,name,'weighted')
                    assert cumulative[take].get().tobytes() == reference_c.tobytes(), (rows,name,'cumulative')
        result = {'rows': rows, 'milliseconds': {k:float(np.median(v)) for k,v in measurements.items()},
                  'sampled_rows_bitwise_equal': True}
        results.append(result)
        (a.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(result, flush=True)
    raise Done

class Module:
    def get_function(self, name):
        if name != 'patchWeighted': return original.get_function(name)
        return lambda grid, block, values: benchmark(values)
core.load_module = Module
best = core.GPUSearch.best_keys
def remember(self, *args, **kw):
    global active
    active = self
    return best(self, *args, **kw)
core.GPUSearch.best_keys = remember
def probe(self, periods, capacity, refine, pbar=None):
    self.primary.chi2_batch(np.asarray(periods)[len(periods)//2:len(periods)//2+752], 752, False)
core.MultiGPUSearch.chi2 = probe
protocol = json.loads((exp.parent/'protocol.json').read_text())
case = next(c for c in protocol['cases'] if c['id']=='0730d_100s')
with np.load(exp.parent/case['input']) as data:
    t,y,dy = (data[k][:case['n_points']].copy() for k in ('t','y','dy'))
dy *= np.exp(np.random.default_rng(20261114).normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try: gtls(t,y,dy,verbose=False).power(**protocol['settings']['gtls'])
except Done: pass
