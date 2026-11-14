"""Compare screening kernels on one representative 64-period batch, using CUDA events."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

repo=Path(__file__).resolve().parents[2]
exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--baseline-ref',default='b979260')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-a8281b40-72b0-f4a1-95ba-d0eeefb61ba4',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
    CUPY_CACHE_DIR=str(exp/'workspace-optimization/cache/cupy'),
    NUMBA_CACHE_DIR=str(exp/'workspace-optimization/cache/numba'))
sys.path.insert(0,str(repo/'src'))
import cupy as cp
import numpy as np
import gputls
from gputls import core

old_source=subprocess.check_output(['git','show',args.baseline_ref+':src/gputls/kernels.cu'],cwd=repo,text=True)
new_source=core.KERNEL_SOURCE.read_text()
(args.output/'baseline.cu').write_text(old_source);(args.output/'candidate.cu').write_text(new_source)
options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',)
old_module=cp.RawModule(code=old_source,options=options)
old=old_module.get_function('screenCandidates')
old_scan=old_module.get_function('scanScreening')
loader=core.load_module
record={'periods':64,'baseline_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
        'candidate_sha256':hashlib.sha256(new_source.encode()).hexdigest(),'timing':'CUDA event kernel time only'}

class Module:
    def __init__(self):self.module=loader()
    def get_function(self,name):
        fun=self.module.get_function(name)
        if name=='scanScreening':
            def scan(grid,block,values):
                weighted,prefix,size=values
                self.old_prefix=cp.empty_like(prefix)
                old_scan(grid,block,(weighted,self.old_prefix,size))
                fun(grid,block,values)
                old_view=self.old_prefix.reshape(grid[0],int(size),4)
                new_view=prefix.reshape(grid[0],2,int(size),2)
                record['prefix_bitwise_equal']=bool(cp.all(old_view[:,:,:2].view(cp.uint64)==new_view[:,0].view(cp.uint64)) and cp.all(old_view[:,:,2:].view(cp.uint64)==new_view[:,1].view(cp.uint64)))
            return scan
        if name!='screenCandidates':return fun
        def run(grid,block,values):
            lower,upper=values[:2]
            def call(which,timed=False):
                upper.fill(core.EMPTY_KEY)
                if timed:
                    start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
                if which=='baseline':old(grid,block,values[:2]+(self.old_prefix,)+values[3:])
                else:fun(grid,block,values)
                if timed:
                    end.record();end.synchronize();return cp.cuda.get_elapsed_time(start,end)
            call('baseline');expected_lower=lower.copy();expected_upper=upper.copy()
            call('candidate');cp.cuda.runtime.deviceSynchronize()
            record['bounds_bitwise_equal']=bool(cp.all(lower.view(cp.uint32)==expected_lower.view(cp.uint32)))
            record['upper_keys_bitwise_equal']=bool(cp.all(upper==expected_upper))
            times={'baseline':[],'candidate':[]}
            for repeat in range(6):
                for which in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
                    times[which].append(call(which,True))
            record.update(times_ms=times,median_ms={k:statistics.median(v) for k,v in times.items()},
                          candidates=int(lower.size),grid=list(grid),block=list(block))
            record['speedup']=record['median_ms']['baseline']/record['median_ms']['candidate']
            call('candidate')
        return run
core.load_module=Module
class Done(Exception):pass

def probe(self,periods,capacity,refine,pbar=None):
    batch=np.asarray(periods)[np.linspace(0,len(periods)-1,64,dtype=int)]
    self.primary.chi2_batch(batch,len(batch),False)
    raise Done
core.MultiGPUSearch.chi2=probe
protocol=json.loads((exp/'protocol.json').read_text())
case=next(c for c in protocol['cases'] if c['id']=='0730d_100s')
with np.load(exp/case['input']) as data:
    t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
rng=np.random.default_rng(20261114)
dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**protocol['settings']['gtls'])
except Done:pass
(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
assert record['prefix_bitwise_equal'] and record['bounds_bitwise_equal'] and record['upper_keys_bitwise_equal']
