"""Compare screening kernels on one representative 64-period batch, using CUDA events."""
import argparse
import re
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
parser.add_argument('--baseline-ref',default='178b894')
parser.add_argument('--periods',type=int,default=64)
parser.add_argument('--unrolls',default='2,4,8,16')
parser.add_argument('--blocks',default='256')
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
variants={'baseline':(old,256,old_source)}
for unroll in map(int,args.unrolls.split(',')):
    replacement=f"""int j = begin;
                    for (; j+{unroll} <= begin+terms; j += {unroll}) {{
                        double2 values[{unroll}], weights[{unroll}];
                        #pragma unroll
                        for (int q=0;q<{unroll};++q) {{
                            int k=start+positions[j+q]-2;
                            values[q]=k>=0?p[k]:make_double2(0,0);
                            weights[q]=coefficients[j+q];
                        }}
                        #pragma unroll
                        for (int q=0;q<{unroll};++q) {{
                            ar=fma(weights[q].x,values[q].x,ar);
                            a2=fma(weights[q].y,values[q].y,a2);
                        }}
                    }}
                    for (;j<begin+terms;++j) {{
                        int k=start+positions[j]-2;
                        double2 v=k>=0?p[k]:make_double2(0,0);
                        double2 c=coefficients[j];
                        ar=fma(c.x,v.x,ar);a2=fma(c.y,v.y,a2);
                    }}"""
    source,count=re.subn(r'for \(int j = begin; j < begin\+terms; \+\+j\) \{[^}]*\}',replacement,old_source)
    assert count==2,count
    module=cp.RawModule(code=source,options=options)
    fun=module.get_function('screenCandidates')
    for block in map(int,args.blocks.split(',')):
        variants[f'u{unroll}-b{block}']=(fun,block,source)
    (args.output/f'unroll-{unroll}.cu').write_text(source)

loader=core.load_module
record={'periods':args.periods,'baseline_sha256':hashlib.sha256(old_source.encode()).hexdigest(),
        'candidate_sha256':hashlib.sha256(new_source.encode()).hexdigest(),'timing':'CUDA event kernel time only'}

class Module:
    def __init__(self):self.module=loader()
    def get_function(self,name):
        fun=self.module.get_function(name)
        if name!='screenCandidates':return fun
        def run(grid,block,values):
            lower,upper=values[:2]
            def call(which,timed=False):
                kernel,threads,_=variants[which]
                launch=((lower.shape[1]+threads-1)//threads,lower.shape[0])
                upper.fill(core.EMPTY_KEY)
                if timed:start,end=cp.cuda.Event(),cp.cuda.Event();start.record()
                kernel(launch,(threads,),values)
                if timed:end.record();end.synchronize();return cp.cuda.get_elapsed_time(start,end)
            call('baseline');expected_lower=lower.copy();expected_upper=upper.copy()
            for name in variants:
                call(name)
                assert bool(cp.all(lower.view(cp.uint32)==expected_lower.view(cp.uint32))),name
                assert bool(cp.all(upper==expected_upper)),name
            for i in range(60):call(list(variants)[i%len(variants)])
            cp.cuda.runtime.deviceSynchronize()
            times={name:[] for name in variants}
            for repeat in range(6):
                for name in (list(variants) if repeat%2==0 else list(reversed(variants))):
                    times[name].append(call(name,True))
            record.update(times_ms=times,median_ms={k:statistics.median(v) for k,v in times.items()},
                candidates=int(lower.size),bounds_bitwise_equal=True,upper_keys_bitwise_equal=True,
                registers={k:v[0].attributes['num_regs'] for k,v in variants.items()})
            record['speedups']={k:record['median_ms']['baseline']/v for k,v in record['median_ms'].items()}
            call('baseline')
        return run
core.load_module=Module
class Done(Exception):pass

def probe(self,periods,capacity,refine,pbar=None):
    batch=np.asarray(periods)[np.linspace(0,len(periods)-1,args.periods,dtype=int)]
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
assert record['bounds_bitwise_equal'] and record['upper_keys_bitwise_equal']
