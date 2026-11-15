"""Timing and exact winner checks for outward-rounded screening budgets."""
import argparse,os,json,time,sys,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--pattern',choices=['mixed','constant'],default='mixed');parser.add_argument('--periods',type=int,default=629);parser.add_argument('--device',default='GPU-a8281b40-72b0-f4a1-95ba-d0eeefb61ba4');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','6715468','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','6715468:'+name],cwd=repo))
sys.path.insert(0,str(snapshot))
import cupy as cp,numpy as np,gputls
from gputls import core
source=(repo/'src/gputls/kernels.cu').read_text();(args.output/'candidate.cu').write_text(source);candidate=cp.RawModule(code=source,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));kernels={name:candidate.get_function(name) for name in ['screenCandidates','refineCandidates']};loader=core.load_module;mode='baseline';measure={};keys=[];checked=False
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  fun=self.module.get_function(name)
  def run(grid,block,values):
   global checked
   chosen=kernels[name] if name in kernels and (mode=='all' or mode=='first' and name=='screenCandidates') else fun
   if name=='screenCandidates' and not checked:
    lower,upper=values[:2];fun(grid,block,values);old_lower=lower.copy();old_upper=upper.copy();upper.fill(core.EMPTY_KEY);kernels[name](grid,block,values)
    assert bool(cp.all(lower<=old_lower)), 'new first-pass lower bound is tighter than baseline'
    assert bool(cp.all((upper>>32)>=(old_upper>>32))), 'new first-pass upper bound is tighter than baseline'
    checked=True;upper.fill(core.EMPTY_KEY)
   begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();chosen(grid,block,values);end.record();end.synchronize();measure[name]=measure.get(name,0)+cp.cuda.get_elapsed_time(begin,end)
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*values,**kwargs):
 result=best(self,*values,**kwargs);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Done(Exception):pass
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global mode,measure,keys
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None
 self.primary.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
 for mode in ['baseline','first','all']:
  measure={};keys=[];start_time=time.perf_counter()
  for start in starts:self.primary.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),mode
  record={'mode':mode,'kernel_ms':dict(measure),'screen_ms':sum(v for k,v in measure.items() if k in kernels),'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_bitwise_equal':True,'first_pass_bounds_not_narrower':checked};results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(args.pattern,mode,'screen',round(record['screen_ms'],3),'wall',round(record['elapsed_seconds'],4),flush=True)
 raise Done
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if args.pattern=='mixed':
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
