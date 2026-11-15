"""Measure compaction including host launch sizing, on actual period batches."""
import argparse,os,json,time,sys,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--pattern',choices=['mixed','constant'],default='mixed');parser.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--periods',type=int,default=629);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','d78df2c','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','d78df2c:'+name],cwd=repo))
sys.path.insert(0,str(snapshot))
import cupy as cp,numpy as np,gputls
from gputls import core
source=(repo/'src/gputls/kernels.cu').read_text();(args.output/'indexed.cu').write_text(source)
indexed_module=cp.RawModule(code=source,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));compact=indexed_module.get_function('compactCandidates');indexed={g:indexed_module.get_function(f'searchChi2CompactedG{g}') for g in [1,2,4,8,16,32]}
loader=core.load_module;enabled=False;measure={};details=[];keys=[];active_search=None;groups=(32,)
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  fun=self.module.get_function(name)
  def timed(kernel,key,grid,block,values):
   begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();kernel(grid,block,values);end.record();end.synchronize();measure[key]=measure.get(key,0)+cp.cuda.get_elapsed_time(begin,end)
  def run(grid,block,values):
   if not enabled or not name.startswith('searchChi2ScreenedG'):return timed(fun,name,grid,block,values)
   g=int(name.removeprefix('searchChi2ScreenedG'))
   if g not in groups:return timed(fun,name,grid,block,values)
   best=values[0];offsets=values[8].get();begin,end=(int(offsets[int(values[i])]) for i in (11,12));count=end-begin;periods=len(best)
   candidates=active_search.workspace.view('scratch',(periods,count),cp.int32);counts=active_search.workspace.view('scratch',(periods,),cp.int32,periods*count*4);counts.fill(0)
   timed(compact,'compact'+str(g),((count+255)//256,periods),(256,),(*values[-3:-1],candidates,counts,np.int32(begin),np.int32(end),values[-1],np.int32(count)))
   maximum=int(counts.max());details.append({'g':g,'max_survivors':maximum,'total_survivors':int(counts.sum()),'candidates':count*periods})
   if maximum>count//4:return timed(fun,name,grid,block,values)
   if maximum:timed(indexed[g],'indexed'+str(g),((maximum*g+255)//256,periods),(256,),values+(candidates,counts,np.int32(count)))
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*values,**kwargs):
 global active_search
 active_search=self;result=best(self,*values,**kwargs);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Complete(Exception):pass
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global measure,details,keys,enabled,groups
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None
 enabled=True;groups=(4,8,16,32)
 self.primary.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
 for groups in [(),(32,),(16,32),(8,16,32),(4,8,16,32)]:
  enabled=bool(groups);measure={};details=[];keys=[];start_time=time.perf_counter()
  for start in starts:self.primary.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),groups
  record={'groups':groups,'kernel_ms':dict(measure),'exact_and_compact_ms':sum(v for k,v in measure.items() if k.startswith(('searchChi2','compact','indexed'))),'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_equal':True,'details':details}
  results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(args.pattern,groups,'exact+compact',round(record['exact_and_compact_ms'],3),'wall',round(record['elapsed_seconds'],4),flush=True)
 raise Complete
core.MultiGPUSearch.chi2=probe
protocol=json.loads((root/'protocol.json').read_text());case=next(x for x in protocol['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if args.pattern=='mixed':
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**protocol['settings']['gtls'])
except Complete:pass
