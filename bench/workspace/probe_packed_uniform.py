import os,argparse,json,time,sys,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');p.add_argument('--periods',type=int,default=752);p.add_argument('--pattern',default='constant');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization';base=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','2b6067a','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True);s=subprocess.check_output(['git','show','2b6067a:'+name],cwd=repo)
 target.write_bytes(s)
sys.path.insert(0,str(snapshot));import cupy as cp,numpy as np,gputls
from gputls import core
source=(repo/'src/gputls/kernels.cu').read_text();(args.output/'candidate.cu').write_text(source);mods={};names=['patchScan','scanScreeningUniform']+[f'searchChi2{kind}G{g}' for kind in ['', 'Compacted','Screened'] for g in [1,2,4,8,16,32]]
module=cp.RawModule(code=source,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));mods['packed']={}
for k in names:
 replacement=k+'Packed' if k in ['patchScan','scanScreeningUniform'] else k.replace('searchChi2','searchChi2Packed')
 mods['packed'][k]=module.get_function(replacement)
loader=core.load_module;mode='baseline';measure={};keys=[];active=None
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  original=self.module.get_function(name)
  def run(grid,block,values):
   chosen=mods.get(mode,{}).get(name,original)
   if mode!='baseline' and name.startswith('searchChi2') and name in names:values=values+(active.uniformWeight,)
   a,b=cp.cuda.Event(),cp.cuda.Event();a.record();chosen(grid,block,values);b.record();b.synchronize();measure[name]=measure.get(name,0)+cp.cuda.get_elapsed_time(a,b)
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*v,**kw):
 global active
 active=self;result=best(self,*v,**kw);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Done(Exception):pass
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global mode,measure,keys
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None
 assert self.primary.uniformWeight is not None
 self.primary.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
 for mode in ['baseline',*mods]:
  measure={};keys=[];start_time=time.perf_counter()
  for start in starts:self.primary.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),mode
  record={'mode':mode,'kernel_ms':dict(measure),'target_ms':sum(v for k,v in measure.items() if k in names),'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_bitwise_equal':True};results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(mode,round(record['target_ms'],3),round(record['elapsed_seconds'],4),flush=True)
 raise Done
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if args.pattern=='mixed':
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
