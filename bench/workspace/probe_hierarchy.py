"""Compare hierarchical certificates while retaining the exact candidate arithmetic."""
import argparse,os,json,statistics,time,sys,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--pattern',choices=['mixed','constant'],required=True);parser.add_argument('--device',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--periods',type=int,default=64);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
files=subprocess.check_output(['git','ls-tree','-r','--name-only','e27ca6c','src/gputls'],cwd=repo,text=True).splitlines()
for name in files:
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','e27ca6c:'+name],cwd=repo))
sys.path.insert(0,str(snapshot))
import cupy as cp,numpy as np,gputls
from gputls import core,screening
source=core.KERNEL_SOURCE.read_text()
refined=source.replace('#if GTLS_CHEAP_SCREEN_BOUNDS\n    // A valid upper', '#if GTLS_CHEAP_SCREEN_BOUNDS || GTLS_REFINE_SCREEN\n    // A valid upper')
refined=refined.replace('if (item < count) {','if (item < count && !(lower[(long long)period*count+item] > earlierUpper)) {')
refined=refined.replace('lower[(long long)period*count+item] = lo;', 'lower[(long long)period*count+item] = fmaxf(lower[(long long)period*count+item], lo);')
(args.output/'refined.cu').write_text(refined)
refine_module=cp.RawModule(code=refined,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0','-DGTLS_REFINE_SCREEN=1'))
refine_screen=refine_module.get_function('screenCandidates')
loader=core.load_module;certificate=screening.template_certificate
config=(.01,);measure={};keys=[];survival=[];cache={}
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  fun=self.module.get_function(name)
  def timed(kernel,key,grid,block,values):
   begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record();kernel(grid,block,values);end.record();end.synchronize();measure[key]=measure.get(key,0)+cp.cuda.get_elapsed_time(begin,end)
  def run(grid,block,values):
   if name!='screenCandidates':return timed(fun,name,grid,block,values)
   for index,tolerance in enumerate(config):
    meta=cache[tolerance]
    timed(fun if index==0 else refine_screen,f'screen{index}',grid,block,values[:12]+meta+values[17:])
   lower,upper=values[:2];threshold=core.decode_keys(upper,0)[0];active=lower<cp.inf;kept=(lower<=threshold[:,None])|cp.isnan(threshold[:,None]);survival.append(int((active&kept).sum()))
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*values,**kwargs):
 result=best(self,*values,**kwargs);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Complete(Exception):pass
configs=[(.01,),(.015,.005,.00025),(.05,.015,.005,.00025)]
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global measure,survival,keys,config,cache
 s=self.primary;signal=s.signalGPU.get();dy2=cp.maximum(s.dyGPU*s.dyGPU,cp.float32(1e-10));weight=cp.float32(1)/dy2;maxz=float(cp.max(cp.abs(weight*s.residualGPU)));maxw=float(cp.max(weight))
 for tolerance in sorted({t for config in configs for t in config}):
  screening.template_certificate=lambda signal:certificate(signal,tolerance)
  pos,coef,meta,error=screening.certificates(signal,s.durations,maxz,maxw,s.patchedDatasSize)
  cache[tolerance]=(cp.asarray(pos),cp.asarray(coef),cp.asarray(meta),*map(np.float64,error))
 screening.template_certificate=certificate
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None
 config=(.01,)
 self.primary.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
 for config in configs:
  measure={};survival=[];keys=[];start_time=time.perf_counter()
  for start in starts:self.primary.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,ref) for x,ref in zip(keys,baseline)),config
  result={'tolerances':config,'kernel_ms':dict(measure),'screen_and_exact_ms':sum(v for k,v in measure.items() if k.startswith(('screen','searchChi2'))),'total_kernel_ms':sum(measure.values()),'survivors':sum(survival),'keys_bitwise_equal':True,'elapsed_seconds':time.perf_counter()-start_time}
  results.append(result);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(args.pattern,config,'screen/exact',round(result['screen_and_exact_ms'],3),'total',round(result['total_kernel_ms'],3),'survivors',result['survivors'],flush=True)
 raise Complete
core.MultiGPUSearch.chi2=probe
protocol=json.loads((root/'protocol.json').read_text());case=next(x for x in protocol['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if args.pattern=='mixed':
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**protocol['settings']['gtls'])
except Complete:pass
print('BEST',min(results,key=lambda r:r['screen_and_exact_ms'])['tolerances'],flush=True)
