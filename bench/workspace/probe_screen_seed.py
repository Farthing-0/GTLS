import os,argparse,json,time,sys,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');p.add_argument('--periods',type=int,default=752);p.add_argument('--pattern',choices=['constant','mixed'],default='mixed');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization';base=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','59393fb','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','59393fb:'+name],cwd=repo))
sys.path.insert(0,str(snapshot));import cupy as cp,numpy as np,gputls
from gputls import core
uniform=args.pattern=='constant';loader=core.load_module;measure={};keys=[];mode=0;active=None
name='screenCandidatesUniformSparse' if uniform else 'screenCandidatesMappedSparse'
mods={}
for samples in [32]:
 m=cp.RawModule(code=(repo/'src/gputls/kernels.cu').read_text(),options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',f'-DSEED_SAMPLES={samples}'))
 mods[samples]=(m.get_function('seedScreenCandidates'),m.get_function('evaluateScreenSeedPacked' if uniform else 'evaluateScreenSeed'))
def timed(name,fn,grid,block,values):
 a,b=cp.cuda.Event(),cp.cuda.Event();a.record();fn(grid,block,values);b.record();b.synchronize();measure[name]=measure.get(name,0)+cp.cuda.get_elapsed_time(a,b)
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,kernel):
  fn=self.module.get_function(kernel)
  def run(grid,block,values):
   if mode and kernel==name:
    choose,exact=mods[mode];v=values;rows=grid[1]
    timed('chooseSeed',choose,(rows,),(256,),(v[1],v[3],v[4],v[6],v[7],v[8],v[9],v[11],*v[17:22]))
    weighted=active.workspace.view('weighted',(rows,active.patchedDatasSize,active.weightedPointBytes//4),cp.float32)
    timed('exactSeed',exact,(rows,),(32,),(v[1],v[3],v[4],weighted,active.signalGPU,v[5],v[6],v[7],v[10],v[19],v[20],np.int32(active.maxWidth),np.float32(active.uniformWeight if uniform else 0)))
   timed(kernel,fn,grid,block,values)
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*v,**kw):
 global active
 active=self
 result=best(self,*v,**kw);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Done(Exception):pass
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global measure,keys,mode
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None;s=self.primary
 for mode in [0,*mods]:
  s.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
  measure={};keys=[];start_time=time.perf_counter()
  for start in starts:s.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),mode
  record={'seed_samples':mode,'kernel_ms':dict(measure),'screen_ms':sum(v for k,v in measure.items() if 'Candidates' in k or 'Seed' in k or 'searchChi2' in k),'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_bitwise_equal':True};results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(mode,round(record['screen_ms'],3),round(record['total_kernel_ms'],3),flush=True)
 raise Done
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if not uniform:
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
