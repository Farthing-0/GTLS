import os,argparse,json,subprocess,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');p.add_argument('--periods',type=int,default=1019);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization';base=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','071e95a','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(subprocess.check_output(['git','show','071e95a:'+name],cwd=repo))
(snapshot/'gputls/screening.py').write_bytes((repo/'src/gputls/screening.py').read_bytes())
sys.path.insert(0,str(snapshot));import cupy as cp,numpy as np,gputls
from gputls import core
from gputls.screening import uniform_sparse_certificate, certificates, SCREEN_TOLERANCES
source=(repo/'src/gputls/kernels.cu').read_text()
module=cp.RawModule(code=source,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',))
names=['scanScreeningUniformPacked','screenCandidatesUniform','refineCandidatesUniform','prepareScreenBudgets']
replacements=['scanScreeningUniformSparse','screenCandidatesUniformSparse','refineCandidatesUniformSparse','prepareScreenBudgetsUniformSparse']
mods={16:{old:module.get_function(new) for old,new in zip(names,replacements)}}
loader=core.load_module;mode=0;measure={};keys=[];active=None
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  original=self.module.get_function(name)
  def run(grid,block,values):
   chosen=mods.get(mode,{}).get(name,original)
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
 assert self.primary.uniformWeight is not None
 original=[entry[2].get() for entry in self.primary.screenCertificates]
 maxz=float(cp.max(cp.abs(self.primary.preparedGPU[:,0]*self.primary.preparedGPU[:,1])))
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None
 self.primary.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
 original_entries=list(self.primary.screenCertificates)
 for mode in [0,*mods]:
  if not mode:self.primary.screenCertificates=original_entries
  else:
   converted=[];p=self.primary.patchedDatasSize;padded=(p+mode-1)//mode*mode
   signal=self.primary.signalGPU.get();weight=float(self.primary.uniformWeight)
   for entry,tolerance in zip(original_entries,SCREEN_TOLERANCES):
    pos,coeff,meta,mass=certificates(signal,self.primary.durations,maxz,weight,padded,tolerance)
    pos,coeff,meta=uniform_sparse_certificate(pos,coeff,meta,mode,padded,maxz)
    converted.append((cp.asarray(pos),cp.asarray(coeff),cp.asarray(meta),*map(np.float64,mass)))
   self.primary.screenCertificates=converted
  self.primary.screenMapKey=None
  measure={};keys=[];start_time=time.perf_counter()
  for start in starts:self.primary.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),mode
  record={'stride':mode,'kernel_ms':dict(measure),'target_ms':sum(v for k,v in measure.items() if k in names),'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_bitwise_equal':True};results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(mode,round(record['target_ms'],3),round(record['elapsed_seconds'],4),flush=True)
 raise Done
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
