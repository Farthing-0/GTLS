import os,argparse,json,time,sys,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-a8281b40-72b0-f4a1-95ba-d0eeefb61ba4');p.add_argument('--periods',type=int,default=752);p.add_argument('--pattern',choices=['constant','mixed'],default='mixed');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization';base=Path(__file__).resolve().parent
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','25c5695','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','25c5695:'+name],cwd=repo))
(snapshot/'gputls/screening.py').write_bytes((repo/'src/gputls/screening.py').read_bytes())
sys.path.insert(0,str(snapshot));import cupy as cp,numpy as np,gputls
from gputls import core
from gputls.screening import certificates,sparse_certificate,SCREEN_TOLERANCES
from gputls.screening import prefix_quantization as parameters
uniform=args.pattern=='constant';loader=core.load_module;measure={};keys=[];mode=0;scales=None
candidate=cp.RawModule(code=(repo/'src/gputls/kernels.cu').read_text(),options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));name='scanScreeningUniformSparse' if uniform else 'scanScreeningSparse';replacement=candidate.get_function('scanScreeningUniformInteger' if uniform else 'scanScreeningInteger')
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,kernel):
  fn=self.module.get_function(kernel)
  def run(grid,block,values):
   function=fn
   if mode and kernel==name:function=replacement;values=values+tuple(scales[0])+tuple(scales[1])
   a,b=cp.cuda.Event(),cp.cuda.Event();a.record();function(grid,block,values);b.record();b.synchronize();measure[kernel]=measure.get(kernel,0)+cp.cuda.get_elapsed_time(a,b)
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*v,**kw):
 result=best(self,*v,**kw);keys.append(result.get());return result
core.GPUSearch.best_keys=remember
class Done(Exception):pass
results=[]
def probe(self,periods,capacity,refine,pbar=None):
 global measure,keys,mode,scales
 starts=np.linspace(0,len(periods)-args.periods,5,dtype=int);baseline=None;s=self.primary
 maxz=float(cp.max(cp.abs(s.preparedGPU[:,0]*s.preparedGPU[:,1])));maxw=float(cp.max(s.preparedGPU[:,1]));signal=s.signalGPU.get();stride=s.screenPrefixStride;padded=(s.patchedDatasSize+stride-1)//stride*stride
 original=s.screenCertificates;scales=parameters(padded,maxz,maxw);delta=scales[2];converted=[]
 for tol in SCREEN_TOLERANCES:
  pos,coef,meta,mass=certificates(signal,s.durations,maxz,maxw,padded,tol,delta)
  pos,coef,meta=sparse_certificate(pos,coef,meta,stride,padded,*np.nextafter(np.array([maxz,maxw])+delta,np.inf),uniform)
  converted.append((cp.asarray(pos),cp.asarray(coef),cp.asarray(meta),*map(np.float64,mass)))
 for mode in [0,1]:
  s.screenCertificates=converted if mode else original;s.screenMapKey=None
  s.chi2_batch(np.asarray(periods)[starts[2]:starts[2]+args.periods],args.periods,False)
  measure={};keys=[];start_time=time.perf_counter()
  for start in starts:s.chi2_batch(np.asarray(periods)[start:start+args.periods],args.periods,False)
  if baseline is None:baseline=[x.copy() for x in keys]
  assert all(np.array_equal(x,y) for x,y in zip(keys,baseline)),mode
  target=sum(v for k,v in measure.items() if 'Candidates' in k or 'searchChi2' in k or k==name)
  record={'integer':bool(mode),'scales':[list(map(float,x)) for x in scales],'kernel_ms':dict(measure),'target_ms':target,'total_kernel_ms':sum(measure.values()),'elapsed_seconds':time.perf_counter()-start_time,'keys_bitwise_equal':True};results.append(record);(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(mode,round(target,3),round(record['total_kernel_ms'],3),round(measure[name],3),flush=True)
 raise Done
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
if not uniform:
 rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
