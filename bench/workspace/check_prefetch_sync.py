import os,sys,subprocess
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-a8281b40-72b0-f4a1-95ba-d0eeefb61ba4')
repo=Path(__file__).resolve().parents[2];exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization';sys.path.insert(0,str(repo/'src'))
import cupy as cp,numpy as np
from gputls import core
from gputls.screening import certificates
rng=np.random.default_rng(413);n=71;widths=np.array([513,601,907],np.int32);size=n+widths.max();signal=np.zeros((3,widths.max()),np.float32)
for i,width in enumerate(widths):
 x=np.linspace(-1,1,width)
 signal[i,:width]=rng.uniform(0,.5,width) if i==1 else .5*(1-x*x)*(1+.1*np.sin((4 if i==0 else 30)*x))
integers=rng.integers(-7,10,(3,size));weight=np.exp(rng.uniform(-3,3,(3,size))).astype(np.float32);z=(integers.astype(np.float32)*np.float32(2**-10))*weight;weighted=cp.asarray(np.stack([z,weight],axis=-1));prefix=cp.empty((3,size,4),cp.float64);module=core.load_module();module.get_function('scanScreening')((3,),(32,),(weighted,prefix,np.int32(size)))
pos,coef,meta,error=certificates(signal,widths,abs(z).max(),weight.max(),size,.05);print('terms:',meta[:,1].tolist(),flush=True)
base=cp.RawModule(code=subprocess.check_output(['git','show','6715468:src/gputls/kernels.cu'],cwd=repo,text=True),options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));candidate=cp.RawModule(code=core.KERNEL_SOURCE.read_text(),options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));lower=cp.empty((3,n*3),cp.float32);upper=cp.empty(3,cp.uint64)
values=(lower,upper,prefix,cp.asarray(np.cumsum(integers,axis=1)),np.float32(2**-10),cp.ones(3,cp.float32),cp.asarray(widths),cp.arange(3,dtype=cp.int32),cp.asarray([0,n,n*2,n*3],cp.int32),cp.ones(3,cp.int32),cp.asarray([1,8,32],cp.int32),cp.asarray([[1,1,1],[1,0,1],[0,0,0]],cp.bool_),cp.asarray(pos),cp.asarray(coef),cp.asarray(meta),*map(np.float64,error),np.int32(3),np.int32(3),np.int32(n),np.int32(size),np.float32(0))
for name in ['screenCandidates','refineCandidates']:
 expected=None
 for m in [base,candidate]:
  upper.fill(core.EMPTY_KEY);lower.fill(-cp.inf);m.get_function(name)((1,3),(256,),values);cp.cuda.runtime.deviceSynchronize();result=(lower.get().view(np.uint32),upper.get())
  if expected is None:expected=result
  else:assert all(np.array_equal(x,y) for x,y in zip(result,expected)),name
print('Divergent durations, empty rows and partial warps: bitwise equal.',flush=True)
