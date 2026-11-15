import os,argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');p.add_argument('--rows',type=int,default=752);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);os.environ['CUDA_VISIBLE_DEVICES']=args.device
import numpy as np,cupy as cp
repo=Path(__file__).resolve().parents[2];exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization';source=subprocess.check_output(['git','show','6d433f5:src/gputls/kernels.cu'],cwd=repo,text=True)
helper=r'''
__device__ __forceinline__ double phaseFraction(double cycles){
 unsigned long long bits=__double_as_longlong(cycles);int exponent=((bits>>52)&2047)-1023;
 if((bits>>63)||exponent>=31)return cycles-(int)cycles;
 if(exponent<0)return cycles;
 int shift=52-exponent;unsigned long long fraction=bits&((1ull<<shift)-1);
 if(!fraction)return 0.;
 int lead=63-__clzll(fraction);
 return __longlong_as_double(((unsigned long long)(lead-shift+1023)<<52)|((fraction<<(52-lead))&0xfffffffffffffull));
}
extern "C" __global__ void checkFraction(const double* x,double* out,double* expected,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){double v=x[i];out[i]=phaseFraction(v);expected[i]=v-(int)v;}
}
'''
a=source.index('    __global__ void foldBuckets(');b=source.index('    /**',a);candidate=source[a:b].replace('foldBuckets(', 'foldBucketsBits(').replace('cycles - (int)(cycles)','phaseFraction(cycles)')
s=source+'\n'+helper+'\nextern "C" {\n'+candidate+'}\n';(args.output/'candidate.cu').write_text(s);module=cp.RawModule(code=s,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));original=module.get_function('foldBuckets');optimized=module.get_function('foldBucketsBits')
rng=np.random.default_rng(42389);exponents=rng.integers(-1070,1022,50000);x=np.ldexp(rng.uniform(.5,1,50000),exponents);x=np.concatenate((x,-x,np.array([0.,-0.,1.,2.,3.,np.nextafter(1.,0.),np.nextafter(1.,2.),2.**31-1,2.**31,2.**52,np.inf,-np.inf])))
out,expected=cp.empty(x.size,cp.float64),cp.empty(x.size,cp.float64);module.get_function('checkFraction')(((x.size+255)//256,),(256,),(cp.asarray(x),out,expected,np.int32(x.size)));assert bool(cp.all(out.view(cp.uint64)==expected.view(cp.uint64)))
print('fraction values bitwise equal',x.size,flush=True)
with np.load(exp/'postprocess-cpu/input.npz') as a:t=a['t']
with np.load(exp/'uniform-730-constant/run/fields.npz') as a:periods=a['periods']
n=len(t);rows=args.rows;buckets=(n+2)//3;tg=cp.asarray(t);phases=[cp.empty((rows,n),cp.float64) for _ in range(2)];counts=[cp.empty((rows,buckets),cp.int32) for _ in range(2)];results=[]
for offset in [0,16000,40000]:
 inv=cp.asarray(1./periods[offset:offset+rows]);samples=[[],[]]
 for repeat in range(5):
  for k in ([0,1] if repeat%2==0 else [1,0]):
   counts[k].fill(0);a,b=cp.cuda.Event(),cp.cuda.Event();a.record();[original,optimized][k](((n+255)//256,rows),(256,),(tg,inv,phases[k],counts[k],np.int32(n),np.int32(buckets)));b.record();b.synchronize()
   if repeat:samples[k].append(cp.cuda.get_elapsed_time(a,b))
 assert bool(cp.all(phases[0].view(cp.uint64)==phases[1].view(cp.uint64)));assert bool(cp.all(counts[0]==counts[1]))
 r={'offset':offset,'phase_bits_equal':True,'counts_equal':True,'median_ms':[float(np.median(x)) for x in samples],'samples':samples};results.append(r);print(r['offset'],r['median_ms'],flush=True)
(args.output/'results.json').write_text(json.dumps({'device':args.device,'rows':rows,'values_checked':int(x.size),'results':results,'attributes':{'original':original.attributes,'candidate':optimized.attributes}},indent=2)+'\n')
