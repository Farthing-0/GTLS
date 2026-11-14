"""Force independent prefix loads ahead of the ordered FP64 FMA chain."""
import argparse,os,json,statistics,sys,subprocess,re
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--periods',type=int,default=629);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','6715468','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','6715468:'+name],cwd=repo))
sys.path.insert(0,str(snapshot))
import cupy as cp,numpy as np,gputls
from gputls import core
source=core.KERNEL_SOURCE.read_text();loader=core.load_module;variants={'baseline':loader().get_function('screenCandidates')}
helper='''__device__ __forceinline__ double2 forcePrefixLoad(const double2* p) {
    double2 v; asm volatile("ld.global.v2.f64 {%0,%1},[%2];" : "=d"(v.x), "=d"(v.y) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ double forceFma(double a, double b, double c) {
    double result; asm volatile("fma.rn.f64 %0,%1,%2,%3;" : "=d"(result) : "d"(a), "d"(b), "d"(c)); return result;
}
'''
for u,loads,fmas in [(4,False,False),(8,False,False),(13,False,False),(16,False,False)]:
 name=f'u{u}-load{int(loads)}-fma{int(fmas)}'
 dot=f'''int j = begin;
                    for (;j+{u}<=begin+terms;j+={u}) {{
                        double2 values[{u}], weights[{u}];
                        #pragma unroll
                        for (int q=0;q<{u};++q) {{
                            int k=start+positions[j+q]-2;
                            values[q]=k>=0?{'forcePrefixLoad(p+k)' if loads else 'p[k]'}:make_double2(0,0);
                            weights[q]={'forcePrefixLoad(coefficients+j+q)' if loads else 'coefficients[j+q]'};
                        }}
                        __syncwarp(__activemask());
                        #pragma unroll
                        for(int q=0;q<{u};++q) {{
                            ar={'forceFma' if fmas else 'fma'}(weights[q].x,values[q].x,ar);
                            a2={'forceFma' if fmas else 'fma'}(weights[q].y,values[q].y,a2);
                        }}
                    }}
                    for(;j<begin+terms;++j) {{
                        int k=start+positions[j]-2;
                        double2 v=k>=0?p[k]:make_double2(0,0),c=coefficients[j];
                        ar=fma(c.x,v.x,ar);a2=fma(c.y,v.y,a2);
                    }}'''
 pattern=r'int j = begin;\s*for \(; j\+4 <= begin\+terms; j \+= 4\) \{.*?\n\s*\}\s*for \(;j<begin\+terms;\+\+j\) \{.*?\n\s*\}'
 s,count=re.subn(pattern,dot,source,flags=re.S);assert count==2,count
 s=s.replace('#define SCREEN_ARGS',helper+'\n#define SCREEN_ARGS',1);(args.output/f'{name}.cu').write_text(s);m=cp.RawModule(code=s,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));variants[name]=m.get_function('screenCandidates')
class Done(Exception):pass
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  fun=self.module.get_function(name)
  if name!='screenCandidates':return fun
  def run(grid,block,values):
   lower,upper=values[:2]
   def call(name,timed=False):
    upper.fill(core.EMPTY_KEY)
    if timed:a,b=cp.cuda.Event(),cp.cuda.Event();a.record()
    variants[name](grid,block,values)
    if timed:b.record();b.synchronize();return cp.cuda.get_elapsed_time(a,b)
   call('baseline');old_lower=lower.copy();old_upper=upper.copy()
   for name in variants:
    call(name);assert bool(cp.all(lower.view(cp.uint32)==old_lower.view(cp.uint32))),name;assert bool(cp.all(upper==old_upper)),name
   del old_lower,old_upper
   for name in variants:call(name)
   times={name:[] for name in variants}
   for repeat in range(5):
    for name in (list(variants) if repeat%2==0 else list(reversed(variants))):times[name].append(call(name,True))
   medians={k:statistics.median(v) for k,v in times.items()};record={'periods':args.periods,'times_ms':times,'median_ms':medians,'speedups':{k:medians['baseline']/v for k,v in medians.items()},'bounds_bitwise_equal':True,'upper_keys_bitwise_equal':True,'registers':{k:v.attributes['num_regs'] for k,v in variants.items()}}
   (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='times_ms'},indent=2));raise Done
  return run
core.load_module=Module
def probe(self,periods,capacity,refine,pbar=None):
 batch=np.asarray(periods)[np.linspace(0,len(periods)-1,args.periods,dtype=int)];self.primary.chi2_batch(batch,len(batch),False)
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
rng=np.random.default_rng(20261114);dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
