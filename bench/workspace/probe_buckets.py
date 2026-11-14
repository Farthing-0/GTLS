"""Tune histogram bucket count and register sorting networks."""
import argparse,os,json,statistics,sys,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--periods',type=int,default=256);parser.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[2];root=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2';exp=root/'workspace-optimization'
os.environ.update(CUDA_VISIBLE_DEVICES=args.device,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='1',CUPY_CACHE_DIR=str(exp/'cache/cupy'),NUMBA_CACHE_DIR=str(exp/'cache/numba'))
snapshot=args.output/'source'
for name in subprocess.check_output(['git','ls-tree','-r','--name-only','6715468','src/gputls'],cwd=repo,text=True).splitlines():
 target=snapshot/Path(name).relative_to('src');target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(subprocess.check_output(['git','show','6715468:'+name],cwd=repo))
sys.path.insert(0,str(snapshot))
import cupy as cp,numpy as np,gputls
from gputls import core
source=core.KERNEL_SOURCE.read_text();loader=core.load_module;original=loader();generic=source
for name in ['foldBuckets','bucketScatter','sortBuckets']:
 a=generic.index('    __global__ void '+name+'(');b=generic.index('\n    /**',a);piece=generic[a:b];piece=piece.replace('int tSize) {','int tSize, int nBuckets) {')
 if name=='foldBuckets':
  piece=piece.replace('pointPhase * tSize), tSize - 1','pointPhase * nBuckets), nBuckets - 1').replace('counts + rowOffset + bucket','counts + (long long)periodIndex*nBuckets + bucket')
 elif name=='bucketScatter':
  piece=piece.replace('pointIndex] * tSize), tSize - 1','pointIndex] * nBuckets), nBuckets - 1').replace('cursor + rowOffset + bucket','cursor + (long long)periodIndex*nBuckets + bucket')
 else:
  piece=piece.replace('bucket >= tSize','bucket >= nBuckets').replace('counts[rowOffset + bucket]','counts[(long long)periodIndex*nBuckets + bucket]').replace('bucketEnds[rowOffset + bucket]','bucketEnds[(long long)periodIndex*nBuckets + bucket]')
 generic=generic[:a]+piece+generic[b:]
network='''template<int N> __device__ __forceinline__ void registerSort(int* points,const double* phase,int count){
    double keys[N];int indices[N];
    #pragma unroll
    for(int i=0;i<N;++i){indices[i]=i<count?points[i]:0x7fffffff;keys[i]=i<count?phase[indices[i]]:__longlong_as_double(0x7ff0000000000000ull);}
    #pragma unroll
    for(int k=2;k<=N;k<<=1){
        #pragma unroll
        for(int j=k>>1;j>0;j>>=1){
            #pragma unroll
            for(int i=0;i<N;++i){int other=i^j;if(other>i){
                bool greater=keys[i]>keys[other] || (keys[i]==keys[other] && indices[i]>indices[other]);
                if((i&k)==0?greater:!greater){double t=keys[i];keys[i]=keys[other];keys[other]=t;int u=indices[i];indices[i]=indices[other];indices[other]=u;}
            }}
        }
    }
    #pragma unroll
    for(int i=0;i<N;++i)if(i<count)points[i]=indices[i];
}
'''
modules={0:cp.RawModule(code=generic,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',))}
for size in []:
 s=network+generic;s=s.replace('        const double* rowPhase = phase + rowOffset;',f'        const double* rowPhase = phase + rowOffset;\n        if(bucketSize<={size}){{registerSort<{size}>(bucketPoints,rowPhase,bucketSize);return;}}',1)
 modules[size]=cp.RawModule(code=s,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));(args.output/f'network-{size}.cu').write_text(s)
(args.output/'generic.cu').write_text(generic)
variants={'original':(original,1,False,0)}
for factor in [2,3,4,5,6]:
 for size,m in modules.items():variants[f'div{factor}-net{size}']=(m,factor,True,size)
functions={k:tuple(m.get_function(name) for name in ['foldBuckets','scanCounts','bucketScatter','sortBuckets']) for k,(m,_,_,_) in variants.items()}
active_search=None
class Done(Exception):pass
class Module:
 def __init__(self):self.module=loader()
 def get_function(self,name):
  fun=self.module.get_function(name)
  if name!='foldBuckets':return fun
  def run(grid,block,values):
   size=int(values[-1]);periods=grid[1];w=active_search.workspace;phase=values[2];index=w.view('scratch',(periods,size),cp.int32)
   def pipeline(name,timed=False):
    _,factor,is_generic,_=variants[name];buckets=(size+factor-1)//factor
    counts=w.view('scratch',(periods,buckets),cp.int32,3*w.index_bytes);cursor=w.view('scratch',(periods,buckets),cp.int32,3*w.index_bytes+periods*buckets*4)
    fold,scan,scatter,sort=functions[name];extra=(np.int32(buckets),) if is_generic else ()
    if timed:events=[cp.cuda.Event() for _ in range(6)];events[0].record()
    counts.fill(0)
    if timed:events[1].record()
    fold(grid,block,values[:3]+(counts,values[-1])+extra)
    if timed:events[2].record()
    scan((periods,),(core.SCAN_THREADS,),(counts,cursor,np.int32(buckets)))
    if timed:events[3].record()
    scatter(grid,block,(phase,cursor,index,np.int32(size))+extra)
    if timed:events[4].record()
    sort(((buckets+255)//256,periods),block,(phase,counts,cursor,index,np.int32(size))+extra)
    if timed:
     events[5].record();events[5].synchronize();return {key:cp.cuda.get_elapsed_time(events[i],events[i+1]) for i,key in enumerate(['zero','fold','scan','scatter','sort'])}
    return counts
   pipeline('original');reference=index.get();stats={}
   for name in variants:
    counts=pipeline(name);assert np.array_equal(index.get(),reference),name
    if name.endswith('net0'):stats[name]={'max_count':int(counts.max()),'mean_count':size/counts.shape[1]}
   del reference
   for name in variants:pipeline(name)
   times={name:[] for name in variants}
   for repeat in range(4):
    for name in (list(variants) if repeat%2==0 else list(reversed(variants))):times[name].append(pipeline(name,True))
   medians={name:{key:statistics.median(row[key] for row in rows) for key in rows[0]} for name,rows in times.items()}
   for row in medians.values():row['total']=sum(row.values())
   result={'periods':periods,'size':size,'bucket_stats':stats,'sorted_indices_bitwise_equal':True,'median_ms':medians,'times_ms':times,'speedups':{name:medians['original']['total']/v['total'] for name,v in medians.items()},'registers':{k:v[-1].attributes['num_regs'] for k,v in functions.items()}}
   (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['times_ms','median_ms']},indent=2));print('totals',json.dumps({k:round(v['total'],3) for k,v in medians.items()}));raise Done
  return run
core.load_module=Module
best=core.GPUSearch.best_keys
def remember(self,*values,**kwargs):
 global active_search
 active_search=self;return best(self,*values,**kwargs)
core.GPUSearch.best_keys=remember
def probe(self,periods,capacity,refine,pbar=None):
 batch=np.asarray(periods)[np.linspace(0,len(periods)-1,args.periods,dtype=int)];self.primary.chi2_batch(batch,len(batch),False)
core.MultiGPUSearch.chi2=probe
p=json.loads((root/'protocol.json').read_text());case=next(x for x in p['cases'] if x['id']=='0730d_100s')
with np.load(root/case['input']) as data:t,y,dy=(data[k][:case['n_points']].copy() for k in ['t','y','dy'])
try:gputls.gtls(t,y,dy,verbose=False).power(**p['settings']['gtls'])
except Done:pass
