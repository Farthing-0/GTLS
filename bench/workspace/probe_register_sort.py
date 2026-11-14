import os,argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',default='GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab');p.add_argument('--rows',type=int,default=752);p.add_argument('--short',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);os.environ['CUDA_VISIBLE_DEVICES']=args.device
import numpy as np,cupy as cp
repo=Path(__file__).resolve().parents[2];exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization';source=subprocess.check_output(['git','show','31b516d:src/gputls/kernels.cu'],cwd=repo,text=True)
helper=r'''
template<int N,bool Rank> __device__ __forceinline__ void registerBucket(int* points,const unsigned long long* phases,int count){
 unsigned long long key[N];int index[N];
 #pragma unroll
 for(int i=0;i<N;++i){index[i]=i<count?points[i]:0x7fffffff;key[i]=i<count?phases[index[i]]:0xffffffffffffffffull;}
 if(Rank){
  int rank[N]={0};
  #pragma unroll
  for(int i=0;i<N;++i){
   #pragma unroll
   for(int j=i+1;j<N;++j){bool greater=key[i]>key[j]||(key[i]==key[j]&&index[i]>index[j]);rank[i]+=greater;rank[j]+=!greater;}
  }
  #pragma unroll
  for(int i=0;i<N;++i)if(i<count)points[rank[i]]=index[i];
 }else{
  #pragma unroll
  for(int k=2;k<=N;k<<=1){
   #pragma unroll
   for(int j=k>>1;j>0;j>>=1){
    #pragma unroll
    for(int i=0;i<N;++i){int other=i^j;if(other>i){
     bool greater=key[i]>key[other]||(key[i]==key[other]&&index[i]>index[other]);
     if((i&k)==0?greater:!greater){auto tmp=key[i];key[i]=key[other];key[other]=tmp;int t=index[i];index[i]=index[other];index[other]=t;}
    }}
   }
  }
  #pragma unroll
  for(int i=0;i<N;++i)if(i<count)points[i]=index[i];
 }
}
'''
a=source.index('    __global__ void sortBuckets(');b=source.index('    /**',a);integer=source[a:b].replace('const double* rowPhase = phase + rowOffset;','const unsigned long long* rowPhase = (const unsigned long long*)(phase + rowOffset);').replace('double keyPhase = rowPhase[keyPoint];','unsigned long long keyPhase = rowPhase[keyPoint];');extra=helper
for name,n,rank in [('integer',0,False),('network4',4,False),('network8',8,False),('rank4',4,True),('rank8',8,True),('network16',16,False)]:
 part=integer.replace('sortBuckets(',name+'(')
 if n:
  point='        for (int next = 1; next < bucketSize; next++) {'
  part=part.replace(point,f'        if(bucketSize<={n}){{registerBucket<{n},{str(rank).lower()}>(bucketPoints,rowPhase,bucketSize);return;}}\n'+point)
 extra+='extern "C" {\n'+part+'}\n'
source+=extra;(args.output/'source.cu').write_text(source);mod=cp.RawModule(code=source,options=('-DGTLS_CHEAP_SCREEN_BOUNDS=0',));names=['foldBuckets','scanCounts','bucketScatter','sortBuckets','integer','network4','network8','rank4','rank8','network16'];f={k:mod.get_function(k) for k in names}
with np.load(exp/'postprocess-cpu/input.npz') as a:t=a['t']
with np.load(exp/'uniform-730-constant/run/fields.npz') as a:periods=a['periods']
n=len(t);rows=args.rows;gput=cp.asarray(t);phase=cp.empty((rows,n),cp.float64);index=cp.empty((rows,n),cp.int32);ref=cp.empty_like(index);storage=cp.empty(rows*((n+2)//3),cp.int32);storage2=cp.empty_like(storage);res=[]
for offset in [0,16000,40000]:
 inv=cp.asarray(1./periods[offset:offset+rows]);variants=[(3,x) for x in ['sortBuckets','integer','network4','network8','rank4','rank8','network16']]+[(x,'network8') for x in [4,6,8]]+[(4,'rank8'),(6,'rank8')]
 if args.short:variants=[(3,'sortBuckets'),(3,'integer'),(3,'network8'),(4,'network8'),(4,'rank8'),(6,'rank8')]
 def run(factor,kind):
  buckets=(n+factor-1)//factor;count=storage[:rows*buckets].reshape(rows,buckets);cursor=storage2[:rows*buckets].reshape(rows,buckets);stages={}
  def timed(name,fn):
   a,b=cp.cuda.Event(),cp.cuda.Event();a.record();fn();b.record();b.synchronize();stages[name]=cp.cuda.get_elapsed_time(a,b)
  timed('zero',lambda:count.fill(0));timed('fold',lambda:f['foldBuckets'](((n+255)//256,rows),(256,),(gput,inv,phase,count,np.int32(n),np.int32(buckets))))
  timed('scan',lambda:f['scanCounts']((rows,),(512,),(count,cursor,np.int32(buckets))))
  timed('scatter',lambda:f['bucketScatter'](((n+255)//256,rows),(256,),(phase,cursor,index,np.int32(n),np.int32(buckets))))
  timed('sort',lambda:f[kind](((buckets+255)//256,rows),(256,),(phase,count,cursor,index,np.int32(n),np.int32(buckets))))
  return stages
 for factor,kind in variants:
  run(factor,kind)
  if (factor,kind)==(3,'sortBuckets'):cp.copyto(ref,index)
  else:assert bool(cp.all(ref==index)),(offset,factor,kind)
  samples=[run(factor,kind) for _ in range(3)];med={k:float(np.median([x[k] for x in samples])) for k in samples[0]};row={'offset':offset,'factor':factor,'kind':kind,'median_ms':med,'total_ms':sum(med.values()),'indices_bitwise_equal':True,'samples':samples};res.append(row);print(offset,factor,kind,med['sort'],row['total_ms'],flush=True)
 (args.output/'results.json').write_text(json.dumps({'rows':rows,'device':args.device,'results':res,'attributes':{k:f[k].attributes for k in names}},indent=2)+'\n')
