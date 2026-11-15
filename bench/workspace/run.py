"""Frozen-source correctness, synchronized timing and nsys capture for workspace work."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2'
parser=argparse.ArgumentParser()
parser.add_argument('--case',default='0730d_100s')
parser.add_argument('--pattern',choices=['constant','smooth','random','spikes','mixed'],default='constant')
parser.add_argument('--baseline',action='store_true')
parser.add_argument('--batch-cap',type=int)
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--profile',action='store_true')
parser.add_argument('--diagnose-memory',action='store_true')
parser.add_argument('--source',type=Path,help='Explicit frozen gputls source directory')
parser.add_argument('--experiment',type=Path,default=EXPERIMENT)
parser.add_argument('--device',help='CUDA_VISIBLE_DEVICES override')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
EXPERIMENT=args.experiment.resolve()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise FileExistsError(out/'result.json')
p=json.loads((EXPERIMENT/'protocol.json').read_text())
c=next(c for c in p['cases'] if c['id']==args.case)
device=args.device or os.environ.get('CUDA_VISIBLE_DEVICES',p['gpu0_uuid'])
os.environ.update(CUDA_VISIBLE_DEVICES=device,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                  CUPY_CACHE_DIR=str(EXPERIMENT/'workspace-optimization/cache/cupy'),
                  NUMBA_CACHE_DIR=str(EXPERIMENT/'workspace-optimization/cache/numba'))
source=args.source if args.source is not None else (EXPERIMENT/'sources/gtls/gputls' if args.baseline else REPO/'src/gputls')
snapshot=out/'source/gputls'
shutil.copytree(source,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
sys.path.insert(0,str(snapshot.parent))
import numpy as np
import cupy as cp
import gputls
from gputls import core

if args.profile:
    original_chi2=core.MultiGPUSearch.chi2
    original_fit=core.search_single_period
    def traced_chi2(self, periods, capacity, refine, pbar=None):
        cp.cuda.nvtx.RangePush('GTLS_refine' if refine else 'GTLS_coarse')
        try:return original_chi2(self,periods,capacity,refine,pbar)
        finally:cp.cuda.nvtx.RangePop()
    def traced_fit(*values,**kwargs):
        cp.cuda.nvtx.RangePush('GTLS_final_fit')
        try:return original_fit(*values,**kwargs)
        finally:cp.cuda.nvtx.RangePop()
    core.MultiGPUSearch.chi2=traced_chi2
    core.search_single_period=traced_fit

spec=importlib.util.spec_from_file_location('_campaign_monitor',EXPERIMENT/'benchmark.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


class SelectedDeviceMonitor(campaign.Monitor):
    """NVML uses physical indices; CUDA_VISIBLE_DEVICES may expose GPU1 as device 0."""
    def __enter__(self):
        self.rows=[];self.stop=threading.Event();self.lib=ct.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2()==0
        self.handle=ct.c_void_p()
        bus=cp.cuda.runtime.deviceGetPCIBusId(0)
        assert self.lib.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(),ct.byref(self.handle))==0
        self.sample()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        return self


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name,data):
    path=out/name;tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def fields(answer):
    values={}
    for key,value in vars(answer).items():
        values[key]=np.asarray(np.ma.getdata(value))
        if np.ma.isMaskedArray(value):values[key+'__mask']=np.ma.getmaskarray(value)
    return values


result=dict(case=c,pattern=args.pattern,baseline=args.baseline,gtls_version=gputls.__version__,
            cupy_version=cp.__version__,gpu=device,source_sha256={str(f.relative_to(snapshot)):digest(f)
            for f in sorted(snapshot.rglob('*')) if f.is_file() and '__pycache__' not in f.parts},
            branch=subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip(),
            commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True),
            runner_sha256=digest(__file__),profiled=args.profile,phase='initializing')
write('result.json',result)
try:
    assert digest(EXPERIMENT/c['input'])==c['input_sha256']
    with np.load(EXPERIMENT/c['input']) as data:
        t,y,dy=(data[k][:c['n_points']].copy() for k in ('t','y','dy'))
    assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in zip(('t','y','dy'),(t,y,dy))}==c['arrays_sha256']
    rng=np.random.default_rng(20261114)
    if args.pattern=='smooth':dy*=.25+1.5*(1+np.sin(np.arange(len(t))*.003))/2
    elif args.pattern=='random':dy*=np.exp(rng.normal(0,.6,len(t)))
    elif args.pattern=='mixed':dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
    elif args.pattern=='spikes':
        dy*=np.where(np.arange(len(t))//1000%2,.4,2.)
        dy[np.arange(23,len(t),max(101,len(t)//11))]=1e-7
    cap=args.batch_cap if args.batch_cap is not None else (256 if args.baseline else None)
    if cap is not None:core.MAX_BATCH=cap
    result.update(batch_cap=cap,input_sha256={k:hashlib.sha256(v.tobytes()).hexdigest()
                                           for k,v in zip(('t','y','dy'),(t,y,dy))},
                  settings=p['settings']['gtls'],dy_min=float(dy.min()),dy_max=float(dy.max()))
    settings=p['settings']['gtls'].copy()
    def call(small=False):
        mask=t<t[0]+30 if small else slice(None)
        kw=settings|dict(period_min=1.,period_max=2.,oversampling_factor=1) if small else settings
        return gputls.gtls(t[mask],y[mask],dy[mask],verbose=False).power(**kw)
    start=time.perf_counter();call(True);cp.cuda.runtime.deviceSynchronize()
    result.update(initialization_seconds=time.perf_counter()-start,phase='full_warmup');write('result.json',result)
    start=time.perf_counter();call();cp.cuda.runtime.deviceSynchronize()
    result.update(warmup_seconds=time.perf_counter()-start,phase='measuring');write('result.json',result)
    times=[];states=[];expected=None
    for repeat in range(1 if args.profile else args.repeats):
        cp.cuda.runtime.deviceSynchronize()
        with SelectedDeviceMonitor() as mon:
            if args.profile:cp.cuda.profiler.start()
            cp.cuda.nvtx.RangePush('GTLS_full_power')
            start=time.perf_counter();answer=call();cp.cuda.runtime.deviceSynchronize()
            elapsed=time.perf_counter()-start
            cp.cuda.nvtx.RangePop()
            if args.profile:cp.cuda.profiler.stop()
        actual=fields(answer)
        if expected is not None:
            assert all(actual[k].dtype==v.dtype and actual[k].shape==v.shape and actual[k].tobytes()==v.tobytes()
                       for k,v in expected.items()),'Nondeterministic result across repeats'
        expected=actual;times.append(elapsed);states.extend(mon.rows)
        result.update(times_seconds=times,phase='measuring');write('result.json',result)
        print(f'{args.case} {args.pattern} repeat {repeat+1}: {elapsed:.6f}s',flush=True)
    np.savez_compressed(out/'fields.npz',**expected)
    assert np.array_equal(np.sort(np.asarray(answer.periods)),np.sort(np.load(EXPERIMENT/c['grid'])))
    write('telemetry.json',states)
    result.update(ok=True,phase='complete',median_seconds=statistics.median(times),
                  field_sha256=digest(out/'fields.npz'),period=float(answer.period),sde=float(answer.SDE),
                  recovered=bool(abs(answer.period/p['injection']['period_days']-1)<.01),
                  peak_device_bytes=max(v['used_bytes'] for v in states),
                  thermal_throttled=any(v['throttle_bits']&0x60 for v in states),
                  pool_used_after=cp.get_default_memory_pool().used_bytes(),
                  pool_total_after=cp.get_default_memory_pool().total_bytes())
    if args.diagnose_memory and result['pool_used_after']:
        # Preserve the observed post-search bytes; GC diagnostics happen after timing.
        import gc
        import types
        live = []
        for obj in gc.get_objects():
            if type(obj) is cp.ndarray:
                refs = gc.get_referrers(obj)
                owners = []
                for ref in refs:
                    if type(ref) is types.FrameType:
                        owners.append(ref.f_code.co_filename+':'+ref.f_code.co_name)
                    else:
                        owners.append(type(ref).__name__)
                live.append(dict(shape=list(obj.shape), dtype=str(obj.dtype),
                                 nbytes=obj.nbytes, owners=owners))
                if refs:
                    del ref
                del refs, owners
        del obj
        result['memory_diagnostic'] = dict(arrays=live, collected=gc.collect())
        cp.get_default_memory_pool().free_all_blocks()
        result['memory_diagnostic'].update(
            used_after_gc=cp.get_default_memory_pool().used_bytes(),
            total_after_gc=cp.get_default_memory_pool().total_bytes())
except Exception:
    result.update(ok=False,phase='error',error=traceback.format_exc())
    print(result['error'],flush=True)
write('result.json',result)
raise SystemExit(0 if result.get('ok') else 1)
