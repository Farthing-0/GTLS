"""Reproducible heteroscedastic end-to-end comparisons and profiler capture ranges.

The baseline is the hash-verified first-round commit 50cb1db.
Inputs are shared with the original campaign. Timings include native
input validation, template/grid construction, GPU transfer, search and fit.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXPERIMENT = Path('/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114')
OUTPUT = EXPERIMENT/'round2-candidate-screening-20261114'
DEVICE = os.environ.get('CUDA_VISIBLE_DEVICES', 'GPU-f0f7673f-e676-9211-a9f6-68bcd9dd2aab')
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  NUMBA_NUM_THREADS='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUDA_VISIBLE_DEVICES=DEVICE)
os.environ.setdefault('CUPY_CACHE_DIR', str(OUTPUT/'cache/cupy'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(OUTPUT/'cache/numba'))
sys.path.insert(0, str(REPO/'src'))
import cupy as cp
import numpy as np
import gputls
from gputls import core, main as gtls_main

monitor_spec = importlib.util.spec_from_file_location(
    '_round1_monitor', EXPERIMENT/'bottleneck-analysis-20261114/profile_local.py')
monitor_module = importlib.util.module_from_spec(monitor_spec)
monitor_spec.loader.exec_module(monitor_module)
Monitor = monitor_module.Monitor
Monitor.device_uuid = DEVICE


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def baseline_module(protocol):
    source = OUTPUT/'baseline/gputls'
    expected = {'core.py': '598602a52b0a61ed3ffd48c432e47e9be30762f2041a4981055f0c06152c7197',
                'kernels.cu': '7177bb4337da7ec9ad99926928e1040a4f1fa678294fd7f37d28c3e3f1425f8b'}
    for name, sha in expected.items():
        assert digest((source/name).read_bytes()) == sha, name
    spec = importlib.util.spec_from_file_location('gputls._window_baseline', source/'core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(protocol, case_id, pattern):
    case = next(c for c in protocol['cases'] if c['id'] == case_id)
    path = EXPERIMENT/case['input']
    assert digest(path.read_bytes()) == case['input_sha256']
    with np.load(path) as data:
        t, y, dy = (data[k][:case['n_points']].copy() for k in ('t', 'y', 'dy'))
    rng = np.random.default_rng(20261114)
    if pattern == 'smooth':
        factor = .25 + 1.5*(1+np.sin(np.arange(len(t))*.003))/2
    elif pattern == 'random':
        factor = np.exp(rng.normal(0, .6, len(t)))
    elif pattern == 'mixed':
        factor = np.exp(rng.normal(0, .6, len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
    elif pattern == 'spikes':
        factor = np.where(np.arange(len(t))//1000 % 2, .4, 2.0).astype(float)
        factor[np.arange(23, len(t), max(101,len(t)//11))] = 1e-7
    else:
        raise ValueError(pattern)
    dy *= factor
    return (t,y,dy), dict(case=case_id, pattern=pattern, n_points=len(t),
                         sha256={k:digest(a.tobytes()) for k,a in zip(('t','y','dy'),(t,y,dy))},
                         dy_min=float(dy.min()),dy_max=float(dy.max()))


def fields(result):
    data = {}
    for name,value in vars(result).items():
        data[name] = np.asarray(np.ma.getdata(value))
        if np.ma.isMaskedArray(value):
            data[name+'__mask'] = np.ma.getmaskarray(value)
    return data


def compare(expected, actual):
    assert expected.keys() == actual.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)
    return {name:True for name in expected}


@contextmanager
def region(label):
    cp.cuda.nvtx.RangePush(label)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


class ProbeComplete(Exception):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['compare','profile','probe'],default='compare')
    parser.add_argument('--variant',choices=['baseline','optimized'],default='optimized')
    parser.add_argument('--case',default='0365d_100s')
    parser.add_argument('--pattern',choices=['smooth','random','mixed','spikes'],default='mixed')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--batch-index',type=int,default=15)
    parser.add_argument('--probe-size',type=int,default=815)
    parser.add_argument('--cool-to',type=int,default=None,
                        help='Wait outside measured/captured ranges until GPU temperature is at most this value.')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((EXPERIMENT/'protocol.json').read_text())
    baseline=baseline_module(protocol)
    arrays,manifest=inputs(protocol,args.case,args.pattern)
    np.savez(args.output/'input.npz',t=arrays[0],y=arrays[1],dy=arrays[2])
    settings=protocol['settings']['gtls'].copy()
    result=dict(baseline_commit='50cb1db',input=manifest,settings=settings,device=DEVICE,cool_to_c=args.cool_to,
                device_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                cupy=cp.__version__,nvrtc=cp.cuda.nvrtc.getVersion(),
                source_sha256={name:digest((REPO/'src/gputls'/name).read_bytes())
                               for name in ('core.py','kernels.cu','screening.py')},runs=[],warmup=[])

    def select(variant):
        if variant == 'optimized':
            gtls_main.core=core
        else:
            gtls_main.core=baseline

    def cool():
        if args.cool_to is None:
            return
        with Monitor() as monitor:
            if monitor.rows[-1]['temperature_c'] > args.cool_to:
                print('cooling',DEVICE,'to',args.cool_to,'C',flush=True)
            deadline=time.monotonic()+300
            while monitor.rows[-1]['temperature_c'] > args.cool_to:
                if time.monotonic() > deadline:
                    raise RuntimeError('GPU did not reach the requested cooldown temperature in 300 seconds')
                time.sleep(1)

    def run(variant):
        if args.mode == 'compare':
            cool()
        select(variant)
        with Monitor() as monitor:
            cp.cuda.runtime.deviceSynchronize()
            start=time.perf_counter()
            answer=gputls.gtls(*arrays,verbose=False).power(**settings)
            cp.cuda.runtime.deviceSynchronize()
            elapsed=time.perf_counter()-start
        result['last_telemetry'] = monitor.rows
        return elapsed,fields(answer)

    if args.mode=='compare':
        reference=None
        variants=['baseline','optimized']
        for variant in variants:
            elapsed,data=run(variant)
            result['warmup'].append(dict(variant=variant,seconds=elapsed))
            if reference is None: reference=data
            result['all_fields_equal']=compare(reference,data)
            np.savez(args.output/(variant+'.npz'),**data)
            print('warmup',variant,elapsed,flush=True)
        for repeat in range(args.repeats):
            for variant in (variants if repeat%2==0 else variants[::-1]):
                elapsed,data=run(variant)
                compare(reference,data)
                result['runs'].append(dict(repeat=repeat,variant=variant,seconds=elapsed,
                                          telemetry=result.pop('last_telemetry')))
                write(args.output/'result.json',result)
                print('measure',repeat,variant,elapsed,'all fields exact',flush=True)
        result['median_seconds']={v:float(np.median([r['seconds'] for r in result['runs'] if r['variant']==v]))
                                  for v in variants}
        result['speedup']=result['median_seconds']['baseline']/result['median_seconds']['optimized']
    elif args.mode=='profile':
        elapsed,reference=run(args.variant)
        result['warmup'].append(dict(variant=args.variant,seconds=elapsed))
        cool()
        cp.cuda.profiler.start()
        try:
            with region('GTLS_round2_'+args.variant):
                elapsed,data=run(args.variant)
        finally:
            cp.cuda.profiler.stop()
        result['traced_seconds']=elapsed
        result['all_fields_equal']=compare(reference,data)
        np.savez(args.output/'periodogram.npz',**data)
    else:
        select(args.variant)
        module=gtls_main.core

        def probe(self,periods,size,refine,pbar=None):
            search=self.primary
            batch=np.asarray(periods)[args.batch_index*815:args.batch_index*815+args.probe_size]
            result.update(batch_index=args.batch_index,size=len(batch),
                          period_range=[float(batch.min()),float(batch.max())])
            expected=search.chi2_batch(batch,len(batch),False)
            cp.cuda.runtime.deviceSynchronize()
            cp.cuda.profiler.start()
            try:
                with region('GTLS_batch_'+args.variant):
                    actual=search.chi2_batch(batch,len(batch),False)
                    cp.cuda.runtime.deviceSynchronize()
            finally:
                cp.cuda.profiler.stop()
            np.testing.assert_array_equal(actual,expected)
            result['batch_chi2_exact']=True
            raise ProbeComplete

        module.MultiGPUSearch.chi2=probe
        try: gputls.gtls(*arrays,verbose=False).power(**settings)
        except ProbeComplete: pass
    write(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k in ('median_seconds','speedup','batch_chi2_exact','traced_seconds')}),flush=True)


if __name__=='__main__': main()
