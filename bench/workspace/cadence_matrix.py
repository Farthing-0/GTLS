"""Frozen GTLS main / native CETRA cadence benchmark with varying dy."""
import argparse
import csv
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.parents[1]
DEFAULT_OUT = ROOT / 'a100-cetra-matrix-20261115/data'
DAYS = (30, 90, 365, 730, 1460)
CADENCES = (25, 50, 100, 200, 400, 600)
VERSIONS = {'gtls': '2.2.1', 'cetra': '1.04'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def prepare(out):
    experiment = ROOT / 'GTLS-Compare/tls2-gtls-delta001-20261114v2'
    base = read(experiment/'protocol.json')
    assert subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip() == 'main'
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    if (out/'protocol.json').exists():
        raise FileExistsError(out/'protocol.json')
    out.mkdir(parents=True,exist_ok=True)
    sources = {}
    for engine, src in [('gtls',REPO/'src/gputls'),('cetra',ROOT/'cetra/src/cetra')]:
        dst=out/'sources'/engine/src.name
        shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        sources[engine]=source_hashes(dst)
    cases=[]
    for cadence in CADENCES:
        original=experiment/'inputs'/f'cadence_{cadence:03d}s.npz'
        reference=next(c for c in base['cases'] if c['days']==1460 and c['cadence_seconds']==cadence)
        assert digest(original)==reference['input_sha256']
        with np.load(original) as data:t,y,dy=(data[k].copy() for k in ('t','y','dy'))
        assert len(t)==reference['n_points']
        rng=np.random.default_rng(20261114)
        dy*=np.exp(rng.normal(0,.6,len(t)))*(1+.4*np.sin(np.arange(len(t))*.003))
        path=out/'inputs'/original.name
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,t=t,y=y,dy=dy)
        file_hash=digest(path)
        for days in DAYS:
            n=int(days*86400/cadence)
            cases.append(dict(id=f'{days:04d}d_{cadence:03d}s',days=days,cadence_seconds=cadence,
                n_points=n,span_days=float(t[n-1]-t[0]),file=str(path.relative_to(out)),sha256=file_hash,
                arrays_sha256={k:array_hash(v[:n]) for k,v in zip(('t','y','dy'),(t,y,dy))}))
        print(f'Prepared cadence={cadence}s',flush=True)
    cases.sort(key=lambda c:(c['days'],-c['cadence_seconds']))
    known=read(ROOT/'optimization-20261115/local-initial/result.json')['input_sha256']
    assert next(c for c in cases if c['id']=='0730d_100s')['arrays_sha256']==known
    protocol=dict(created_utc=now(),versions=VERSIONS,physical_gpu=0,
        gtls_branch='main',gtls_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        cetra_commit=subprocess.check_output(['git','-C',str(ROOT/'cetra'),'rev-parse','HEAD'],text=True).strip(),
        repeats=3,days=list(DAYS),cadences_seconds=list(CADENCES),cases=cases,
        injection=base['injection'],noise=base['noise'],dy_pattern='mixed',
        dy_formula='250e-6*exp(default_rng(20261114).normal(0,0.6,N))*(1+0.4*sin(arange(N)*0.003))',
        search=base['settings']['gtls'],
        cetra=dict(min_duration=.02,max_duration=.5,duration_log_step=1.1,t0_stride_fraction=.1),
        grid_policy='Native automatic period grid per engine; periods argument omitted.',
        timing='Synchronized complete API wall time; small + full warmup excluded; median of three.',
        sources=sources,runner_sha256=digest(__file__))
    shutil.copy2(__file__,out/'benchmark_frozen.py')
    write(out/'protocol.json',protocol)
    write(out/'status.json',dict(phase='prepared',completed_jobs=0,total_jobs=60,updated_utc=now()))


class Monitor:
    class Memory(ct.Structure):
        _fields_ = [('total',ct.c_ulonglong),('free',ct.c_ulonglong),('used',ct.c_ulonglong)]

    def __init__(self):
        self.lib = ct.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2() == 0
        self.handle = ct.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(int(os.environ.get('GTLS_NVML_DEVICE','0')),ct.byref(self.handle)) == 0
        self.rows = []
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.is_set():
            mem = self.Memory(); temp = ct.c_uint(); clock = ct.c_uint(); flags = ct.c_ulonglong()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ct.byref(mem)) == 0:
                self.lib.nvmlDeviceGetTemperature(self.handle,0,ct.byref(temp))
                self.lib.nvmlDeviceGetClockInfo(self.handle,1,ct.byref(clock))
                self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle,ct.byref(flags))
                self.rows.append(dict(monotonic=time.perf_counter(),used_bytes=mem.used,
                    temperature_c=temp.value,sm_mhz=clock.value,throttle_bits=flags.value))
            self.stop.wait(.1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample,daemon=True)
        self.thread.start()
        deadline=time.monotonic()+5
        while not self.rows and time.monotonic()<deadline:time.sleep(.001)
        if not self.rows:
            self.stop.set();self.thread.join()
            raise RuntimeError("No GPU telemetry sample")
        return self

    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
        self.lib.nvmlShutdown()


def worker(out, case_id, engine, smoke=False, profile=False):
    protocol = read(out/'protocol.json')
    case = next(c for c in protocol['cases'] if c['id']==case_id)
    folder = out / ('profiles' if profile else ('smoke' if smoke else 'measurements')) / case_id
    folder.mkdir(parents=True,exist_ok=True)
    result = dict(ok=False,status='initializing',engine=engine,case=case,started_utc=now(),
                  protocol_sha256=digest(out/'protocol.json'),physical_gpu=0,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    output = folder/(engine+'.json')
    write(output,result)
    try:
        package = 'gputls' if engine=='gtls' else 'cetra'
        src = out/'sources'/engine
        assert source_hashes(src/package)==protocol['sources'][engine], 'Frozen source mismatch'
        assert digest(out/case['file'])==case['sha256'], 'Input hash mismatch'
        sys.path.insert(0,str(ROOT/'GTLS-Compare/.deps'))
        sys.path.insert(0,str(src))
        if engine=='gtls':
            import gputls as module
            import cupy as cp
            cp.cuda.Device(0).use(); cp.zeros(1).sum().get()
            assert cp.cuda.runtime.getDeviceCount()==1
            sync = cp.cuda.runtime.deviceSynchronize
            from gputls.grid import period_grid

            def expected(t):
                s=protocol['search']
                return period_grid(time_span=float(np.ptp(t)),**{k:s[k] for k in
                    ('R_star','M_star','period_min','period_max','oversampling_factor','n_transits_min')})

            def call(t,y,dy,warm=False):
                settings=dict(protocol['search'],GPUDeviceID=0)
                if warm: settings.update(period_min=1.,period_max=2.,oversampling_factor=1)
                r=module.gtls(t,y,dy,verbose=False).power(**settings)
                return np.asarray(r.periods),np.asarray(r.power),dict(period=float(r.period),snr=float(r.snr))
        else:
            import cetra as module
            import pycuda.driver as cuda
            import pycuda.autoinit
            assert cuda.Device.count()==1
            sync=cuda.Context.synchronize
            from cetra.cetra import period_grid

            def expected(t):
                return period_grid(float(np.ptp(t)),min_period=1.,max_period=12.,R_star=1.,M_star=1.,
                                   oversampling_factor=3,n_transits_min=2)

            def call(t,y,dy,warm=False):
                d=module.TransitDetector(module.LightCurve(t,y,dy,verbose=False),**protocol['cetra'],verbose=False)
                d.linear_search(verbose=False)
                r=d.period_search(min_period=1.,max_period=2. if warm else 12.,n_transits_min=2,
                    pgrid_R_star=1.,pgrid_M_star=1.,pgrid_oversample=1 if warm else 3,
                    random_order=False,verbose=False)
                c=d.get_max_likelihood_periodic_transit()
                return np.asarray(r.period_array),np.asarray(r.like_ratio_array),dict(
                    period=float(c.period),t0=float(c.t0),duration=float(c.duration),depth=float(c.depth))
        assert module.__version__==VERSIONS[engine], 'Wrong algorithm version'
        result.update(version=module.__version__,module_path=module.__file__)
        with np.load(out/case['file'],allow_pickle=False) as a:
            t,y,dy=(a[k][:case['n_points']].copy() for k in ('t','y','dy'))
        assert {k:array_hash(v) for k,v in zip(('t','y','dy'),(t,y,dy))}==case['arrays_sha256']
        warm=t<t[0]+30
        start=time.perf_counter();call(t[warm],y[warm],dy[warm],True);sync()
        result.update(warmup_seconds=time.perf_counter()-start,status='running',
                      timings_seconds=[],candidates=[],telemetry=[])
        write(output,result)
        if not smoke:
            start=time.perf_counter();call(t,y,dy);sync()
            result.update(full_warmup_seconds=time.perf_counter()-start);write(output,result)
        exp=np.sort(expected(t))
        for repeat in range(1 if smoke or profile else protocol['repeats']):
            sync()
            with Monitor() as monitor:
                if profile:assert ct.CDLL('libcuda.so.1').cuProfilerStart()==0
                start=time.perf_counter();pp,score,candidate=call(t,y,dy);sync()
                elapsed=time.perf_counter()-start
                if profile:assert ct.CDLL('libcuda.so.1').cuProfilerStop()==0
            assert pp.shape==score.shape and np.isfinite(score).all(), 'Invalid periodogram'
            assert np.array_equal(np.sort(pp),exp), 'Unexpected native grid'
            result['timings_seconds'].append(elapsed)
            result['candidates'].append(candidate)
            rows=monitor.rows
            result['telemetry'].append(dict(sample_count=len(rows),
                peak_device_gib=max(r['used_bytes'] for r in rows)/1024**3,
                max_temperature_c=max(r['temperature_c'] for r in rows),
                min_sm_mhz=min(r['sm_mhz'] for r in rows),max_sm_mhz=max(r['sm_mhz'] for r in rows),
                thermal_throttled=any(r['throttle_bits']&0x60 for r in rows)))
            write(folder/f'{engine}_telemetry_{repeat+1}.json',rows)
            write(output,result)
            print(f'{case_id} {engine} repeat {repeat+1}: {elapsed:.6f}s P={candidate["period"]:.8f}',flush=True)
        np.savez_compressed(folder/(engine+'_periodogram.npz'),periods=pp,power=score)
        result.update(ok=True,status='complete',finished_utc=now(),median_seconds=float(np.median(result['timings_seconds'])),
            std_seconds=float(np.std(result['timings_seconds'],ddof=1)) if len(result['timings_seconds'])>1 else 0.,
            grid=dict(count=len(pp),min=float(pp.min()),max=float(pp.max()),sha256=hashlib.sha256(np.sort(pp).astype('<f8').tobytes()).hexdigest()),
            all_periods_recovered=all(abs(c['period']/7.123-1)<.01 for c in result['candidates']))
    except Exception:
        result.update(status='error',error=traceback.format_exc(),finished_utc=now())
        print(result['error'],flush=True)
    write(output,result)
    return 0 if result['ok'] else 1


def report(out, plots=True):
    protocol=read(out/'protocol.json'); rows=[]; results=[]
    for c in protocol['cases']:
        row={k:c[k] for k in ('id','days','cadence_seconds','n_points')}
        pair={}
        for engine in VERSIONS:
            f=out/'measurements'/c['id']/(engine+'.json')
            if not f.exists(): continue
            r=read(f);results.append(r)
            if not r.get('ok'):continue
            pair[engine]=r
            row[engine+'_seconds']=r['median_seconds']
            row[engine+'_std_seconds']=r['std_seconds']
            row[engine+'_thermal_throttled']=any(t['thermal_throttled'] for t in r['telemetry'])
        if len(pair)==2:row['cetra_over_gtls']=pair['cetra']['median_seconds']/pair['gtls']['median_seconds']
        rows.append(row)
    fields=['id','days','cadence_seconds','n_points']+[e+s for e in VERSIONS for s in ('_seconds','_std_seconds','_thermal_throttled')]+['cetra_over_gtls']
    with (out/'timings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    complete=sum(r.get('ok',False) for r in results)
    write(out/'summary.json',dict(complete=complete==60,completed_jobs=complete,total_jobs=60,
        updated_utc=now(),rows=rows,failed_jobs=[r for r in results if r['status']=='error']))
    if not plots:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,days in zip(axes.flat,DAYS):
        subset=sorted([r for r in rows if r['days']==days],key=lambda r:r['cadence_seconds'])
        for engine,color in [('gtls','#2466b4'),('cetra','#df7226')]:
            valid=[r for r in subset if engine+'_seconds' in r]
            if valid:
                ax.errorbar([r['cadence_seconds'] for r in valid],[r[engine+'_seconds'] for r in valid],
                    yerr=[r[engine+'_std_seconds'] for r in valid],fmt='o-',capsize=3,color=color,
                    label=f'{engine.upper()} {VERSIONS[engine]}')
                hot=[r for r in valid if r[engine+'_thermal_throttled']]
                ax.scatter([r['cadence_seconds'] for r in hot],[r[engine+'_seconds'] for r in hot],marker='x',s=90,color='black',zorder=5)
        ax.set(xscale='log',yscale='log',title=f'{days} days',xlabel='Cadence (seconds)',ylabel='Wall time (seconds)')
        ax.set_xticks(CADENCES,labels=CADENCES);ax.grid(alpha=.2,which='both')
    axes.flat[5].axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles,labels,loc='upper left',frameon=False)
    axes.flat[5].text(0,.68,'Single NVIDIA A100\nMedian of 3 warmed full searches\nError bars: sample standard deviation\nBlack x: thermal throttle observed\n\nP = 1–12 days; native period grids\nGTLS coarse/refine = 0.01 / 0.01\nCETRA t0 stride fraction = 0.1\n250 ppm white noise, varying dy; P_inj = 7.123 d\n7 sub-exposures per cadence\n\nCompleted jobs: '+str(complete)+'/60',va='top',linespacing=1.5)
    fig.suptitle('GTLS 2.2.1 vs CETRA 1.04 — cadence and baseline',fontsize=16)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'runtime_vs_cadence.{ext}',dpi=180)
    plt.close(fig)
    matrix=np.full((len(DAYS),len(CADENCES)),np.nan)
    for r in rows:
        if 'cetra_over_gtls' in r:matrix[DAYS.index(r['days']),CADENCES.index(r['cadence_seconds'])]=r['cetra_over_gtls']
    if np.isfinite(matrix).any():
        log=np.log2(matrix); bound=max(1.,float(np.nanmax(np.abs(log))))
        fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
        im=ax.imshow(log,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
        ax.set_xticks(range(6),labels=CADENCES);ax.set_yticks(range(5),labels=DAYS)
        ax.set(xlabel='Cadence (seconds)',ylabel='Baseline (days)',title='CETRA wall time / GTLS wall time\n>1: GTLS faster; <1: CETRA faster')
        for i in range(5):
            for j in range(6):
                value=matrix[i,j]
                ax.text(j,i,f'{value:.2f}×' if np.isfinite(value) else 'pending',ha='center',va='center',
                    color='white' if np.isfinite(value) and abs(log[i,j])>.55*bound else 'black')
        fig.colorbar(im,ax=ax,label='log2(CETRA / GTLS)')
        for ext in ('png','pdf','svg'):fig.savefig(out/f'speed_ratio.{ext}',dpi=180)
        plt.close(fig)
    print(f'Report: {complete}/60 complete',flush=True)


def run(out):
    import fcntl
    lock=(out/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(out/'protocol.json')
    assert digest(__file__)==p['runner_sha256'], 'Runner changed after protocol freeze'
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],text=True)
    gpu0=next(line for line in hardware.splitlines() if line.startswith('0,'))
    uuid=gpu0.split(',')[2].strip()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE='0',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        MPLCONFIGDIR=str(out/'mpl-cache'),CUPY_CACHE_DIR=str(out/'cupy-cache'))
    if not (out/'hardware.json').exists():
        write(out/'hardware.json',dict(physical_gpu=0,gpu_uuid=uuid,started_utc=now(),
            nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
            cpu=subprocess.check_output(['lscpu'],text=True),python=sys.version,executable=sys.executable))
    jobs=[(p['cases'][0],e,True) for e in VERSIONS]
    for index,case in enumerate(p['cases']):
        order=('gtls','cetra') if index%2==0 else ('cetra','gtls')
        jobs.extend((case,e,False) for e in order)
    for case,engine,smoke in jobs:
        f=out/('smoke' if smoke else 'measurements')/case['id']/(engine+'.json')
        if f.exists() and read(f).get('ok'):continue
        summary=read(out/'summary.json') if (out/'summary.json').exists() else {'completed_jobs':0}
        write(out/'status.json',dict(phase='smoke' if smoke else 'running',case=case['id'],engine=engine,
            completed_jobs=summary['completed_jobs'],total_jobs=60,updated_utc=now(),pid=os.getpid()))
        print('[start]',case['id'],engine,'smoke' if smoke else '',flush=True)
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--out',str(out),'--case',case['id'],'--engine',engine]
        if smoke:command.append('--smoke')
        f.parent.mkdir(parents=True,exist_ok=True)
        with (f.with_suffix('.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            # Each worker is isolated, runs serially and checkpoints every completed repeat.
            code=process.wait()
        if code or not f.exists() or not read(f).get('ok'):
            write(out/'status.json',dict(phase='error',case=case['id'],engine=engine,returncode=code,updated_utc=now()))
            print('[failed]',f,flush=True)
            return 1
        r=read(f); print('[done]',case['id'],engine,r['median_seconds'],'s',flush=True)
        if not smoke:report(out,False)
    report(out)
    write(out/'status.json',dict(phase='complete',completed_jobs=60,total_jobs=60,updated_utc=now()))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker','report'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--case');parser.add_argument('--engine',choices=VERSIONS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if args.command=='prepare':prepare(out)
    elif args.command=='worker':return worker(out,args.case,args.engine,args.smoke,args.profile)
    elif args.command=='run':return run(out)
    else:report(out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
