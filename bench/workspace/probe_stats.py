import subprocess,types,time,json,sys
from pathlib import Path
import numpy as np
from gputls import stats
repo=Path(__file__).resolve().parents[2];exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization'
old=types.ModuleType('gputls.old_stats');old.__package__='gputls';exec(subprocess.check_output(['git','show','ba518d1:src/gputls/stats.py'],text=True),old.__dict__)
rng=np.random.default_rng(83745)
for variant in range(8):
 t=np.sort(rng.integers(-10,11,1000).astype(np.float64));y=rng.normal(size=t.size)
 if variant%2:t=t[rng.permutation(t.size)]
 if variant//2==1:t[0]=np.nan
 if variant//2==2:t[0]=-np.inf;t[-1]=np.inf
 if variant//2==3:t=t[::-1];y=y[::-1]
 times=np.array([-20.,-5.,0.,8.,30.,np.nan]);duration=[0.,-2.,4.,np.inf][variant//2]
 with np.errstate(all='ignore'):
  a=old.intransit_stats(t,y,times,duration);b=stats.intransit_stats(t,y,times,duration)
 assert all(x.dtype==z.dtype and x.shape==z.shape and x.tobytes()==z.tobytes() for x,z in zip(a,b)),variant
with np.load(exp/'postprocess-cpu/input.npz') as f:d=dict(f)
results={}
for name,m in [('baseline',old),('candidate',stats)]:
 start=time.perf_counter();a=m.intransit_stats(d['t'],d['y'],d['transit_times'],d['duration']);in_t=time.perf_counter()-start
 start=time.perf_counter();b=m.all_transit_times(d['T0'],d['t'],d['period']);all_t=time.perf_counter()-start
 start=time.perf_counter();c=m.snr_stats(d['t'],d['y'],d['period'],d['raw_duration'],d['T0'],d['transit_times'],d['duration'],d['counts']);snr_t=time.perf_counter()-start
 if name=='baseline':refs=(a,b,c)
 else:
  assert all(x.tobytes()==y.tobytes() for x,y in zip(a,refs[0]));assert b==refs[1];assert c.tobytes()==refs[2].tobytes()
 results[name]={'intransit_seconds':in_t,'transit_times_seconds':all_t,'snr_seconds':snr_t}
print(json.dumps(results,indent=2));(exp/'postprocess-cpu/window-results.json').write_text(json.dumps(results,indent=2)+'\n')
