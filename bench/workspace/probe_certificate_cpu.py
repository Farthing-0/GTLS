import os,sys,subprocess,json,time,inspect
from pathlib import Path
repo=Path(__file__).resolve().parents[2];exp=repo.parents[1]/'GTLS-Compare/tls2-gtls-delta001-20261114v2/workspace-optimization';sys.path.insert(0,str(repo/'src'));os.environ['NUMBA_CACHE_DIR']=str(exp/'cache/numba')
import numpy as np
from gputls import screening
baseline_source=subprocess.check_output(['git','show','d78df2c:src/gputls/screening.py'],cwd=repo,text=True);ns={};exec(baseline_source,ns);original=ns['template_certificate']
old='''        approx = target[left] + fraction[:, None] * (target[right] - target[left])
        error = np.max(np.abs(target[left + 1:right] - approx) / scale, axis=1)'''
new='''        a = target[left,0] + fraction*(target[right,0]-target[left,0])
        b = target[left,1] + fraction*(target[right,1]-target[left,1])
        error = np.maximum(np.abs(target[left+1:right,0]-a)/scale[0], np.abs(target[left+1:right,1]-b)/scale[1])'''
assert old in baseline_source
ns2={};exec(baseline_source.replace(old,new),ns2);column=ns2['template_certificate']
with np.load(exp/'certificate-cpu/input.npz') as data:signal,widths=data['signal'],data['widths']
screening.template_certificate(signal[0,:widths[0]])
refs={};times={}
for name,fun in [('original',original),('columns',column),('numba',screening.template_certificate)]:
 start=time.perf_counter()
 for tolerance in screening.SCREEN_TOLERANCES:
  for index,(row,width) in enumerate(zip(signal,widths)):
   result=fun(row[:width],tolerance)
   if name=='original':refs[tolerance,index]=result
   else:assert all(np.array_equal(x,y) for x,y in zip(result,refs[tolerance,index])),(name,tolerance,index)
 times[name]=time.perf_counter()-start
print(json.dumps(times,indent=2));(exp/'certificate-cpu/columns-result.json').write_text(json.dumps(times,indent=2)+'\n')
