"""Count certified survivors over the complete search, outside timing/profiler runs."""
import json
from pathlib import Path
import validate as bench
from validate import cp, np, core, gputls

root=bench.OUTPUT
protocol=json.loads((bench.EXPERIMENT/'protocol.json').read_text())
arrays,_=bench.inputs(protocol,'0365d_100s','mixed')
loader=core.load_module
records=[]
class Module:
    def __init__(self): self.module=loader()
    def get_function(self,name):
        kernel=self.module.get_function(name)
        def call(grid,block,args):
            kernel(grid,block,args)
            if name=='screenCandidates':
                lower,upper=args[:2]
                threshold=core.decode_keys(upper,0)[0]
                active=cp.isfinite(lower)
                kept=(lower <= threshold[:,None]) | cp.isnan(threshold[:,None])
                records.append(dict(periods=lower.shape[0], candidates=lower.size,
                    certified_active=int(active.sum()),
                    certified_survivors=int(cp.sum(active & kept)),
                    unbounded=int(cp.sum(cp.isneginf(lower)))))
        return call
core.load_module=Module
result=bench.fields(gputls.gtls(*arrays,verbose=False).power(**protocol['settings']['gtls']))
with np.load(root/'full-mixed-gpu0/baseline.npz') as expected:
    bench.compare(dict(expected),result)
summary={k:sum(r[k] for r in records) for k in records[0]}
summary['survival_fraction']=summary['certified_survivors']/summary['certified_active']
summary['all_fields_equal']=True
(root/'pruning-audit.json').write_text(json.dumps(dict(summary=summary,batches=records),indent=2))
print(json.dumps(summary),flush=True)
