"""Inspect pruning and device event time on the same fixed 64 periods (not E2E timing)."""
import json
import sys
from pathlib import Path
import validate as bench
from validate import cp, np, core, gtls_main, gputls

if len(sys.argv) > 2:
    import gputls.screening as screening
    fit = screening.template_certificate
    screening.template_certificate = lambda signal: fit(signal, float(sys.argv[2]))

protocol=json.loads((bench.EXPERIMENT/'protocol.json').read_text())
arrays,_=bench.inputs(protocol, '0365d_100s', 'mixed')
settings=protocol['settings']['gtls']
output=Path(sys.argv[1])
measure=[]
real_loader=core.load_module
class Module:
    def __init__(self): self.module=real_loader()
    def get_function(self,name):
        fun=self.module.get_function(name)
        def call(grid,block,args):
            begin,end=cp.cuda.Event(),cp.cuda.Event()
            begin.record(); fun(grid,block,args); end.record(); end.synchronize()
            row=dict(kernel=name,ms=cp.cuda.get_elapsed_time(begin,end))
            if name=='screenCandidates':
                lower,upper=args[:2]
                threshold=core.decode_keys(upper,0)[0]
                active=cp.isfinite(lower)
                row.update(candidates=lower.size, active=int(active.sum()),
                    survivors=int(cp.sum(active & (lower <= threshold[:,None]))),
                    unbounded=int(cp.sum(cp.isneginf(lower))),
                    terms=args[14].get()[:,1].tolist())
            measure.append(row)
        return call
core.load_module=Module

def probe(self,periods,size,refine,pbar=None):
    batch=np.asarray(periods)[15*815:15*815+64]
    self.primary.chi2_batch(batch,len(batch),False)
    measure.clear()
    self.primary.chi2_batch(batch,len(batch),False)
    output.write_text(json.dumps(measure,indent=2))
    print(json.dumps(measure,indent=2),flush=True)
    raise bench.ProbeComplete
core.MultiGPUSearch.chi2=probe
try: gputls.gtls(*arrays,verbose=False).power(**settings)
except bench.ProbeComplete: pass
