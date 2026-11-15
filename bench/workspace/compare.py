"""Compare every saved native result field, including masks and FP bit patterns."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('reference',type=Path)
parser.add_argument('candidate',type=Path)
args=parser.parse_args()
left=json.loads((args.reference/'result.json').read_text())
right=json.loads((args.candidate/'result.json').read_text())
assert left['ok'] and right['ok']
assert left['input_sha256']==right['input_sha256']
assert left['settings']==right['settings']
rows=[]
with np.load(args.reference/'fields.npz') as a,np.load(args.candidate/'fields.npz') as b:
    assert set(a.files)==set(b.files)
    for key in a.files:
        x,y=a[key],b[key]
        exact=x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
        rows.append(dict(field=key,bitwise_equal=exact,shape=list(x.shape)))
result=dict(all_fields_bitwise_equal=all(r['bitwise_equal'] for r in rows),fields=rows,
            reference=str(args.reference.resolve()),candidate=str(args.candidate.resolve()),
            reference_median_seconds=left['median_seconds'],candidate_median_seconds=right['median_seconds'],
            speedup=left['median_seconds']/right['median_seconds'])
(args.candidate/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['all_fields_bitwise_equal'] else 1)
