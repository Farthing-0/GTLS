"""Verify saved output bytes, source hashes, common profiler inputs and normal-user NCU."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

repo=Path(__file__).resolve().parents[2]
root=Path('/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114/round2-candidate-screening-20261114')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
current={name:sha(repo/'src/gputls'/name) for name in ('core.py','kernels.cu','screening.py')}
verification=dict(source_sha256=current,cases={},profilers={})
for case in ['small-smooth-gpu1','medium-random-gpu1','medium-spikes-final-gpu1','full-mixed-gpu0']:
    p=root/case
    d=json.loads((p/'result.json').read_text())
    assert d['source_sha256']==current
    a,b=dict(np.load(p/'baseline.npz')),dict(np.load(p/'optimized.npz'))
    assert a.keys()==b.keys()
    for k in a:
        assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape
        assert a[k].tobytes()==b[k].tobytes(),(case,k)
    verification['cases'][case]=dict(all_24_fields_bitwise_equal=True,repeats=len(d['runs'])//2,
                                    input_sha256=d['input']['sha256'])
reference=dict(np.load(root/'full-mixed-gpu0/baseline.npz'))
full=json.loads((root/'full-mixed-gpu0/result.json').read_text())
for mode in ['nsys','ncu']:
    metadata=[]
    for variant in ['baseline','optimized']:
        p=root/f'{mode}-{variant}'
        d=json.loads((p/'result.json').read_text())
        assert d['source_sha256']==current
        assert d['input']==full['input'] and d['settings']==full['settings']
        metadata.append(d)
        if mode=='nsys':
            a=dict(np.load(p/'periodogram.npz'))
            assert a.keys()==reference.keys()
            for k in reference: assert reference[k].tobytes()==a[k].tobytes(),(mode,variant,k)
        else:
            assert d['batch_chi2_exact']
            assert (p/'profile.ncu-rep').stat().st_uid==1000
            assert 'ERR_NVGPUCTRPERM' not in (p/'profile.log').read_text()
    assert metadata[0]['device']==metadata[1]['device']
    if mode=='ncu':
        assert metadata[0]['size']==metadata[1]['size']==815
        assert metadata[0]['period_range']==metadata[1]['period_range']
    verification['profilers'][mode]=dict(same_inputs=True,same_device=True,final_source=True,
                                        normal_user_uid=1000 if mode=='ncu' else None)
files=[Path('src/gputls')/n for n in current]
files += [p.relative_to(repo) for p in (repo/'bench/candidate_screening').glob('*') if p.is_file()]
files += [Path('tests/test_candidate_screening.py')]
manifest={}
for relative in files:
    destination=root/'final-source'/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(repo/relative,destination)
    manifest[str(relative)]=sha(destination)
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
