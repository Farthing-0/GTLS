"""Plot warmed API timing separately from diagnostic GPU-kernel times."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path)
root=parser.parse_args().experiment
report=json.loads((root/'a100/comparison.json').read_text())
runs=report['runs']
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.9),gridspec_kw={'width_ratios':[1,1.05]})
colors=['#426b9c','#dc8b3e']
x=np.arange(2)
for i,(device,names) in enumerate([
    ('RTX 3090',['3090_workspace_mixed','3090_workspace_constant']),
    ('A100 PCIe 40GB',['A100_device_bound_mixed','A100_device_bound_constant'])]):
    values=[runs[name]['median_seconds'] for name in names]
    bars=ax.bar(x+(i-.5)*.32,values,.32,label=device,color=colors[i])
    ax.bar_label(bars,fmt='%.2f s',padding=3,fontsize=10)
ax.set_xticks(x,['Varying dy','Constant dy']);ax.set_ylabel('Seconds (lower is better)')
ax.set_ylim(0,65);ax.set_title('Warmed API runtime: median of 3 calls')
ax.legend(frameon=False,loc='upper left');ax.spines[['top','right']].set_visible(False)

categories=['Candidate screening','Screening prefixes','Exact candidates','Other kernels']
for device,xpos,path in [('RTX 3090',0,root/'nsys-workspace-730-mixed/summary.json'),
                         ('A100 PCIe',1,root/'a100/results/nsys-cheap-screen-mixed/summary.json')]:
    values=np.zeros(4)
    for row in json.loads(path.read_text())['kernels']:
        name=row['name']
        index=0 if name=='screenCandidates' else 1 if name=='scanScreening' else 2 if name.startswith('searchChi2Screened') else 3
        values[index]+=row['seconds']
    bottom=0
    for i,(name,value,color) in enumerate(zip(categories,values,['#5285bd','#86bcb6','#df9b50','#b3b7be'])):
        bx.bar(xpos,value,.56,bottom=bottom,color=color,label=name if xpos==0 else None)
        if value>1:bx.text(xpos,bottom+value/2,f'{value:.2f}',ha='center',va='center',fontsize=9)
        bottom+=value
bx.set_xticks([0,1],['RTX 3090','A100 PCIe']);bx.set_ylabel('GPU kernel seconds')
bx.set_title('nsys kernel totals: varying dy')
bx.spines[['top','right']].set_visible(False);bx.legend(frameon=False,fontsize=9,loc='upper right')
fig.suptitle('GTLS 2.2.0 branch | 730 days / 100 s | 630,720 samples',fontsize=14)
fig.text(.5,.025,'Same inputs and 48,935 periods; all 24 result fields match bitwise.\nReusable workspace on both GPUs; cheap bound enabled on A100. Different host CPUs/drivers.',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.09,1,.95))
for suffix in ['png','pdf']:
 fig.savefig(root/f'a100/gpu-comparison.{suffix}',dpi=180)
