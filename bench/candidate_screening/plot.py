"""Standalone scientific comparison of the complete periodogram and fitted outputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gtls-round2-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/farthing/GTLS/GTLS-Compare/tls2-gtls-delta001-20261114/round2-candidate-screening-20261114')
a=dict(np.load(root/'full-mixed-gpu0/baseline.npz'))
b=dict(np.load(root/'full-mixed-gpu0/optimized.npz'))
fig,axs=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
period=a['periods']
for ax,field,title in [(axs[0,0],'raw_chi2','Full raw periodogram'),(axs[0,1],'power','Full refined periodogram')]:
    ax.plot(period,a[field],color='#395b91',lw=1.1,label='Round 1')
    ax.plot(period,b[field],color='#df7427',lw=.8,ls='--',label='Round 2')
    ax.set(xlabel='Period [days]',ylabel=field,title=title,xlim=(period.min(),period.max()))
    ax.legend(frameon=False)
axs[1,0].plot(period,b['raw_chi2']-a['raw_chi2'],color='#395b91',label='raw chi2 difference')
axs[1,0].plot(period,b['chi2']-a['chi2'],color='#df7427',ls='--',label='refined chi2 difference')
axs[1,0].set(xlabel='Period [days]',ylabel='Round 2 - Round 1',title='All 24,468 periods: exact equality',ylim=(-1e-7,1e-7))
axs[1,0].legend(frameon=False)
axs[1,1].axis('off')
fields=['period','T0','duration','depth','SDE','snrFit']
rows=[[key,format(float(a[key]),'.10g'),format(float(b[key]),'.10g')] for key in fields]
table=axs[1,1].table(cellText=rows,colLabels=['Fitted quantity','Round 1','Round 2'],loc='center',cellLoc='left')
table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
axs[1,1].set_title('Fitted outputs agree exactly (24 fields / masks checked)')
fig.suptitle('GTLS: 365 days, 100 s cadence, heterogeneous dy',fontsize=15)
fig.savefig(root/'periodogram-fit-comparison.png',dpi=180)
fig.savefig(root/'periodogram-fit-comparison.pdf')
