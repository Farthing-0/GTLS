import lightkurve as lk
import matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares
from gputls import gtls

search_result_q2 = lk.search_lightcurve('KIC 11446443', author='Kepler', quarter=2)
lc = search_result_q2.download()
# lc.plot()

from wotan import flatten
import numpy as np
time = lc.time.to_value('bkjd', 'long')
flux = lc.flux.unmasked.value
flux = flux / np.nanmedian(flux)
flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, method='biweight', return_trend=True)
# plt.plot(time, flatten_lc, '.')

import time as Nowtime
start = Nowtime.time()
model = gtls(t = time, y = flux)
gtlsResult = model.power()
print('period', gtlsResult.period, 'duration', gtlsResult.duration, 'depth', gtlsResult.depth, 'T0', gtlsResult.T0,'SDE', gtlsResult.SDE)
print('Time taken for GPU',Nowtime.time() - start)