import os
from astropy.io import fits
import numpy as np
import numpy
import time
import random
from wotan import flatten

def cleaned_array(t, y, dy=None):
    """Takes numpy arrays with masks and non-float values.
    Returns unmasked cleaned arrays."""

    def isvalid(value):
        valid = False
        if value is not None:
            if not numpy.isnan(value):
                if value > 0 and value < numpy.inf:
                    valid = True
        return valid

    # Start with empty Python lists and convert to numpy arrays later (reason: speed)
    clean_t = []
    clean_y = []
    if dy is not None:
        clean_dy = []
    # Cleaning numpy arrays with both NaN and None values is not trivial, as the usual
    # mask/delete filters do not accept their simultanous ocurrence without warnings.
    # Instead, we iterate over the array once; this is not Pythonic but works reliably.
    for i in range(len(y)):

        # Case: t, y, dy
        if dy is not None:
            if isvalid(y[i]) and isvalid(t[i]) and isvalid(dy[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])
                clean_dy.append(dy[i])

        # Case: only t, y
        else:
            if isvalid(y[i]) and isvalid(t[i]):
                clean_y.append(y[i])
                clean_t.append(t[i])

    clean_t = numpy.array(clean_t, dtype=float)
    clean_y = numpy.array(clean_y, dtype=float)

    if dy is None:
        return clean_t, clean_y
    else:
        clean_dy = numpy.array(clean_dy, dtype=float)
        return clean_t, clean_y, clean_dy

def findRandomLc():
    dir = '../../../HDD/'
    lc_dir = None
    for file in os.listdir(dir):
        # print(file)
        if file.endswith('lightcurve_58'):
            lc_dir = dir + file + '/'
            break
    if lc_dir == None:
        print('No lightcurve directory found')
        return None
        # exit()

    files = []
    for lc_file in os.listdir(lc_dir):
        if(lc_file.endswith('.fits')):
            files.append(lc_file)
    # lc_file = lc_dir + random.choice(files)
    
    #35 can be a good example
    lc_file = lc_dir + files[2]
    # for lc_file in files:
    #     # if '0000000020640548' in lc_file:
    #     # if '0000000028473414' in lc_file:
    # #     # if '0000000010596267' in lc_file:
    # #     if '0000000015422557' in lc_file:
    #         break
    # lc_file = lc_dir + lc_file
    
    print(lc_file)
    fluxes = None
    
    with fits.open(lc_file,mode = "readonly") as lc_file:
        fluxes = lc_file[1].data['PDCSAP_FLUX']
        times = lc_file[1].data['TIME']
        dy = lc_file[1].data['PDCSAP_FLUX_ERR']
    # exit()
    return times,fluxes,dy

def normalize(times,fluxes,dy):
    fluxes = fluxes/np.nanmean(fluxes)
    return times,fluxes,dy

if __name__ == '__main__':
    modelStart = time.time()
    times,fluxes,dy = findRandomLc()
    times,fluxes,dy = cleaned_array(times,fluxes,dy)
    times,fluxes,dy = normalize(times,fluxes,dy)

    flatten_lc, trend_lc = flatten(times, fluxes, window_length=0.5, method='biweight', return_trend=True)


    # from transitleastsquares import transitleastsquares
    # # # # from main import transitleastsquares

    # # # # model = transitleastsquares(t = times, y = flatten_lc, GPU = False ,dy = dy)
    # model = transitleastsquares(t = times, y = flatten_lc,dy = dy)
    # results = model.power()

    from gputls import gtls

    time0 = time.time()
    model = gtls(t = times, y = flatten_lc, dy = dy)
    gtlsResult = model.power()
    print('Time taken for GPU',time.time() - time0)
    print('CPU results')
    print('period', results.period, 'duration', results.duration, 'depth', results.depth, 'T0', results.T0,'SDE', results.SDE,'snr', results.snr,'DepthMean',results.depth_mean)

    print('GPU results')
    print('period', gtlsResult.period, 'duration', gtlsResult.duration, 'depth', gtlsResult.depth, 'T0', gtlsResult.T0,'SDE', gtlsResult.SDE,
          'snr', gtlsResult.snr,'snrPink', gtlsResult.snrPink)

    # plt.plot(periods,results.chi2,'o',label = 'CPU',color = 'red',markersize = 5,markerfacecolor='none')
    # plt.plot(periods,chi2,'x',label = 'GPU',color = 'black',alpha = 0.3,markersize = 5)
    # plt.xlabel('Period (days)')
    # plt.ylabel('Chi2')
    # plt.legend()
    # plt.title('GPU-CPU Chi2 comparison-')
    # plt.savefig('chi2.png',dpi = 300)
    # plt.close()
    
    # plt.plot(periods,results.chi2,'.',label = 'CPU',)
    # # plt.plot(periods,chi2,'x',label = 'GPU',color = 'black',alpha = 0.3,markersize = 5)
    # plt.xlabel('Period (days)')
    # plt.ylabel('Chi2')
    # plt.legend()
    # plt.title('CPU Chi2 ')
    # plt.savefig('chi2CPU.png',dpi = 300)
    # plt.close()

    # plt.plot(periods,chi2,'.',label = 'GPU')
    # plt.xlabel('Period (days)')
    # plt.ylabel('Chi2')
    # plt.legend()
    # plt.title('GPU Chi2 ')
    # plt.savefig('chi2GPU.png',dpi = 300)
    # plt.close()