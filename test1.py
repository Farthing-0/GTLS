import os
from astropy.io import fits
import numpy as np
import numpy
import time
import random
import wotan

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
    # lc_file = lc_dir + files[10]
    for lc_file in files:
        # if '0000000021132157' in lc_file:
        if '0000000373961316' in lc_file:
    #     # if '0000000028473414' in lc_file:
    # #     # if '0000000010596267' in lc_file:
    # #     if '0000000015422557' in lc_file:
            break
    lc_file = lc_dir + lc_file
    
    print(lc_file)
    fluxes = None
    
    with fits.open(lc_file,mode = "readonly") as lc_file:
        fluxes = lc_file[1].data['PDCSAP_FLUX']
        times = lc_file[1].data['TIME']
        dy = lc_file[1].data['PDCSAP_FLUX_ERR']
        radius = lc_file[0].header['RADIUS']
        logg = lc_file[0].header['LOGG']
    # exit()
    return times,fluxes,dy,radius,logg

def normalize(times,fluxes,dy):
    fluxes = fluxes/np.nanmean(fluxes)
    return times,fluxes,dy

if __name__ == '__main__':

    M_s = 1.98892e30 # mass of sun in kg
    R_s = 6.957e8 # radius of sun in m
    Gc = 6.67408e-11 # gravitational constant in m^3 kg^-1 s^-2

    modelStart = time.time()
    times,fluxes,dy,radius,logg = findRandomLc()
    times,fluxes,dy = cleaned_array(times,fluxes,dy)
    times,fluxes,dy = normalize(times,fluxes,dy)

    if (radius != None and logg != None):
        mass = np.power(10,logg) / 100 * np.power(radius*R_s,2) / Gc / M_s # mass in solar mass
        window = 3 * wotan.t14(R_s=radius, M_s=mass, P=14, small_planet=True)
    else:
        window = 0.5

    # clipped_flux = wotan.slide_clip(
    # times,
    # fluxes,
    # window_length=window,
    # low=3,
    # high=2,
    # method='mad',  # mad or std
    # center='median'  # median or mean
    # )

    # flatten_lc, trend_lc = wotan.flatten(times, clipped_flux, window_length=window, method='biweight', return_trend=True)
    flatten_lc, trend_lc = wotan.flatten(times, fluxes, window_length=window, method='biweight', return_trend=True)
    
    # flatten_lc, trend_lc = flatten(times, fluxes, window_length=0.15, method='biweight', return_trend=True)


    from transitleastsquares import transitleastsquares
    # # # # from main import transitleastsquares

    # # # # model = transitleastsquares(t = times, y = flatten_lc, GPU = False ,dy = dy)
    # model = transitleastsquares(t = times, y = flatten_lc,dy = dy)
    # results = model.power()

    from gputls import gtls

    time0 = time.time()
    model = gtls(t = times, y = flatten_lc, dy = dy)
    # gtlsResult = model.power(useLocalPTXCUBIN=True)
    gtlsResult = model.power(GPUDeviceID = 0)

    # print('Time taken for GPU',time.time() - time0)
    # print('CPU results')
    # print('period', results.period, 'duration', results.duration, 'depth', results.depth, 'T0', results.T0,'SDE', results.SDE,'snr', results.snr,'DepthMean',results.depth_mean)

    print('GPU results')
    print('period', gtlsResult.period, 'duration', gtlsResult.duration, 'depth', gtlsResult.depth, 'T0', gtlsResult.T0,'SDE', gtlsResult.SDE,
          'snr', gtlsResult.snr,'snrPink', gtlsResult.snrPink,'snrFit',gtlsResult.snrFit,'snrFitPink',gtlsResult.snrFitPink)

    print('periods searched',(gtlsResult.periods))
    print('rawPoints',(gtlsResult.durationPoints))
    # print('DFToutlineValue',gtlsResult.DFToutlineValue)

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