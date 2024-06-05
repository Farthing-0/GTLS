import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from astropy.io import fits
sys.path.insert(1, '../../GTLSTest/precision')
from lcFuns import gpubls, cleaned_array,normalize,checkParams
from transitleastsquares import transitleastsquares
from gputls import gtls
import lightkurve as lk
import wotan

saveFile = 'KeplerGTLSFastTest.csv'
saveFileData = pd.DataFrame(columns=['KIC','period','T0','duration','depth','SNR','SDE'])

data = pd.read_csv('allKoi2lcResult_update.csv')
# rank by pointsLength
data = data.sort_values(by='pointsLength', ascending=False)

from tqdm import tqdm
for index,line in data.iterrows():
    # iter over all columns
    dir = '/mnt/HDD0/Kepler/lightcurves'
    files = []

    # if 9171801 != int(line["kepid"]):
    if 10227501 != int(line["kepid"]):
    # if 11013201 != int(line["kepid"]):
        continue

    for name in line.index:
        if "public" in name and "long" in name:
            if type(line[name]) == str:
                files.append(dir + '/' + name + '/' + line[name])
    AllTimes = []
    AllFluxes = []
    AllDys = []
    for file in (files):
        with fits.open(file) as hdul:
            # times = hdul[1].data['TIME']
            # fluxes = hdul[1].data['PDCSAP_FLUX']
            # dys = hdul[1].data['PDCSAP_FLUX_ERR']
            # # remove nan
            # times, fluxes, dys = cleaned_array(times, fluxes, dys)
            # # normalize
            # times,fluxes,dy = normalize(times,fluxes,dys)

            # window = 0.5
            times,fluxes,dys,radius,logg = checkParams(file)
            times,fluxes,dys = cleaned_array(times,fluxes,dys)
            times,fluxes,dys = normalize(times,fluxes,dys)

            # detrend
            M_s = 1.98892e30 # mass of sun in kg
            R_s = 6.957e8 # radius of sun in m
            Gc = 6.67408e-11 # gravitational constant in m^3 kg^-1 s^-2

            if len(times) < 10:
                continue

            if (radius != None and logg != None):
                mass = np.power(10,logg) / 100 * np.power(radius*R_s,2) / Gc / M_s # mass in solar mass
                window = 3 * wotan.t14(R_s=radius, M_s=mass, P=14, small_planet=True)
            else:
                window = 0.5

            fluxes, trend_lc = wotan.flatten(times, fluxes, window_length=window, method='biweight', return_trend=True)

            AllTimes.extend(times.tolist())
            AllFluxes.extend(fluxes.tolist())
            AllDys.extend(dys.tolist())

    #sort by time
    AllIndex = np.argsort(AllTimes)
    AllTimes = np.array(AllTimes)[AllIndex]
    AllFluxes = np.array(AllFluxes)[AllIndex]
    AllDys = np.array(AllDys)[AllIndex]

    T0_fit_margin = 0.125
    TLSTestFlag = False
    # TLSTestFlag = True
    GTLSTestFlag = True
    if TLSTestFlag:
        model = transitleastsquares(AllTimes, AllFluxes, AllDys)
        results = model.power(T0_fit_margin = T0_fit_margin)

    if GTLSTestFlag:
        GTLSmodel = gtls(t = AllTimes, y = AllFluxes, dy = AllDys)
        gtlsResult = GTLSmodel.power(bar_location = 0,GPUDeviceID = 1,T0_fit_margin = T0_fit_margin)
        print(gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE)
    break

    # saveFileData.loc[len(saveFileData)] = [line['kepid'],gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE]
    # saveFileData.to_csv(saveFile, index=False)