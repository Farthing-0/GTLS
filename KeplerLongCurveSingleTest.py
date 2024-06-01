import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from astropy.io import fits
sys.path.insert(1, '../../GTLSTest/precision')
from lcFuns import gpubls, cleaned_array,normalize
from transitleastsquares import transitleastsquares
from gputls import gtls
import lightkurve as lk

saveFile = 'KeplerGTLSFastTest.csv'
saveFileData = pd.DataFrame(columns=['KIC','period','T0','duration','depth','SNR','SDE'])

data = pd.read_csv('allKoi2lcResult_update.csv')
# rank by pointsLength
data = data.sort_values(by='pointsLength', ascending=False)

# # load light curve 10666592.0
# line = data.head(1)
# print(line)

from tqdm import tqdm
# for index,line in tqdm(data.iterrows(), total=data.shape[0]):
for index,line in data.iterrows():
    # iter over all columns
    dir = '/mnt/HDD0/Kepler/lightcurves'
    files = []
    for name in line.index:
        if "public" in name and "long" in name:
            if type(line[name]) == str:
                files.append(dir + '/' + name + '/' + line[name])

    AllTimes = []
    AllFluxes = []
    AllDys = []
    for file in (files):
        with fits.open(file) as hdul:
            # print(hdul.info())
            # print(hdul[1].columns)
            times = hdul[1].data['TIME']
            fluxes = hdul[1].data['PDCSAP_FLUX']
            dys = hdul[1].data['PDCSAP_FLUX_ERR']
            # remove nan
            times, fluxes, dys = cleaned_array(times, fluxes, dys)
            # normalize
            times,fluxes,dy = normalize(times,fluxes,dys)
            AllTimes.extend(times.tolist())
            AllFluxes.extend(fluxes.tolist())
            AllDys.extend(dy.tolist())

    #sort by time
    AllIndex = np.argsort(AllTimes)
    AllTimes = np.array(AllTimes)[AllIndex]
    AllFluxes = np.array(AllFluxes)[AllIndex]
    AllDys = np.array(AllDys)[AllIndex]

    # lkFlux = lk.LightCurve(time=AllTimes, flux=AllFluxes, flux_err=AllDys)
    # lkFluxBin = lkFlux.bin(binsize=3)
    # AllTimes = lkFluxBin.time.value
    # AllFluxes = lkFluxBin.flux.value
    # AllDys = lkFluxBin.flux_err.value

    T0_fit_margin = 0.1
    TLSTestFlag = False
    # TLSTestFlag = True
    GTLSTestFlag = True
    if TLSTestFlag:
        model = transitleastsquares(AllTimes, AllFluxes)
        results = model.power(T0_fit_margin = T0_fit_margin)

    if GTLSTestFlag:
        GTLSmodel = gtls(t = AllTimes, y = AllFluxes)
        # gtlsResult = GTLSmodel.power(T0_fit_margin = T0_fit_margin,bar_location = 0,GPUDeviceID = 0,bin_size = 10)
        # gtlsResult = GTLSmodel.power(T0_fit_margin = T0_fit_margin,bar_location = 0,GPUDeviceID = 1)
        gtlsResult = GTLSmodel.power(bar_location = 0,GPUDeviceID = 1)
        print(gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE)

    break

    # saveFileData.loc[len(saveFileData)] = [line['kepid'],gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE]
    # saveFileData.to_csv(saveFile, index=False)