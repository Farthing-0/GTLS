import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from transitleastsquares import transitleastsquares
from gputls import gtls
from gputls.helpers import cleaned_array
import lightkurve as lk
import wotan

project_dir = Path(__file__).resolve().parent
saveFile = 'KeplerGTLSFastTest.csv'
saveFileData = pd.DataFrame(columns=['KIC','period','T0','duration','depth','SNR','SDE'])

data = pd.read_csv(project_dir / 'allKoi2lcResult_update.csv')
# rank by pointsLength
data = data.sort_values(by='pointsLength', ascending=False)
test_data_dir = project_dir / 'test_data'

from tqdm import tqdm
for index,line in data.iterrows():
    files = []

    if 11403044 != int(line["kepid"]):
    # if 7295235 != int(line["kepid"]):
    # if 11336883 != int(line["kepid"]):
    # if 3003992 != int(line["kepid"]):
    # if 11013201 != int(line["kepid"]):
        continue

    name = 'public_Q3_long_5'
    if isinstance(line[name], str):
        files.append(test_data_dir / name / line[name])
    AllTimes = []
    AllFluxes = []
    AllDys = []
    for file in (files):
        with fits.open(file) as hdul:
            times = hdul[1].data['TIME']
            fluxes = hdul[1].data['PDCSAP_FLUX']
            dys = hdul[1].data['PDCSAP_FLUX_ERR']
            radius = hdul[0].header.get('RADIUS')
            logg = hdul[0].header.get('LOGG')
            times,fluxes,dys = cleaned_array(times,fluxes,dys)

            if len(times) < 10:
                continue

            mean_flux = np.mean(fluxes)
            fluxes = fluxes / mean_flux
            dys = dys / mean_flux

            # detrend
            M_s = 1.98892e30 # mass of sun in kg
            R_s = 6.957e8 # radius of sun in m
            Gc = 6.67408e-11 # gravitational constant in m^3 kg^-1 s^-2

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
    GTLSTestFlag = False

    # TLSTestFlag = True
    GTLSTestFlag = True
    if TLSTestFlag:
        model = transitleastsquares(AllTimes, AllFluxes, AllDys)
        results = model.power(T0_fit_margin = T0_fit_margin)
        print(results.period,results.T0,results.duration,results.depth,results.snr,results.SDE)

    if GTLSTestFlag:
        GTLSmodel = gtls(t = AllTimes, y = AllFluxes, dy = AllDys)
        # gtlsResult = GTLSmodel.power(bar_location = 0,GPUDeviceID = 1,T0_fit_margin = T0_fit_margin)
        gtlsResult = GTLSmodel.power(
            periods = np.linspace(100,150,500),
            bar_location = 0,GPUDeviceID = 1,T0_fit_margin = T0_fit_margin)
        
        print(gtlsResult.raw_chi2.tolist())
        print(gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE)
    break

    # saveFileData.loc[len(saveFileData)] = [line['kepid'],gtlsResult.period,gtlsResult.T0,gtlsResult.duration,gtlsResult.depth,gtlsResult.snr,gtlsResult.SDE]
    # saveFileData.to_csv(saveFile, index=False)