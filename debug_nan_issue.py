#!/usr/bin/env python3
"""
Debug script to investigate NaN issue with custom periods
"""
import numpy as np
import sys
sys.path.insert(0, 'src')
from gputls import gtls
from astropy.io import fits
import pandas as pd

# Load the same data as KeplerLongCurveSingleTest.py
sys.path.insert(1, '../../GTLSTest/precision')
from lcFuns import cleaned_array, normalize, checkParams
import wotan

print("=" * 80)
print("DEBUG: NaN Issue Investigation")
print("=" * 80)

# Load Kepler data
data = pd.read_csv('allKoi2lcResult_update.csv')
data = data.sort_values(by='pointsLength', ascending=False)

for index, line in data.iterrows():
    dir = '/mnt/HDD0/Kepler/lightcurves'
    files = []

    if 11403044 != int(line["kepid"]):
        continue

    for name in line.index:
        if "public" in name and "long" in name:
            if type(line[name]) == str:
                files.append(dir + '/' + name + '/' + line[name])
    
    AllTimes = []
    AllFluxes = []
    AllDys = []
    
    for file in files:
        with fits.open(file) as hdul:
            window = 0.5
            times, fluxes, dys, radius, logg = checkParams(file)
            times, fluxes, dys = cleaned_array(times, fluxes, dys)
            times, fluxes, dys = normalize(times, fluxes, dys)

            M_s = 1.98892e30
            R_s = 6.957e8
            Gc = 6.67408e-11

            if len(times) < 10:
                continue

            if (radius is not None and logg is not None):
                mass = np.power(10, logg) / 100 * np.power(radius*R_s, 2) / Gc / M_s
                window = 3 * wotan.t14(R_s=radius, M_s=mass, P=14, small_planet=True)
            else:
                window = 0.5

            fluxes, trend_lc = wotan.flatten(times, fluxes, window_length=window, 
                                            method='biweight', return_trend=True)

            AllTimes.extend(times.tolist())
            AllFluxes.extend(fluxes.tolist())
            AllDys.extend(dys.tolist())

    # Sort by time
    AllIndex = np.argsort(AllTimes)
    AllTimes = np.array(AllTimes)[AllIndex]
    AllFluxes = np.array(AllFluxes)[AllIndex]
    AllDys = np.array(AllDys)[AllIndex]

    print(f"\nData loaded: {len(AllTimes)} points")
    print(f"Time span: {AllTimes[-1] - AllTimes[0]:.2f} days")
    
    T0_fit_margin = 0.125
    
    print("\n" + "=" * 80)
    print("TEST 1: Small period range (should work)")
    print("=" * 80)
    
    GTLSmodel = gtls(t=AllTimes, y=AllFluxes, dy=AllDys)
    periods_small = np.linspace(140, 150, 100)
    print(f"Testing with {len(periods_small)} periods: [{periods_small[0]:.2f}, {periods_small[-1]:.2f}]")
    
    result1 = GTLSmodel.power(
        periods=periods_small,
        bar_location=0,
        GPUDeviceID=1,
        T0_fit_margin=T0_fit_margin,
        verbose=True
    )
    
    nan_count1 = np.sum(np.isnan(result1.raw_chi2))
    print(f"\nResult: {nan_count1} NaN values out of {len(periods_small)}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Large period range (problematic)")
    print("=" * 80)
    
    GTLSmodel2 = gtls(t=AllTimes, y=AllFluxes, dy=AllDys)
    periods_large = np.linspace(100, 150, 500)
    print(f"Testing with {len(periods_large)} periods: [{periods_large[0]:.2f}, {periods_large[-1]:.2f}]")
    
    result2 = GTLSmodel2.power(
        periods=periods_large,
        bar_location=0,
        GPUDeviceID=1,
        T0_fit_margin=T0_fit_margin,
        verbose=True
    )
    
    nan_count2 = np.sum(np.isnan(result2.raw_chi2))
    print(f"\nResult: {nan_count2} NaN values out of {len(periods_large)}")
    
    if nan_count2 > 0:
        nan_indices = np.where(np.isnan(result2.raw_chi2))[0]
        print(f"\nNaN indices: {nan_indices[:10]}... (showing first 10)")
        print(f"NaN periods: {periods_large[nan_indices[:10]]}")
        print(f"Valid periods around NaN:")
        for idx in nan_indices[:3]:
            if idx > 0 and idx < len(periods_large) - 1:
                print(f"  [{idx-1}]={periods_large[idx-1]:.3f} (chi2={result2.raw_chi2[idx-1]:.6f})")
                print(f"  [{idx}]={periods_large[idx]:.3f} (chi2={result2.raw_chi2[idx]})")
                print(f"  [{idx+1}]={periods_large[idx+1]:.3f} (chi2={result2.raw_chi2[idx+1]:.6f})")
    
    break

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
