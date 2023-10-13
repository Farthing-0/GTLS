import numpy as np
import batman
import matplotlib.pyplot as plt
from matplotlib import rcParams; rcParams["figure.dpi"] = 150
from transitleastsquares import transitleastsquares
from gputls import gtls
from tqdm import tqdm
import time

days = 100
print('days',days)
np.random.seed(0)
period = np.random.uniform(1, 20)
depth = np.random.uniform(2e-5,0.03)
duration = np.random.uniform(0.01, 0.5)
time_start = 1
T0 = np.random.uniform(time_start, period+time_start)
ppm = 1000

semiMajorAxis = period / (np.pi * duration)

# Create test data
data_duration = days #Unit: days
samples_per_day = 720
samples = int(data_duration * samples_per_day)
times = np.linspace(time_start, time_start + data_duration, samples)

# Use batman to create transits
ma = batman.TransitParams()
ma.t0 = T0  # times of inferior conjunction; first transit is X days after start
ma.per = period  # orbital period
# ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
ma.rp = np.sqrt(depth)  # planet radius (in units of stellar radii)
ma.a = semiMajorAxis  # semi-major axis (in units of stellar radii)
ma.inc = 90  # orbital inclination (in degrees)
ma.ecc = 0  # eccentricity
ma.w = 90  # longitude of periastron (in degrees)
ma.u = [0.4, 0.4]  # limb darkening coefficients
ma.limb_dark = "quadratic"  # limb darkening model
m = batman.TransitModel(ma, times)  # initializes model
synthetic_signal = m.light_curve(ma)  # calculates light curve

noise = np.random.normal(0, 10**-6 * ppm, int(samples))
flux = synthetic_signal + noise

# # TLS
# start = time.time()
# model = transitleastsquares(times, flux)
# results = model.power()
# tlsTime = time.time() - start
# # totalTime += tlsTime

# GTLS
start = time.time()
model = gtls(t = times, y = flux)
# gtlsResult = model.power(GPUDeviceID = 1)
gtlsResult = model.power()
gtlsTime = time.time() - start

print("TrueResult")
print("period: ",period,"duration: ",duration,"depth: ",depth,"T0: ",T0,"noisePPM: ",ppm)
print("GTLSResult")
print("period: ",gtlsResult.period,"duration: ",gtlsResult.duration,"depth: ",gtlsResult.depth,"T0: ",gtlsResult.T0,"SDE: ",gtlsResult.SDE,"snr: ",gtlsResult.snr,"snrPink: ",gtlsResult.snrPink,"snrFit: ",gtlsResult.snrFit,"snrFitPink: ",gtlsResult.snrFitPink,"lossSDE: ",gtlsResult.lossSDE,"KLossMean: ",gtlsResult.KLossMean,"KLossStd: ",gtlsResult.KLossStd)
