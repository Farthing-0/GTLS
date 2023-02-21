# from main import gtls
from gtls import gtls
import batman

if __name__ == "__main__":
    import numpy
    numpy.random.seed(seed=0)  # reproducibility 

    # Create test data
    time_start = 8
    data_duration = 100
    samples_per_day = 480
    samples = int(data_duration * samples_per_day)
    time = numpy.linspace(time_start, time_start + data_duration, samples)

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = time_start  # time of inferior conjunction; first transit is X days after start
    # ma.per = 10.123  # orbital period
    ma.per = 20.123  # orbital period
    ma.rp = 6371 / 696342  # 6371 planet radius (in units of stellar radii)
    ma.a = 19  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.4, 0.4]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, time)  # initializes model
    synthetic_signal = m.light_curve(ma)  # calculates light curve

    # Create noise and merge with flux
    ppm = 50  # Noise level in parts per million
    noise = numpy.random.normal(0, 10**-6 * ppm, int(samples))
    flux = synthetic_signal + noise

    # Plot raw data
    # import matplotlib.pyplot as plt
    # from matplotlib import rcParams; rcParams["figure.dpi"] = 150
    # import sys

    # plt.figure()
    # ax = plt.gca()
    # ax.scatter(time, flux, color='black', s=1)
    # ax.set_ylabel("Flux")
    # ax.set_xlabel("Time (days)")
    # plt.xlim(min(time), max(time))
    # plt.ylim(0.999, 1.001);

    model = gtls(time, flux)

    period, duration, depth, T0, SDE = model.power()
    print('period', period, 'duration', duration, 'depth', depth, 'T0', T0)
    print('SDE', SDE)