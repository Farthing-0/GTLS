import numpy as np

from . import constants
from .helpers import running_median, transit_mask


def spectra(chi2, oversampling_factor):
    SR = np.min(chi2) / chi2
    SDE_raw = (1 - np.mean(SR)) / np.std(SR)

    # Scale SDE_power from 0 to SDE_raw
    power_raw = SR - np.mean(SR)  # shift down to the mean being zero
    scale = SDE_raw / np.max(power_raw)  # scale factor to touch max=SDE_raw
    power_raw = power_raw * scale

    # Detrended SDE, named "power"
    kernel = oversampling_factor * constants.SDE_MEDIAN_KERNEL_SIZE
    if kernel % 2 == 0:
        kernel = kernel + 1
    if len(power_raw) > 2 * kernel:
        my_median = running_median(power_raw, kernel)
        power = power_raw - my_median
        # Re-normalize to range between median = 0 and peak = SDE
        # shift down to the mean being zero
        power = power - np.mean(power)
        SDE = np.max(power / np.std(power))
        # scale factor to touch max=SDE
        scale = SDE / np.max(power)
        power = power * scale
    else:
        power = power_raw
        SDE = SDE_raw

    return power, SDE


def all_transit_times(T0, t, period):
    """Return all mid-transit times within t"""

    if T0 < min(t):
        transit_times = [T0 + period]
    else:
        transit_times = [T0]
    previous_transit_time = transit_times[0]
    while True:
        next_transit_time = previous_transit_time + period
        if next_transit_time < (np.min(t) + (np.max(t) - np.min(t))):
            transit_times.append(next_transit_time)
            previous_transit_time = next_transit_time
        else:
            break
    return transit_times


def calcDurationDays(times, period, StartT0, rawDuration):
    """Return estimate for transit duration in days (Alternative method)"""

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def centerFold(time, period, T0):
        """Normal phase folding"""
        T0 = T0 + period/2
        return (time - T0) / period - np.floor((time - T0) / period)

    phases = centerFold(times, period, StartT0)
    phasesIndex = np.argsort(phases)
    phases_sorted = phases[phasesIndex]
    startIdx = find_nearest(phases_sorted,0.5)
    endIdx = startIdx + (np.array(rawDuration) * len(times)).astype(int)
    # Seems duration can not be longer than 0.5
    endPhases = phases_sorted[endIdx]
    DurationDays = (endPhases - 0.5) * period
    return DurationDays


def intransit_stats(t, y, transit_times, transit_duration_in_days):
    """Return all in-transit flux points of odd and even transits, and the point count per transit"""

    all_flux_intransit_odd = np.array([])
    all_flux_intransit_even = np.array([])
    per_transit_count = np.zeros([len(transit_times)])

    for i, mid_transit in enumerate(transit_times):
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            flux_intransit = np.array([])
        else:
            flux_intransit = y[np.where(np.logical_and(t > tmin, t < tmax))]
        per_transit_count[i] = np.size(flux_intransit)

        if i % 2 == 0:
            all_flux_intransit_even = np.append(all_flux_intransit_even, flux_intransit)
        else:
            all_flux_intransit_odd = np.append(all_flux_intransit_odd, flux_intransit)

    return all_flux_intransit_odd, all_flux_intransit_even, per_transit_count


def pink_noise(data, width):
    """Mean standard deviation of all windows of width points, divided by sqrt(width)."""
    std = np.float64(0.0)
    datapoints = len(data) - width + 1
    if datapoints > 0 and width > 0:
        windows = np.lib.stride_tricks.sliding_window_view(data, width)
        # Bound the temporary memory of np.std to about 2**22 values per chunk
        chunk = max(1, 2 ** 22 // width)
        for i in range(0, datapoints, chunk):
            std += np.std(windows[i:i + chunk], axis=1).sum()
    return std / (datapoints * width ** 0.5)


def snr_stats(t, y, period, duration, T0, transit_times, transit_duration_in_days, per_transit_count):
    """Return the pink-noise SNR of each transit"""

    snr_pink_per_transit = np.zeros([len(transit_times)])
    intransit = transit_mask(t, period, 2 * duration, T0)
    flux_ootr = y[~intransit]

    try:
        pinknoise = pink_noise(flux_ootr, int(np.mean(per_transit_count)))
    except Exception:
        pinknoise = np.nan

    if len(flux_ootr) == 0 or np.isnan(np.std(flux_ootr)):
        return snr_pink_per_transit

    for i, mid_transit in enumerate(transit_times):
        tmin = mid_transit - 0.5 * transit_duration_in_days
        tmax = mid_transit + 0.5 * transit_duration_in_days
        if np.isnan(tmin) or np.isnan(tmax):
            continue
        flux_intransit = y[np.where(np.logical_and(t > tmin, t < tmax))]
        if len(flux_intransit) > 0:
            try:
                snr_pink_per_transit[i] = (1 - np.mean(flux_intransit)) / pinknoise
            except Exception:
                snr_pink_per_transit[i] = 0

    return snr_pink_per_transit
