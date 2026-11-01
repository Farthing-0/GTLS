import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np
import numpy.ma as ma
import tqdm

from .helpers import transit_mask
from .stats import all_transit_times, calcDurationDays, intransit_stats, snr_stats, spectra

KERNEL_SOURCE = Path(__file__).with_name("kernels.cu")


def load_module():
    """Compile the CUDA kernels."""
    return cp.RawModule(code=KERNEL_SOURCE.read_text())


def foldCPU(time, flux, dy, period):
    """Fold time series data."""
    phase = (time % period) / period
    rank = np.argsort(phase)
    return np.array(time[rank]), flux[rank], dy[rank]


def set_cuda_device(device_id):
    """Set the CUDA device."""
    cp.cuda.Device(device_id).use()


def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    blockSize = min(size, MAX_BLOCK_SIZE)
    gridSizeX = int((size / blockSize) + 1)
    return blockSize, gridSizeX


def int_scalar(value):
    return cp.asarray(np.array([value])).astype(cp.int32)


def find_nearest_indices(a, b):
    a = np.array(a)
    b = np.array(b)

    nearest_indices = np.zeros(a.shape, dtype=int)
    nearest_elements = np.zeros(a.shape)

    for i, val in enumerate(a):
        nearest_index = np.argmin(np.abs(b - val))
        nearest_indices[i] = nearest_index
        nearest_elements[i] = b[nearest_index]

    return nearest_indices, nearest_elements


EMPTY_KEY = 0xFFFFFFFFFFFFFFFF


def start_strides(widths, T0_fit_margin, refine):
    """Start-point stride of each duration (in samples).

    The coarse search tests start points about T0_fit_margin of a duration apart (every point
    for T0_fit_margin = 0); the refinement tests every point.
    """
    widths = np.asarray(widths)
    if refine or T0_fit_margin == 0:
        return np.ones(len(widths), dtype=np.int32)
    skip = int(1 / T0_fit_margin)
    return np.where(widths > skip, widths // skip, 1).astype(np.int32)


def decode_keys(keys):
    """Split searchChi2 keys into (chi2, flattened duration/start location).

    chi2 is NaN where no window was searched (key left at EMPTY_KEY).
    """
    high = (keys >> 32).astype(cp.uint32)
    bits = cp.where(high >= 0x80000000, high ^ 0x80000000, ~high).astype(cp.uint32)
    return bits.view(cp.float32), (keys & 0xFFFFFFFF).astype(cp.int64)


class GPUSearch:
    """GPU state shared by all chi2 evaluations of one light curve."""

    def __init__(self, module, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview, T0_fit_margin):
        self.module = module
        self.t = t
        self.T0_fit_margin = T0_fit_margin

        durations, indices = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
        self.durations = durations
        self.lc_arr = lc_arr[indices]
        self.lc_cache_overview = lc_cache_overview[indices]
        maxDuration = int(max(durations))
        if maxDuration % 2 != 0:
            maxDuration = maxDuration + 1
        self.maxDuration = maxDuration

        self.tSize = len(t)
        self.patchedDatasSize = int(self.tSize + maxDuration)

        self.tGPU = cp.asarray(t).astype(cp.float64)
        # 1 - y is formed in float64, so the float32 residuals keep their full precision
        self.residualGPU = cp.asarray((1.0 - np.asarray(y, dtype=np.float64)).astype(np.float32))
        self.dyGPU = cp.asarray(dy).astype(cp.float32)
        self.tSizeGPU = int_scalar(self.tSize)
        self.tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
        self.transit_depth_min = np.float32(transit_depth_min)
        self.constant_residual = float(np.sum(((1.0 - y) / dy) ** 2))

        # Template depth profiles s = 1 - lc, one row per duration
        self.maxWidth = int(np.max(durations))
        signal = np.zeros((len(durations), self.maxWidth), dtype=np.float32)
        for i, row in enumerate(self.lc_arr):
            signal[i, :len(row)] = 1 - np.asarray(row)
        self.signalGPU = cp.asarray(signal)
        self.overshootGPU = cp.asarray(self.lc_cache_overview["overshoot"]).astype(cp.float32)
        self.widthsGPU = cp.asarray(durations).astype(cp.int32)

    def duration_mask(self, periods):
        """Boolean array (periods x durations) of physically plausible durations per period."""
        module = self.module
        periodsGPU = cp.asarray(periods).astype(cp.float64)
        durationsMaxGPU = cp.empty((len(periods),), dtype=cp.int32)
        durationsMinGPU = cp.empty((len(periods),), dtype=cp.int32)
        periodsSizeGPU = int_scalar(len(periods))

        blockSize, gridSizeX = calcGridBlockSize(len(periods))
        module.get_function("durationsGrid")(
            (gridSizeX, 1, 1), (blockSize,),
            (periodsGPU, durationsMaxGPU, durationsMinGPU, self.tLengthGPU, self.tSizeGPU, periodsSizeGPU))

        durationBoolArrayGPU = cp.empty((len(periods), len(self.durations)), dtype=cp.bool_)
        module.get_function("durationBool")(
            (gridSizeX, len(self.durations), 1), (blockSize, 1, 1),
            (durationsMaxGPU, durationsMinGPU, int_scalar(len(self.durations)), periodsSizeGPU,
             cp.asarray(self.durations).astype(cp.int32), durationBoolArrayGPU))
        return durationBoolArrayGPU

    def best_keys(self, periods, durationsBool, refine):
        """Packed lowest chi2 and its location of each period (see searchChi2 and decode_keys).

        The location is durationIndex * tSize + start, with durationIndex counting the
        durations selected by durationsBool. refine=False tests start points with the coarse
        stride of start_strides; refine=True tests every start point.
        """
        module = self.module
        tSize = self.tSize
        patchedSize = self.patchedDatasSize
        nPeriods = len(periods)

        selected = np.flatnonzero(durationsBool).astype(np.int32)
        strides = start_strides(self.durations[selected], self.T0_fit_margin, refine)
        offsets = np.concatenate(([0], np.cumsum(-(-tSize // strides))))
        if offsets[-1] >= 2 ** 31:
            raise ValueError("Too many start points per period for the GPU search")

        # Phase fold and sort (bucket sort, equal to a stable argsort of the phases)
        periodsGPU = cp.asarray(periods).astype(cp.float64)
        phasesGPU = cp.empty((nPeriods, tSize), dtype=cp.float64)
        countsGPU = cp.zeros((nPeriods, tSize), dtype=cp.int32)
        grid = ((tSize + 255) // 256, nPeriods)
        module.get_function("foldBuckets")(
            grid, (256,), (self.tGPU, periodsGPU, phasesGPU, countsGPU, np.int32(tSize)))
        cursorGPU = cp.cumsum(countsGPU, axis=1, dtype=cp.int32) - countsGPU
        sortIndexGPU = cp.empty((nPeriods, tSize), dtype=cp.int32)
        module.get_function("bucketScatter")(
            grid, (256,), (phasesGPU, cursorGPU, sortIndexGPU, np.int32(tSize)))
        module.get_function("sortBuckets")(
            grid, (256,), (phasesGPU, countsGPU, cursorGPU, sortIndexGPU, np.int32(tSize)))
        del phasesGPU, countsGPU, cursorGPU

        # Folded data, extended by maxDuration points to handle transits at the phase edge
        residualGPU = cp.empty((nPeriods, patchedSize), dtype=cp.float32)
        weightedGPU = cp.empty((nPeriods, patchedSize, 2), dtype=cp.float32)
        module.get_function("patchWeighted")(
            ((patchedSize + 255) // 256, nPeriods), (256,),
            (sortIndexGPU, self.residualGPU, self.dyGPU, residualGPU, weightedGPU,
             np.int32(tSize), np.int32(patchedSize)))
        del sortIndexGPU

        cumResidualGPU = cp.cumsum(residualGPU, axis=1, dtype=cp.float64)
        baseGPU = cp.sum(weightedGPU[:, :tSize, 0].astype(cp.float64) * residualGPU[:, :tSize], axis=1)
        del residualGPU

        bestGPU = cp.full(nPeriods, EMPTY_KEY, dtype=cp.uint64)
        module.get_function("searchChi2")(
            ((int(offsets[-1]) + 255) // 256, nPeriods), (256,),
            (bestGPU, cumResidualGPU, weightedGPU, baseGPU, self.signalGPU, self.overshootGPU,
             self.widthsGPU, cp.asarray(selected), cp.asarray(offsets.astype(np.int32)),
             cp.asarray(strides), np.int32(len(selected)), np.int32(tSize), np.int32(patchedSize),
             np.int32(self.maxWidth), self.transit_depth_min))
        return bestGPU

    def chi2_batch(self, periods, size, refine):
        """Lowest chi2 of each period of one batch, padded to size periods.

        All periods of a batch are searched over the union of their plausible durations,
        so results depend on the batching, not on the device evaluating each batch.
        """
        valid_range = len(periods)
        # Durations needed by any period of this batch
        durationsBool = cp.any(self.duration_mask(periods), axis=0).get()
        if not durationsBool.any():
            return np.full(valid_range, self.constant_residual, dtype=np.float32)

        # Pad the last batch with its last period, so all batches have the same shape
        if valid_range < size:
            periods = np.append(periods, np.full(size - valid_range, periods[-1]))

        chi2, _ = decode_keys(self.best_keys(periods, durationsBool, refine)[:valid_range])
        chi2 = cp.where(cp.isfinite(chi2), chi2, self.constant_residual)
        return chi2.astype(cp.float32).get()


class MultiGPUSearch:
    """One GPUSearch per device; batches of periods are distributed among the devices."""

    def __init__(self, device_ids, T0_fit_margin, *args):
        self.device_ids = list(device_ids)
        self.searches = []
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                self.searches.append(GPUSearch(load_module(), *args, T0_fit_margin=T0_fit_margin))

    @property
    def primary(self):
        return self.searches[0]

    def batch_size(self, n_periods):
        """Periods per batch, limited by the free memory of the fullest device."""
        free = min(cp.cuda.Device(device_id).mem_info[0] for device_id in self.device_ids)
        patchedDatasSize = self.primary.patchedDatasSize
        nDurations = len(self.primary.durations)
        singleCalcPeriods_max = free / (
            5 * (patchedDatasSize * 2 + 2 + nDurations * patchedDatasSize * 4 + 2 * nDurations))
        singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max), n_periods / 30]))
        if singleCalcPeriods < 15:
            singleCalcPeriods = int(singleCalcPeriods / 1.1)
        return max(singleCalcPeriods, 1)

    def chi2(self, periods, singleCalcPeriods, refine, pbar=None):
        """Lowest chi2 of each period, evaluated in batches of singleCalcPeriods periods."""
        periods = np.asarray(periods)
        chi2 = np.empty(len(periods), dtype=np.float32)
        batches = iter(range(0, len(periods), singleCalcPeriods))
        lock = threading.Lock()

        def worker(device_id, search):
            with cp.cuda.Device(device_id):
                while True:
                    with lock:
                        start_idx = next(batches, None)
                    if start_idx is None:
                        return
                    end_idx = start_idx + singleCalcPeriods
                    chi2[start_idx:end_idx] = search.chi2_batch(periods[start_idx:end_idx], singleCalcPeriods, refine)
                    if pbar is not None:
                        with lock:
                            pbar.update(1)

        if len(self.searches) == 1:
            worker(self.device_ids[0], self.primary)
        else:
            with ThreadPoolExecutor(max_workers=len(self.searches)) as executor:
                futures = [executor.submit(worker, device_id, search)
                           for device_id, search in zip(self.device_ids, self.searches)]
                for future in futures:
                    future.result()
        return chi2

    def close(self):
        """Release the GPU arrays and return the cached memory to the driver."""
        self.searches = []
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def search_multi_periods(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose,
    GPUDeviceIDs=(0,),
    # fast: just return periods and power, not the full result
    fast=False,
    bar_location=0,
):
    set_cuda_device(GPUDeviceIDs[0])
    search = MultiGPUSearch(GPUDeviceIDs, T0_fit_margin, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview)
    try:
        return _search_multi_periods(search, periods, t, y, dy, oversampling_factor, verbose, fast, bar_location)
    finally:
        search.close()


def _search_multi_periods(search, periods, t, y, dy, oversampling_factor, verbose, fast, bar_location):
    # Due to GPU memory size limitation, GPU can only do several periods at a time.
    singleCalcPeriods = search.batch_size(len(periods))
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    pbar = tqdm.tqdm(total=TotalIter, position=bar_location) if verbose else None
    raw_chi2 = search.chi2(periods, singleCalcPeriods, refine=False, pbar=pbar)
    if pbar is not None:
        pbar.close()

    median = np.median(raw_chi2)
    chi2_mask = raw_chi2 > (100 * median)
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods = ma.array(periods, mask=chi2_mask)

    power, _ = spectra(chi2, oversampling_factor)
    raw_power = power.copy()

    if fast:
        return periods, power

    # Refine the 100 strongest peaks, plus the 100 strongest peaks with period > 1 day,
    # without skipping start points
    combined = list(enumerate(zip(periods, -power)))
    sorted_combined = sorted(combined, key=lambda x: x[1][1])
    top_100_indices = [item[0] for item in sorted_combined[:100]]
    top_100_periods = [item[1][0] for item in sorted_combined[:100]]
    remaining_combined = [item for item in sorted_combined if item[0] not in top_100_indices]
    remaining_combined_greater_than_1 = [item for item in remaining_combined if item[1][0] > 1]
    sorted_remaining_combined_greater_than_1 = sorted(remaining_combined_greater_than_1, key=lambda x: x[1][1])
    next_100_indices = [item[0] for item in sorted_remaining_combined_greater_than_1[:100]]
    next_100_periods = [item[1][0] for item in sorted_remaining_combined_greater_than_1[:100]]

    possiblePeriodsIndices = top_100_indices + next_100_indices
    possiblePeriods = top_100_periods + next_100_periods

    chi2[possiblePeriodsIndices] = search.chi2(possiblePeriods, singleCalcPeriods, refine=True)

    power, _ = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]

    # Check aliases of the best period
    possiblePeriodsTimesRate = [0.5, 1, 2, 2 / 3, 3 / 2]
    possiblePeriodsTemp = [period * rate for rate in possiblePeriodsTimesRate]
    possiblePeriodsIndices_multi, possiblePeriods_multi = find_nearest_indices(possiblePeriodsTemp, periods)

    chi2[possiblePeriodsIndices_multi] = search.chi2(possiblePeriods_multi, singleCalcPeriods, refine=True)

    power, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices_multi]
    periodIndex = possiblePeriodsIndices_multi[np.argmax(power_again)]
    period = periods[periodIndex]

    rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, transit_times, \
        snr, snr_pink, snrFit, snrFitPink = search_single_period(search.primary, period, t, y, dy)

    return periods, period, rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, \
        SDE, chi2, transit_times, power, snr, snr_pink, snrFit, snrFitPink, raw_power, raw_chi2, \
        possiblePeriodsIndices, possiblePeriods


def search_single_period(search, period, t, y, dy):
    """Refit duration, T0 and depth at the best period, since the coarse search skips start points."""
    tSize = search.tSize
    maxDuration = search.maxDuration

    durationsBool = search.duration_mask(np.array([period]))[0].get()
    durations = search.durations[durationsBool]
    single_lc_cache_overview = search.lc_cache_overview[durationsBool]

    chi2, location = decode_keys(search.best_keys(np.array([period]), durationsBool, refine=True))
    bestLocation = int(location[0]) if bool(cp.isfinite(chi2[0])) else 0

    durationIndex = np.floor(bestLocation / (tSize)).astype(int)
    durationPointsNum = durations[durationIndex]
    rawDuration = single_lc_cache_overview["duration"][durationIndex]

    bestTime, bestFlux, _ = foldCPU(t, y, dy, period)
    bestFlux = np.concatenate((bestFlux, bestFlux[:maxDuration]))

    bestRowT0 = bestLocation % (tSize)

    transitMean = bestFlux[bestRowT0:bestRowT0 + durationPointsNum].mean()

    # Transit Depth
    overshoot = single_lc_cache_overview["overshoot"][durationIndex]
    transitDepth = ((1 - transitMean) * overshoot).item()

    dataOutTransit = np.concatenate((bestFlux[0:bestRowT0], bestFlux[bestRowT0 + durationPointsNum:]))

    snrFit = (1 - transitDepth) * (durationPointsNum ** 0.5) / np.std(dataOutTransit)
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durationPointsNum:] - DataCumsum[:-durationPointsNum]) / durationPointsNum
    redNoise = np.std(DataSlideAvg)

    Tx = bestTime[bestRowT0]
    T0 = Tx - int((Tx - min(t)) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = (1 - transitDepth) / (
        (np.std(dataOutTransit) ** 2 / (durationPointsNum)) + (redNoise ** 2 / (len(transit_times)))) ** 0.5

    transit_duration_in_days = calcDurationDays(t, period, T0, rawDuration)

    T0 = T0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if T0 < min(t):
        T0 = T0 + period

    # SNR
    all_flux_intransit_odd, all_flux_intransit_even, per_transit_count = intransit_stats(
        t, y, transit_times, transit_duration_in_days)
    snr_pink_per_transit = snr_stats(
        t=t,
        y=y,
        period=period,
        duration=rawDuration,
        T0=T0,
        transit_times=transit_times,
        transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count,
    )

    all_flux_intransit = np.concatenate([all_flux_intransit_odd, all_flux_intransit_even])
    intransit = transit_mask(t, period, 2 * rawDuration, T0)
    flux_ootr = y[~intransit]
    depth_mean = np.mean(all_flux_intransit)
    snr = ((1 - depth_mean) / np.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)

    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times) ** (0.5))

    return rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, transit_times, \
        snr, snr_pink, snrFit, snrFitPink
