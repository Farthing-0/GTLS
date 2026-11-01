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


def load_module(T0_fit_margin):
    """Compile the CUDA kernels; T0_fit_margin sets the start-point stride of the coarse search."""
    skip_point = "0x7f800000" if T0_fit_margin == 0 else str(int(1 / T0_fit_margin))
    code = KERNEL_SOURCE.read_text().replace("#define SKIP_POINT 8", "#define SKIP_POINT " + skip_point)
    return cp.RawModule(code=code)


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


class GPUSearch:
    """GPU state shared by all chi2 evaluations of one light curve."""

    def __init__(self, module, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview):
        self.module = module
        self.t = t

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
        self.yGPU = cp.asarray(y).astype(cp.float32)
        self.dyGPU = cp.asarray(dy).astype(cp.float32)
        self.tSizeGPU = int_scalar(self.tSize)
        self.tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
        self.patchedDatasSizeGPU = int_scalar(self.patchedDatasSize)
        self.maxDurationGPU = int_scalar(maxDuration)
        self.transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)
        self.constant_residual = float(np.sum(((1.0 - y) / dy) ** 2))

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

    def residuals(self, periods, durationsBool, refine):
        """chi2 of shape (periods, durations[durationsBool], start points).

        refine=False skips start points according to SKIP_POINT (coarse search);
        refine=True evaluates every start point.
        """
        module = self.module
        tSize = self.tSize
        patchedDatasSize = self.patchedDatasSize
        nPeriods = len(periods)

        singleDurations = self.durations[durationsBool]
        single_lc_arr = self.lc_arr[durationsBool]
        overshootGPU = cp.array(self.lc_cache_overview["overshoot"][durationsBool]).astype(cp.float32)
        durationsGPU = cp.asarray(singleDurations).astype(cp.int32)
        durationsSizeGPU = int_scalar(len(singleDurations))
        periodsGPU = cp.asarray(periods).astype(cp.float64)
        periodSizeGPU = int_scalar(nPeriods)

        # Phase fold
        phasesGPU = cp.empty((nPeriods, tSize), dtype=cp.float64)
        sortIndexGPU = cp.empty((nPeriods, tSize), dtype=cp.int32)
        blockSize, gridSizeX = calcGridBlockSize(tSize)
        module.get_function("foldFast")(
            (gridSizeX, nPeriods), (blockSize,), (self.tGPU, periodsGPU, phasesGPU, self.tSizeGPU))

        # Sort in chunks to limit peak GPU memory
        chunks = 10
        for i in range(chunks):
            lo, hi = i * nPeriods // chunks, (i + 1) * nPeriods // chunks
            sortIndexGPU[lo:hi] = phasesGPU[lo:hi].argsort()

        # Folded data, extended by maxDuration points to handle transits at the phase edge
        patchedDatasGPU = cp.empty((nPeriods, patchedDatasSize), dtype=cp.float32)
        patchedDysGPU = cp.empty((nPeriods, patchedDatasSize), dtype=cp.float32)
        blockSize, gridSizeX = calcGridBlockSize(patchedDatasSize)
        module.get_function("patchData")(
            (gridSizeX, nPeriods), (blockSize,),
            (patchedDatasGPU, patchedDysGPU, self.patchedDatasSizeGPU, sortIndexGPU,
             self.maxDurationGPU, self.yGPU, self.dyGPU, self.tSizeGPU))

        inverseSquaredPatchedDysGPU = cp.empty((nPeriods, patchedDatasSize), dtype=cp.float32)
        module.get_function("calcInverseSquaredPatchedDy")(
            (gridSizeX, nPeriods, 1), (blockSize, 1, 1),
            (inverseSquaredPatchedDysGPU, patchedDysGPU, self.patchedDatasSizeGPU))

        edgeEffectCorrectionsGPU = cp.empty((nPeriods), dtype=cp.float32)
        blockSize, gridSizeX = calcGridBlockSize(nPeriods)
        module.get_function("calcEdgeEffectCorrections")(
            (gridSizeX, 1, 1), (blockSize, 1, 1),
            (edgeEffectCorrectionsGPU, patchedDatasGPU, inverseSquaredPatchedDysGPU,
             self.patchedDatasSizeGPU, self.maxDurationGPU, periodSizeGPU))

        cumsumGPU = cp.empty((nPeriods, patchedDatasSize), dtype=cp.float32)
        for i in range(nPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        # Prefix sum of the per-point squared error, shared by the full sums and the coarse ootr
        base_error = cp.empty((nPeriods, patchedDatasSize), dtype=cp.float32)
        module.get_function("calculate_base_error")(
            ((patchedDatasSize + 255) // 256, nPeriods), (256,),
            (base_error, patchedDatasGPU, inverseSquaredPatchedDysGPU, np.int32(patchedDatasSize),
             np.int32(nPeriods)))
        error_prefix_sum = cp.cumsum(base_error, axis=1)
        del base_error

        fullSumGPU = cp.empty((nPeriods, len(singleDurations)), dtype=cp.float32)
        blockSize, gridSizeX = calcGridBlockSize(len(singleDurations))
        module.get_function("calcAllFullSum")(
            (gridSizeX, nPeriods, 1), (blockSize, 1, 1),
            (fullSumGPU, error_prefix_sum, np.int32(patchedDatasSize), durationsGPU,
             np.int32(len(singleDurations)), np.int32(nPeriods)))

        # Out-of-transit residuals for every (period, duration, start point)
        blockSize, gridSizeX = calcGridBlockSize(tSize)
        if refine:
            ootrGPU = cp.empty((nPeriods, len(singleDurations), tSize), dtype=cp.float32)
            module.get_function("calcAllOutOfTransitResiduals_step1_2GPU")(
                (gridSizeX, len(singleDurations), nPeriods), (blockSize, 1, 1),
                (ootrGPU, patchedDatasGPU, durationsGPU, durationsSizeGPU,
                 inverseSquaredPatchedDysGPU, self.patchedDatasSizeGPU, self.tSizeGPU))
            ootrGPU = cp.cumsum(ootrGPU, axis=-1)
            module.get_function("calcAllOutOfTransitResiduals_step2_2GPU")(
                (gridSizeX, len(singleDurations), nPeriods), (blockSize, 1, 1),
                (ootrGPU, durationsSizeGPU, self.tSizeGPU, fullSumGPU))
            residualKernel = "calcAllLowestResidualsNoSkip"
        else:
            ootrGPU = cp.empty((nPeriods, len(singleDurations), tSize), dtype=cp.float32)
            module.get_function("calculate_final_ootr")(
                ((tSize + 255) // 256, len(singleDurations), nPeriods), (256, 1, 1),
                (ootrGPU, error_prefix_sum, fullSumGPU, durationsGPU,
                 tSize, patchedDatasSize, len(singleDurations), nPeriods))
            residualKernel = "calcAllLowestResidualsSkip"

        lc_arr_max_len = np.array([np.max(singleDurations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array(
            [np.pad(x, (0, lc_arr_max_len[0] - len(x)), "constant") for x in single_lc_arr])
        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)

        lowestResidualsGPU = cp.empty((nPeriods, len(singleDurations), tSize), dtype=cp.float32)
        module.get_function(residualKernel)(
            (gridSizeX, len(singleDurations), nPeriods), (blockSize, 1, 1),
            (lowestResidualsGPU, self.tSizeGPU,
             patchedDatasGPU, self.patchedDatasSizeGPU,
             durationsGPU, durationsSizeGPU,
             lcArrFullLengthGPU, lcArrMaxLenGPU, inverseSquaredPatchedDysGPU,
             overshootGPU, ootrGPU, fullSumGPU, edgeEffectCorrectionsGPU, cumsumGPU,
             self.transitDepthMinGPU))
        return lowestResidualsGPU

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

        # Pad the last batch with its last period to avoid division by zero
        if valid_range < size:
            periods = np.append(periods, np.full(size - valid_range, periods[-1]))

        lowestResidualsGPU = self.residuals(periods, durationsBool, refine)
        flattened_residuals = lowestResidualsGPU[:valid_range].reshape(valid_range, -1)
        min_values = cp.min(flattened_residuals, axis=-1)
        min_values = cp.where(cp.isfinite(min_values), min_values, self.constant_residual)
        return min_values.astype(cp.float32).get()


class MultiGPUSearch:
    """One GPUSearch per device; batches of periods are distributed among the devices."""

    def __init__(self, device_ids, T0_fit_margin, *args):
        self.device_ids = list(device_ids)
        self.searches = []
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                self.searches.append(GPUSearch(load_module(T0_fit_margin), *args))

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

    lowestResidualsGPU = search.residuals(np.array([period]), durationsBool, refine=True)
    bestLocation = lowestResidualsGPU.argmin().get()

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
