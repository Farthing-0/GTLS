import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np
import numpy.ma as ma
import tqdm

from . import constants
from .grid import duration_limit_masses
from .helpers import transit_mask
from .screening import (certificates, MIN_WIDTH, SCREEN_TOLERANCES,
                        UNIFORM_PREFIX_STRIDE, GENERAL_PREFIX_STRIDE,
                        uniform_sparse_certificate, sparse_certificate, prefix_quantization)
from .stats import all_transit_times, calcDurationDays, intransit_stats, snr_stats, spectra
from .workspace import (BatchWorkspace, LAYOUT_ALLOWANCE, bucket_count,
                        bytes_per_period as workspace_bytes_per_period)

KERNEL_SOURCE = Path(__file__).with_name("kernels.cu")


def load_module():
    """Compile the CUDA kernels."""
    # The extra FP64 bound pays off on A100-class devices; the 3090's 1:64 FP64
    # rate showed no gain. Compile out the shortcut on those devices entirely.
    ratio = cp.cuda.Device().attributes.get("SingleToDoublePrecisionPerfRatio", 64)
    options = (f"-DGTLS_CHEAP_SCREEN_BOUNDS={int(ratio <= 2)}",)
    return cp.RawModule(code=KERNEL_SOURCE.read_text(), options=options)


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
SCAN_THREADS = 512  # threads per row of the integer scan kernels
SCREEN_SCAN_THREADS = 32  # 16 samples per thread: a 512-point screening tile
SCREEN_THREADS = 32  # nearby starts share prefix cache lines across durations
MAX_BATCH = 65535  # periods are the y dimension of the kernel grids
BATCHES_MIN = 30  # the first pass is split into at least this many batches
MEMORY_FRACTION = 0.9  # share of the free device memory used by one batch


def start_strides(widths, margin):
    """Start-point stride of each duration (in samples): about margin of the duration,
    every point for margin = 0."""
    widths = np.asarray(widths)
    if margin == 0:
        return np.ones(len(widths), dtype=np.int32)
    skip = int(1 / margin)
    return np.where(widths > skip, widths // skip, 1).astype(np.int32)


def screen_block_map(offsets, strides):
    """Schedule contiguous candidate blocks by start time, retaining original IDs."""
    counts = (np.diff(offsets) + SCREEN_THREADS - 1) // SCREEN_THREADS
    duration = np.repeat(np.arange(len(strides), dtype=np.int32), counts)
    item = np.concatenate([np.arange(left, right, SCREEN_THREADS, dtype=np.int32)
                           for left, right in zip(offsets[:-1], offsets[1:])])
    start = (item - offsets[duration]) * strides[duration]
    order = np.argsort(start, kind='stable')
    return np.column_stack((duration[order], item[order])).astype(np.int32)


WINDOW_LANES = (1, 2, 4, 8, 16, 32)  # threads per window of the searchChi2G* kernels
LANE_STEP_BYTES = 64  # bytes of start stride per thread of a window, see lanes_per_window
POINT_BYTES = 8  # bytes read per point by searchChi2 (float2 of w r and w)


def lanes_per_window(strides):
    """Threads sharing one searchChi2 window: the smallest G in WINDOW_LANES with
    stride * POINT_BYTES / G <= LANE_STEP_BYTES, at most 32.

    One read instruction of a warp covers 32 / G windows of G consecutive points each, spaced
    stride points apart: it spans stride / G times the bytes it uses, and the overlap of
    neighbouring windows is reused stride / G steps later. Keeping stride / G <= 8 keeps both
    small (bench/window_lanes on branch perf/window-lanes: within a few percent of the fastest
    G for every stride, data size, batch size and L1 size measured). Depends only on the
    stride, not on the device.
    """
    needed = np.asarray(strides, dtype=np.int64) * POINT_BYTES / LANE_STEP_BYTES
    lanes = np.full(needed.shape, WINDOW_LANES[-1], dtype=np.int32)
    for g in reversed(WINDOW_LANES):
        lanes[needed <= g] = g
    return lanes


def duration_limit_args(R_star_min, R_star_max, M_star_min, M_star_max):
    """Scalar arguments (rShort, gmShort, rLong, gmLong, upperLimit) of the durationsGrid kernel.

    Same terms as grid.T14: the shortest duration pairs R_star_min (small planet) with the
    mass m_short, the longest R_star_max (planet of 2 R_jup) with m_long, as TLS 2.
    """
    m_short, m_long = duration_limit_masses(R_star_min, R_star_max, M_star_min, M_star_max)
    pi_G = np.pi * constants.G
    return (
        np.float64(constants.R_sun * R_star_min),
        np.float64(pi_G * (constants.M_sun * m_short)),
        np.float64(constants.R_sun * R_star_max + 2 * constants.R_jup),
        np.float64(pi_G * (constants.M_sun * m_long)),
        np.float64(constants.FRACTIONAL_TRANSIT_DURATION_MAX),
    )


def decode_keys(keys, base):
    """Split searchChi2 keys into (chi2, flattened duration/start location).

    The keys hold delta = chi2 - base in float32; chi2 = float32(base + delta), base being the
    chi2 without transit (GPUSearch.base). chi2 is NaN where no window was searched (key left
    at EMPTY_KEY).
    """
    high = (keys >> 32).astype(cp.uint32)
    bits = cp.where(high >= 0x80000000, high ^ 0x80000000, ~high).astype(cp.uint32)
    chi2 = (base + bits.view(cp.float32).astype(cp.float64)).astype(cp.float32)
    return chi2, (keys & 0xFFFFFFFF).astype(cp.int64)


class GPUSearch:
    """GPU state shared by all chi2 evaluations of one light curve."""

    def window_standard_deviations(self, data, width):
        """Reuse completed search buffers for the final pink-noise statistic."""
        size = len(data)
        count = size - width + 1
        with cp.cuda.Device(self.tGPU.device.id):
            source = self.workspace.view('scratch', (size,), cp.float64)
            output = self.workspace.view('cumulative', (count,), cp.float64)
            source.set(np.ascontiguousarray(data))
            self.module.get_function('windowStd')(
                ((count + 255) // 256,), (256,),
                (source, output, np.int32(count), np.int32(width)))
            return output.get()

    def __init__(self, module, t, y, dy, transit_depth_min, lc_arr, lc_cache_overview, T0_fit_margin,
                 T0_refine_margin, R_star_min=constants.R_STAR_MIN, R_star_max=constants.R_STAR_MAX,
                 M_star_min=constants.M_STAR_MIN, M_star_max=constants.M_STAR_MAX):
        self.module = module
        self.t = t
        self.T0_fit_margin = T0_fit_margin
        self.T0_refine_margin = T0_refine_margin
        self.duration_limits = duration_limit_args(R_star_min, R_star_max, M_star_min, M_star_max)

        durations, indices = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
        self.durations = durations
        self.lc_arr = lc_arr[indices]
        self.lc_cache_overview = lc_cache_overview[indices]
        maxDuration = int(max(durations))
        if maxDuration % 2 != 0:
            maxDuration = maxDuration + 1
        self.maxDuration = maxDuration

        self.tSize = len(t)
        self.nBuckets = bucket_count(self.tSize)
        self.patchedDatasSize = int(self.tSize + maxDuration)

        self.tGPU = cp.asarray(t).astype(cp.float64)
        # 1 - y is formed in float64, so the float32 residuals keep their full precision
        self.residualGPU = cp.asarray((1.0 - np.asarray(y, dtype=np.float64)).astype(np.float32))
        self.dyGPU = cp.asarray(dy).astype(cp.float32)
        self.tSizeGPU = int_scalar(self.tSize)
        self.tLengthGPU = cp.asarray(np.array([np.max(t) - np.min(t)], dtype=np.float64))
        self.transit_depth_min = np.float32(transit_depth_min)
        self.constant_residual = float(np.sum(((1.0 - y) / dy) ** 2))

        # Rebuild from this call's dy; only the per-period permutation is repeated.
        self.preparedGPU = cp.empty((self.tSize, 2), dtype=cp.float32)
        module.get_function("prepareWeights")(
            ((self.tSize + 255) // 256,), (256,),
            (self.residualGPU, self.dyGPU, self.preparedGPU, np.int32(self.tSize)))

        # chi2 without transit, sum of w r^2 as searchChi2 weights the points:
        # it does not depend on the order of the points, so it is the same for every period
        dy2 = cp.maximum(self.dyGPU * self.dyGPU, cp.float32(1e-10))
        weightedResidual = (cp.float32(1.0) / dy2) * self.residualGPU
        self.base = float(cp.sum(weightedResidual.astype(cp.float64) * self.residualGPU.astype(cp.float64)))

        # Fixed-point scale 2^E of the residual prefix sums: the largest E with
        # patchedDatasSize * max|r| * 2^E < 2^62, so no prefix sum can overflow int64
        bound = self.patchedDatasSize * float(cp.max(cp.abs(self.residualGPU)))
        exponent = 62 - int(np.ceil(np.log2(bound))) if bound > 0 else 62
        exponent = int(np.clip(exponent, -100, 100))
        self.fixedScale = np.float32(2.0 ** exponent)
        self.fixedInvScale = np.float32(2.0 ** -exponent)

        # Template depth profiles s = 1 - lc, one row per duration
        self.maxWidth = int(np.max(durations))
        signal = np.zeros((len(durations), self.maxWidth), dtype=np.float32)
        for i, row in enumerate(self.lc_arr):
            signal[i, :len(row)] = 1 - np.asarray(row)
        self.signalGPU = cp.asarray(signal)
        self.overshootGPU = cp.asarray(self.lc_cache_overview["overshoot"]).astype(cp.float32)
        self.widthsGPU = cp.asarray(durations).astype(cp.int32)

        # Screening is useful for long templates and a sampled start grid. The final
        # every-point fit and small curves retain the adjacent-window path.
        self.screening = (self.tSize >= 8192 and self.maxWidth >= MIN_WIDTH
                          and min(T0_fit_margin, T0_refine_margin) > 0)
        self.uniformWeight = None
        self.screenPrefixStride = 1
        self.screenQuantization = None
        if self.screening:
            max_z = float(cp.max(cp.abs(weightedResidual)))
            max_w = float(cp.max(cp.float32(1) / dy2))
            self.screening = bool(np.isfinite(max_z) and np.isfinite(max_w))
        # First-time Numba compilation can retain its caller frame until GC.
        # Drop temporary device arrays before entering that compiler.
        del dy2, weightedResidual
        if self.screening:
            self.screenPrefixStride = GENERAL_PREFIX_STRIDE
            # Check this call's actual FP32 weights, including the variance
            # floor. No uniform-weight assumption survives across searches.
            if bool(cp.all(self.preparedGPU[:, 1] == self.preparedGPU[0, 1])):
                self.uniformWeight = np.float32(self.preparedGPU[0, 1].get())
                self.screenPrefixStride = UNIFORM_PREFIX_STRIDE
            stride = self.screenPrefixStride
            certificateSize = (self.patchedDatasSize+stride-1)//stride*stride
            if cp.cuda.Device().attributes.get("SingleToDoublePrecisionPerfRatio", 64) > 2:
                self.screenQuantization = prefix_quantization(certificateSize, max_z, max_w)
            point_error = None if self.screenQuantization is None else self.screenQuantization[2]
            prefix_maxima = (np.array([max_z, max_w]) if point_error is None else
                             np.nextafter(np.array([max_z, max_w])+point_error, np.inf))
            self.screenCertificates = []
            for tolerance in SCREEN_TOLERANCES:
                pos, coef, meta, mass_error = certificates(
                    signal, durations, max_z, max_w, certificateSize, tolerance, point_error)
                if self.uniformWeight is not None:
                    pos, coef, meta = uniform_sparse_certificate(
                        pos, coef, meta, stride, certificateSize, prefix_maxima[0])
                else:
                    pos, coef, meta = sparse_certificate(
                        pos, coef, meta, stride, certificateSize, *prefix_maxima)
                self.screenCertificates.append((cp.asarray(pos), cp.asarray(coef),
                                                cp.asarray(meta), *map(np.float64, mass_error)))
        self.workspace = None
        self.screenMapKey = None
        self.screenMapGPU = None
        self.screenBudgetsGPU = None
        self.uniformSquareGPU = (cp.empty(len(self.durations), dtype=cp.float32)
                                 if self.uniformWeight is not None else None)
        self.screenPrefixBytes = 12 if self.uniformWeight is not None else 24
        self.screenPrefixSize = ((self.patchedDatasSize+self.screenPrefixStride-1)//self.screenPrefixStride+1
                                 if self.screenPrefixStride > 1 else self.patchedDatasSize)
        self.weightedPointBytes = 4 if self.uniformWeight is not None else 8
        self.maxScreenStarts = (int(np.sum(-(-self.tSize // start_strides(
            self.durations, min(T0_fit_margin, T0_refine_margin))))) if self.screening else 0)

    def close(self):
        """Release device resources even if a compiler/debugger retains this object."""
        self.workspace = None
        if hasattr(self, 'screenCertificates'):
            self.screenCertificates.clear()
        for name, value in vars(self).items():
            if isinstance(value, cp.ndarray):
                setattr(self, name, None)

    def reserve_workspace(self, capacity):
        """Keep the largest batch allocation for the search, including short tail/refit calls."""
        if self.workspace is None or self.workspace.capacity < capacity:
            self.workspace = None
            self.workspace = BatchWorkspace(capacity, self.tSize, self.patchedDatasSize,
                                            self.maxScreenStarts, self.screening,
                                            prefix_bytes=self.screenPrefixBytes,
                                            weighted_bytes=self.weightedPointBytes,
                                            prefix_points=self.screenPrefixSize)
        return self.workspace

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
            (periodsGPU, durationsMaxGPU, durationsMinGPU, self.tLengthGPU, self.tSizeGPU, periodsSizeGPU)
            + self.duration_limits)

        durationBoolArrayGPU = cp.empty((len(periods), len(self.durations)), dtype=cp.bool_)
        module.get_function("durationBool")(
            (gridSizeX, len(self.durations), 1), (blockSize, 1, 1),
            (durationsMaxGPU, durationsMinGPU, int_scalar(len(self.durations)), periodsSizeGPU,
             cp.asarray(self.durations).astype(cp.int32), durationBoolArrayGPU))
        return durationBoolArrayGPU

    def best_keys(self, periods, periodMaskGPU, margin):
        """Packed lowest chi2 and its location of each period (see searchChi2 and decode_keys).

        periodMaskGPU (periods x durations, from duration_mask) selects the durations searched
        for each period. The location is durationIndex * tSize + start, with durationIndex
        counting the durations used by any period of the batch. Start points are tested about
        margin of a duration apart (every point for margin = 0).
        """
        module = self.module
        tSize = self.tSize
        patchedSize = self.patchedDatasSize
        nPeriods = len(periods)
        workspace = self.reserve_workspace(nPeriods)
        # Offsets use the allocated capacity; active views have tightly packed rows
        # for the current (possibly shorter) batch, as expected by the CUDA kernels.
        index_bytes = workspace.index_bytes

        durationsBool = cp.any(periodMaskGPU, axis=0).get()
        selected = np.flatnonzero(durationsBool).astype(np.int32)
        strides = start_strides(self.durations[selected], margin)
        offsets = np.concatenate(([0], np.cumsum(-(-tSize // strides))))
        if offsets[-1] >= 2 ** 31:
            raise ValueError("Too many start points per period for the GPU search")

        # Phase fold and sort (bucket sort, equal to a stable argsort of the phases)
        inversePeriodsGPU = cp.asarray(1.0 / np.asarray(periods, dtype=np.float64))
        phasesGPU = workspace.view('scratch', (nPeriods, tSize), cp.uint32, index_bytes)
        nBuckets = self.nBuckets
        countsGPU = workspace.view('scratch', (nPeriods, nBuckets), cp.int32, 2*index_bytes)
        countsGPU.fill(0)
        grid = ((tSize + 255) // 256, nPeriods)
        module.get_function("foldBucketsPackedPhase")(
            grid, (256,), (self.tGPU, inversePeriodsGPU, phasesGPU, countsGPU, np.int32(tSize), np.int32(nBuckets)))
        cursorGPU = workspace.view('scratch', (nPeriods, nBuckets), cp.int32, workspace.cursor_offset)
        module.get_function("scanCounts")((nPeriods,), (SCAN_THREADS,), (countsGPU, cursorGPU, np.int32(nBuckets)))
        sortIndexGPU = workspace.view('scratch', (nPeriods, tSize), cp.int32)
        module.get_function("bucketScatterPackedPhase")(
            grid, (256,), (phasesGPU, self.tGPU, inversePeriodsGPU, cursorGPU, sortIndexGPU,
                          np.int32(tSize), np.int32(nBuckets)))
        module.get_function("sortBucketsPackedPhase")(
            ((nBuckets+255)//256, nPeriods), (256,),
            (phasesGPU, self.tGPU, inversePeriodsGPU, countsGPU, cursorGPU, sortIndexGPU,
             np.int32(tSize), np.int32(nBuckets)))
        del phasesGPU, countsGPU, cursorGPU

        # Folded data, extended by maxDuration points to handle transits at the phase edge
        packed = self.uniformWeight is not None
        weightedGPU = workspace.view('weighted', (nPeriods, patchedSize, 1 if packed else 2), cp.float32)
        cumResidualGPU = workspace.view('cumulative', (nPeriods, patchedSize), cp.int64)
        patchThreads = min(1024, max(32, 1 << ((patchedSize+1)//2-1).bit_length()))
        module.get_function("patchScanPacked" if packed else "patchScan")(
            (nPeriods,), (patchThreads,),
            (sortIndexGPU, self.preparedGPU, weightedGPU, cumResidualGPU,
             self.fixedScale, np.int32(tSize), np.int32(patchedSize)))
        del sortIndexGPU

        # Adjacent windows share input loads for the longer windows. Group within each
        # duration, so an incomplete group never consumes starts from the next duration.
        bestGPU = cp.full(nPeriods, EMPTY_KEY, dtype=cp.uint64)
        lanes = lanes_per_window(strides)
        screening = self.screening and margin > 0
        if not screening and not packed and np.any(lanes >= 8):
            adjacent = np.where(lanes == 32, 4, np.where(lanes >= 8, 2, 1))
            group_counts = (np.diff(offsets) + adjacent - 1) // adjacent
            offsets = np.concatenate(([0], np.cumsum(group_counts)))
        selectedGPU = cp.asarray(selected)
        offsetsGPU = cp.asarray(offsets.astype(np.int32))
        stridesGPU = cp.asarray(strides)
        if screening:
            # Only geometry is reused, within this search. The latest map suffices
            # for consecutive batches; weights and bounds are always input-dependent.
            mapKey = (tuple(selected), tuple(strides))
            lanesGPU = cp.asarray(lanes)
            if mapKey != self.screenMapKey:
                self.screenMapGPU = cp.asarray(screen_block_map(offsets, strides))
                if self.screenBudgetsGPU is None:
                    self.screenBudgetsGPU = cp.empty((3*len(self.durations), 4), dtype=cp.float64)
                module.get_function("prepareScreenBudgetsUniformSparse" if packed else "prepareScreenBudgetsSparse")(
                    ((len(selected)+63)//64,), (64,),
                    (self.widthsGPU, selectedGPU, lanesGPU, self.screenCertificates[0][2],
                     self.screenBudgetsGPU, np.int32(len(selected)), np.int32(len(self.durations))))
                if self.uniformWeight is not None:
                    module.get_function("prepareUniformSquare")(
                        (len(selected),), (32,),
                        (self.signalGPU, self.widthsGPU, selectedGPU, lanesGPU,
                         np.int32(self.maxWidth), self.uniformWeight, self.uniformSquareGPU))
                self.screenMapKey = mapKey
            prefixGPU = workspace.view('scratch', (nPeriods, self.screenPrefixSize, self.screenPrefixBytes), cp.uint8)
            prefixKernel = "scanScreeningUniformSparse" if packed else "scanScreeningSparse"
            prefix_args = (weightedGPU, prefixGPU, np.int32(patchedSize))
            if self.screenQuantization is not None:
                prefixKernel = "scanScreeningUniformInteger" if packed else "scanScreeningInteger"
                prefix_args += (*self.screenQuantization[0], *self.screenQuantization[1])
            module.get_function(prefixKernel)(
                (nPeriods,), (SCREEN_SCAN_THREADS,), prefix_args)
            nBounds = int(offsets[-1])
            lowerGPU = workspace.view('lower', (nPeriods, nBounds), cp.float32)
            upperGPU = cp.full(nPeriods, EMPTY_KEY, dtype=cp.uint64)
            if len(selected)*tSize < 2**32:
                # Reuse the upper-bound buffer for the chosen original-grid
                # location, then replace it with that candidate's exact delta.
                module.get_function("seedScreenCandidates")(
                    (nPeriods,), (256,),
                    (upperGPU, cumResidualGPU, self.fixedInvScale, self.widthsGPU,
                     selectedGPU, offsetsGPU, stridesGPU, periodMaskGPU,
                     np.int32(len(selected)), np.int32(len(self.durations)),
                     np.int32(tSize), np.int32(patchedSize), self.transit_depth_min))
                module.get_function("evaluateScreenSeedPacked" if packed else "evaluateScreenSeed")(
                    (nPeriods,), (32,),
                    (upperGPU, cumResidualGPU, self.fixedInvScale, weightedGPU,
                     self.signalGPU, self.overshootGPU, self.widthsGPU, selectedGPU,
                     lanesGPU, np.int32(tSize), np.int32(patchedSize), np.int32(self.maxWidth),
                     self.uniformWeight if packed else np.float32(0)))
            screen_before = (lowerGPU, upperGPU, prefixGPU, cumResidualGPU, self.fixedInvScale,
                             self.overshootGPU, self.widthsGPU, selectedGPU, offsetsGPU, stridesGPU,
                             lanesGPU, periodMaskGPU)
            screen_after = (np.int32(len(selected)), np.int32(len(self.durations)), np.int32(tSize),
                            np.int32(patchedSize), self.transit_depth_min)
            # Tight certificates only visit survivors of the preceding pass. All
            # passes reuse the same prefix and bound buffers; final FP32 evaluation
            # and tie ordering are unchanged.
            for level, certificate in enumerate(self.screenCertificates):
                kernel = "refineCandidates" if level else "screenCandidates"
                kernel += "UniformSparse" if self.uniformWeight is not None else "MappedSparse"
                uniform_args = ((self.uniformSquareGPU, self.uniformWeight)
                                if self.uniformWeight is not None else ())
                module.get_function(kernel)(
                    (len(self.screenMapGPU), nPeriods), (SCREEN_THREADS,),
                    screen_before + certificate + screen_after
                    + (self.screenMapGPU, self.screenBudgetsGPU) + uniform_args)
            del prefixGPU
        yBegin = 0
        while yBegin < len(selected):
            G = int(lanes[yBegin])
            yEnd = yBegin + 1
            while yEnd < len(selected) and lanes[yEnd] == G:
                yEnd += 1
            nGroups = int(offsets[yEnd] - offsets[yBegin])
            kernel = (f"searchChi2ScreenedG{G}" if screening else
                      f"searchChi2AdjacentG{G}" if G >= 8 else f"searchChi2G{G}")
            if packed:
                kernel = f"searchChi2PackedScreenedG{G}" if screening else f"searchChi2PackedG{G}"
            work_items = nGroups
            compact_args = ()
            # Prefix data are dead after screening. Reuse their arena for the
            # surviving indices and row counts, retaining the original path if
            # this layout does not fit or the candidate set is dense.
            if screening and 4*nPeriods*(nGroups+1) <= workspace.regions['scratch'][1]:
                candidatesGPU = workspace.view('scratch', (nPeriods, nGroups), cp.int32)
                countsGPU = workspace.view('scratch', (nPeriods,), cp.int32, 4*nPeriods*nGroups)
                countsGPU.fill(0)
                module.get_function("compactCandidates")(
                    ((nGroups + 255)//256, nPeriods), (256,),
                    (lowerGPU, upperGPU, candidatesGPU, countsGPU, np.int32(offsets[yBegin]),
                     np.int32(offsets[yEnd]), np.int32(nBounds), np.int32(nGroups)))
                survivors = int(cp.max(countsGPU))
                if survivors*4 <= nGroups:
                    kernel = f"searchChi2{'Packed' if packed else ''}CompactedG{G}"
                    work_items = survivors
                    compact_args = (candidatesGPU, countsGPU, np.int32(nGroups))
            if work_items:
                module.get_function(kernel)(
                    ((work_items * G + 255) // 256, nPeriods), (256,),
                    (bestGPU, cumResidualGPU, self.fixedInvScale, weightedGPU, self.signalGPU, self.overshootGPU,
                     self.widthsGPU, selectedGPU, offsetsGPU, stridesGPU, periodMaskGPU,
                     np.int32(yBegin), np.int32(yEnd), np.int32(len(self.durations)),
                     np.int32(tSize), np.int32(patchedSize), np.int32(self.maxWidth), self.transit_depth_min)
                    + ((lowerGPU, upperGPU, np.int32(nBounds)) if screening else ()) + compact_args
                    + ((self.uniformWeight,) if packed else ()))
            yBegin = yEnd
        return bestGPU

    def chi2_batch(self, periods, size, refine):
        """Lowest chi2 of each period of one batch, using its actual length.

        Each period is searched over its own plausible durations only, so results depend
        neither on the batching nor on the device evaluating each batch. A period without
        plausible duration gets the chi2 of the light curve without transit.
        size is the planned batch capacity; the final batch need not fill it.
        """
        valid_range = len(periods)

        periodMaskGPU = self.duration_mask(periods)
        if not bool(periodMaskGPU.any()):
            return np.full(valid_range, self.constant_residual, dtype=np.float32)

        margin = self.T0_refine_margin if refine else self.T0_fit_margin
        chi2, _ = decode_keys(self.best_keys(periods, periodMaskGPU, margin)[:valid_range], self.base)
        chi2 = cp.where(cp.isfinite(chi2), chi2, self.constant_residual)
        return chi2.astype(cp.float32).get()


class MultiGPUSearch:
    """One GPUSearch per device; batches of periods are distributed among the devices."""

    def __init__(self, device_ids, T0_fit_margin, T0_refine_margin, *args, **star_limits):
        self.device_ids = list(device_ids)
        self.searches = []
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                self.searches.append(GPUSearch(
                    load_module(), *args, T0_fit_margin=T0_fit_margin, T0_refine_margin=T0_refine_margin,
                    **star_limits))

    @property
    def primary(self):
        return self.searches[0]

    def bytes_per_period(self):
        """Persistent workspace plus small per-period masks, scalars and decoded results.

        One allocation owns the workspace across all batches. Its stage aliases
        are explicit, so pool splitting cannot change its physical footprint.
        Layout padding is budgeted separately in batch_size.
        """
        s = self.primary
        return (workspace_bytes_per_period(s.tSize, s.patchedDatasSize, s.maxScreenStarts, s.screening,
                                           prefix_bytes=s.screenPrefixBytes,
                                           weighted_bytes=s.weightedPointBytes,
                                           prefix_points=s.screenPrefixSize)
                + len(s.durations) + 128)

    @staticmethod
    def available_memory(device_id):
        """Free device memory, including blocks cached but unused by the CuPy memory pool."""
        with cp.cuda.Device(device_id):
            pool = cp.get_default_memory_pool()
            free = cp.cuda.Device(device_id).mem_info[0] + pool.total_bytes() - pool.used_bytes()
            limit = pool.get_limit()
            return min(free, max(0, limit-pool.used_bytes())) if limit else free

    def batch_size(self, n_periods):
        """Periods per batch: at least BATCHES_MIN batches (progress bar and balance between
        devices), within the memory of the fullest device. Results do not depend on it."""
        free = min(self.available_memory(device_id) for device_id in self.device_ids)
        by_memory = int((MEMORY_FRACTION * free - LAYOUT_ALLOWANCE) // self.bytes_per_period())
        if free < self.bytes_per_period() + LAYOUT_ALLOWANCE:
            raise MemoryError(
                f"One period needs {self.bytes_per_period() / 2**20:.0f} MiB of GPU memory, "
                f"{free / 2**20:.0f} MiB are free")
        return max(min(by_memory, int(n_periods / BATCHES_MIN), MAX_BATCH), 1)

    def chi2(self, periods, singleCalcPeriods, refine, pbar=None):
        """Lowest chi2 of each period, evaluated in batches of singleCalcPeriods periods."""
        periods = np.asarray(periods)
        # Bound the planned capacity for short lists such as the refinement passes.
        singleCalcPeriods = max(min(singleCalcPeriods, len(periods)), 1)
        chi2 = np.empty(len(periods), dtype=np.float32)
        batches = iter(range(0, len(periods), singleCalcPeriods))
        lock = threading.Lock()

        def worker(device_id, search):
            with cp.cuda.Device(device_id):
                capacity = singleCalcPeriods
                while True:
                    with lock:
                        start_idx = next(batches, None)
                    if start_idx is None:
                        return
                    end_idx = min(start_idx + singleCalcPeriods, len(periods))
                    cursor = start_idx
                    while cursor < end_idx:
                        stop = min(cursor+capacity, end_idx)
                        failed = False
                        try:
                            chi2[cursor:stop] = search.chi2_batch(periods[cursor:stop], capacity, refine)
                        except cp.cuda.memory.OutOfMemoryError:
                            if stop-cursor == 1:
                                raise
                            failed = True
                        if failed:
                            # Leave the except block before releasing storage: the
                            # exception traceback otherwise keeps failed arrays alive.
                            search.workspace = None
                            cp.get_default_memory_pool().free_all_blocks()
                            capacity = max(1, (stop-cursor)//2)
                            continue
                        cursor = stop
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
        while self.searches:
            self.searches.pop().close()
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
    T0_refine_margin,
    oversampling_factor,
    verbose,
    GPUDeviceIDs=(0,),
    # fast: just return periods and power, not the full result
    fast=False,
    bar_location=0,
    R_star_min=constants.R_STAR_MIN,
    R_star_max=constants.R_STAR_MAX,
    M_star_min=constants.M_STAR_MIN,
    M_star_max=constants.M_STAR_MAX,
):
    set_cuda_device(GPUDeviceIDs[0])
    search = MultiGPUSearch(GPUDeviceIDs, T0_fit_margin, T0_refine_margin, t, y, dy, transit_depth_min,
                            lc_arr, lc_cache_overview, R_star_min=R_star_min, R_star_max=R_star_max,
                            M_star_min=M_star_min, M_star_max=M_star_max)
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

    # Refine the 100 strongest peaks, plus the 100 strongest remaining peaks with period > 1 day,
    # with the finer T0 step T0_refine_margin. Ties keep the period order; masked power sorts last.
    order = np.argsort(-ma.filled(power, -np.inf), kind="stable")
    remaining = order[100:]
    remaining = remaining[ma.filled(periods, 0)[remaining] > 1]
    possiblePeriodsIndices = order[:100].tolist() + remaining[:100].tolist()
    possiblePeriods = list(ma.getdata(periods)[possiblePeriodsIndices])

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

    periodMaskGPU = search.duration_mask(np.array([period]))
    durationsBool = periodMaskGPU[0].get()
    durations = search.durations[durationsBool]
    single_lc_cache_overview = search.lc_cache_overview[durationsBool]

    chi2, location = decode_keys(search.best_keys(np.array([period]), periodMaskGPU, margin=0), search.base)
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

    whiteNoise = np.std(dataOutTransit)
    snrFit = transitDepth * (durationPointsNum ** 0.5) / whiteNoise
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durationPointsNum:] - DataCumsum[:-durationPointsNum]) / durationPointsNum
    redNoise = np.std(DataSlideAvg)

    Tx = bestTime[bestRowT0]
    minTime = min(t)
    T0 = Tx - int((Tx - minTime) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = transitDepth / (
        (whiteNoise ** 2 / (durationPointsNum)) + (redNoise ** 2 / (len(transit_times)))) ** 0.5

    transit_duration_in_days = calcDurationDays(t, period, T0, rawDuration)

    T0 = T0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if T0 < minTime:
        T0 = T0 + period

    # SNR
    all_flux_intransit_odd, all_flux_intransit_even, per_transit_count = intransit_stats(
        t, y, transit_times, transit_duration_in_days)
    snr_pink_per_transit = snr_stats(
        t=t,
        y=y,
        period=period,
        duration=transit_duration_in_days,
        T0=T0,
        transit_times=transit_times,
        transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count,
        window_std=search.window_standard_deviations,
    )

    all_flux_intransit = np.concatenate([all_flux_intransit_odd, all_flux_intransit_even])
    # The mask uses the same time units as t; rawDuration is a fraction of P.
    intransit = transit_mask(t, period, 2 * transit_duration_in_days, T0)
    flux_ootr = y[~intransit]
    depth_mean = np.mean(all_flux_intransit)
    snr = ((1 - depth_mean) / np.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)

    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times) ** (0.5))

    return rawDuration, durationPointsNum, transit_duration_in_days, transitDepth, T0, transit_times, \
        snr, snr_pink, snrFit, snrFitPink
