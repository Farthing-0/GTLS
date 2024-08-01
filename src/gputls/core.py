import numpy as np
import numpy.ma as ma
import cupy as cp
from .stats import spectra,all_transit_times,calculate_transit_duration_in_days,intransit_stats,snr_stats,calcDurationDays
from .helpers import transit_mask
from .transit import mutipleTransitFit
from . import GPUFun
import pynvml
import tqdm

def foldCPU(time,flux,dy,period):
    """Fold time series data."""
    phase = (time % period) / period
    rank = np.argsort(phase)
    return np.array(time[rank]),flux[rank],dy[rank]

def set_cuda_device(device_id):
    """Set the CUDA device."""
    cp.cuda.Device(device_id).use()

def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    blockSize = size
    if blockSize > MAX_BLOCK_SIZE:
        blockSize = MAX_BLOCK_SIZE
    gridSizeX = int((size / blockSize) + 1)
    return blockSize,gridSizeX

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

def search_multi_periods(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    R_star_min,
    R_star_max,
    M_star_min,
    M_star_max,
    lc_arr,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose,
    useLocalPTXCUBIN = False,
    GPUDeviceID = 0,
    #fast: just return periods and power, not the full result
    fast = False,
    #legacy: Skip-points search, like the original TLS,not implemented yet.
    legacy = False,
    # SimplifyEdgeEffect if dy is nearly uniform, we can simplify the edge effect correction
    SimplifyEdgeEffect = True,
    bar_location = 0
):
    
    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
    module = cp.RawModule(code=GPUCode)
    module.compile()

    durations,indices = np.unique(lc_cache_overview["width_in_samples"],return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))

    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods) / 30]))

    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)

    #Due to GPU memory size limitation, GPU can only do several periods at a time.
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    if verbose:
        pbar = tqdm.tqdm(total=TotalIter,position=bar_location)

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    iterFlagGPU = cp.int32(0)

    fulldurationsMaxGPU = cp.empty((len(periods),),dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods),),dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods).astype(cp.float64)
    
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,fulldurationsMaxGPU, fulldurationsMinGPU,tLengthGPU,tSizeGPU, fullperiodsSizeGPU))

    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods),len(durations)),dtype=cp.bool_)

    durationBoolFunGPU = module.get_function('durationBool')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationBoolFunGPU((gridSizeX,len(durations),1),(blockSize,1,1),
                    (fulldurationsMaxGPU,fulldurationsMinGPU,fulldurationsSizeGPU,fullperiodsSizeGPU,fulldurationsGPU,durationBoolArrayGPU))

    durationsGridCollectionGPU = cp.empty((TotalIter,len(durations)),dtype=cp.bool_)

    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    for iterFlag in range(TotalIter):

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            # enlarge the SinglePeriods array to the same size as singleCalcPeriods
            SinglePeriods = np.append(SinglePeriods,np.zeros((singleCalcPeriods - len(SinglePeriods),)))
            durationsGridCollectionGPU[iterFlag] = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[-1])
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            # durationsGridGPU = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[(iterFlag+1)*singleCalcPeriods])
            durationsGridCollectionGPU[iterFlag] = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[(iterFlag+1)*singleCalcPeriods])

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        single_lc_arr = lc_arr[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        lowestResidualsGPU = cp.empty((singleCalcPeriods,len(singleDurations),tSize),dtype=cp.float32)
        # lowestResidualsGPU = cp.empty((singleCalcPeriods,len(singleDurations)*tSize),dtype=cp.float32)

        # Phase fold
        phasesGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.float64)
        sortIndexGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.int32)
        tGPU = cp.asarray(t).astype(cp.float64)
        periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
        tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1),(blockSize,),
                        (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

        patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)

        lc_arr_max_len = np.array([np.max(singleDurations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])

        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
        
        edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)

        inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
        periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        durationsGPU = cp.asarray(singleDurations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(singleDurations)])).astype(cp.int32)
        datapointsGPU = cp.array([len(y)]).astype(cp.int32)
        transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

        #GPU variables for the loop
        fullSumGPU = cp.empty((singleCalcPeriods,len(singleDurations)),dtype=cp.float32)
        cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
        ootrGPU = cp.empty((singleCalcPeriods,len(singleDurations),(tSize)),dtype=cp.float32)

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))

        # # incase gpu memory is not enough, split the phasesGPU into several parts
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()
        # todo: change to below way, seems a bug here
        # sortIndexGPU = phasesGPU.argsort()

        #calculate patched data
        patchDataGPU = module.get_function('patchData')
        blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
        (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
        maxDurationGPU,yGPU,dyGPU,tSizeGPU))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
        (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,maxDurationGPU,periodSizeGPU,))
        
        for i in range(singleCalcPeriods):
            # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        calcAllFullSumGPU = module.get_function('calcAllFullSum')
        blockSize,gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
        periodSizeGPU,))

        calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
        inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,tSizeGPU,))

        ootrGPU = cp.cumsum(ootrGPU,axis=-1)
        calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,
        durationsSizeGPU,patchedDatasSizeGPU,
        durationsGPU,tSizeGPU,fullSumGPU,))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUB')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        # calcAllLowestResidualsGPU((gridSizeX,singleCalcPeriods,1),
        (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,
        durationsGPU,durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
        transitDepthMinGPU
        ))

        start_idx = iterFlag * singleCalcPeriods
        end_idx = start_idx + singleCalcPeriods
        valid_range = min(end_idx, len(periods)) - start_idx
        valid_lowest_residuals = lowestResidualsGPU[:valid_range]
        flattened_residuals = valid_lowest_residuals.reshape(valid_range, -1)
        min_indices = cp.argmin(flattened_residuals, axis=-1)
        min_values = cp.min(flattened_residuals, axis=-1)
        locationGPU[start_idx:start_idx + valid_range] = min_indices
        LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = min_values

        iterFlagGPU = iterFlagGPU + 1

        if verbose:
            pbar.update(1)

    chi2 = LowestResidualsEachPeriodGPU.get()

    raw_chi2 = chi2.copy()
    median = np.median(raw_chi2)
    chi2_mask = raw_chi2 > (100 * median)
    chi2 = ma.array(raw_chi2, mask=chi2_mask)
    periods = ma.array(periods, mask=chi2_mask)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    raw_power = power.copy()

    if fast:
        return periods,power

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

    chi2_again = search_multi_periods_again(
        possiblePeriods,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID,
        singleCalcPeriods
    )

    chi2[possiblePeriodsIndices] = chi2_again
    # chi2_median = np.median(chi2)
    # # replace the extreme outliers
    # chi2 = np.where(np.abs(chi2 - chi2_median) > 50 * chi2_median, chi2_median, chi2)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]

    possiblePeriodsTimesRate = [0.5,1,2,2/3,3/2]
    possiblePeriodsTemp = [period * rate for rate in possiblePeriodsTimesRate]

    possiblePeriodsIndices, possiblePeriods = find_nearest_indices(possiblePeriodsTemp, periods)

    chi2_again = search_multi_periods_again(
        possiblePeriods,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID,
        singleCalcPeriods
    )

    chi2[possiblePeriodsIndices] = chi2_again
    # chi2_median = np.ma.median(chi2)
    # replace the extreme outliers
    # chi2 = np.where(np.abs(chi2 - chi2_median) > 50 * chi2_median, chi2_median, chi2)

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    power_again = power[possiblePeriodsIndices]
    periodIndex = possiblePeriodsIndices[np.argmax(power_again)]
    period = periods[periodIndex]

    rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,transit_times,snr,snr_pink,snrFit,snrFitPink = search_single_periods(
        period,
        t,
        y,
        dy,
        transit_depth_min,
        lc_arr,
        lc_cache_overview,
        GPUDeviceID
    )

    return periods,period,rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,\
            SDE,chi2,transit_times,power,snr,snr_pink,snrFit,snrFitPink,raw_power,raw_chi2

def search_multi_periods_again(
    periods,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID,
    singleCalcPeriods,
):
    
    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    module = cp.RawModule(code=GPUCode)

    durations,indices = np.unique(lc_cache_overview["width_in_samples"],return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    # nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))

    # # singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods)]))

    # # if singleCalcPeriods < 15:
    # #     singleCalcPeriods = int(singleCalcPeriods / 1.1)

    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    iterFlagGPU = cp.int32(0)

    fulldurationsMaxGPU = cp.empty((len(periods),),dtype=cp.int32)
    fulldurationsMinGPU = cp.empty((len(periods),),dtype=cp.int32)
    fullperiodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
    periodsGPU = cp.asarray(periods).astype(cp.float64)
    
    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,fulldurationsMaxGPU, fulldurationsMinGPU,tLengthGPU,tSizeGPU, fullperiodsSizeGPU))

    fulldurationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    fulldurationsGPU = cp.asarray(durations).astype(cp.int32)
    durationBoolArrayGPU = cp.empty((len(periods),len(durations)),dtype=cp.bool_)

    durationBoolFunGPU = module.get_function('durationBool')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationBoolFunGPU((gridSizeX,len(durations),1),(blockSize,1,1),
                    (fulldurationsMaxGPU,fulldurationsMinGPU,fulldurationsSizeGPU,fullperiodsSizeGPU,fulldurationsGPU,durationBoolArrayGPU))

    durationsGridCollectionGPU = cp.empty((TotalIter,len(durations)),dtype=cp.bool_)

    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    for iterFlag in range(TotalIter):

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            # enlarge the SinglePeriods array to the same size as singleCalcPeriods
            SinglePeriods = np.append(SinglePeriods,np.zeros((singleCalcPeriods - len(SinglePeriods),)))
            durationsGridCollectionGPU[iterFlag] = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[-1])
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            # durationsGridGPU = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[(iterFlag+1)*singleCalcPeriods])
            durationsGridCollectionGPU[iterFlag] = cp.logical_or(durationBoolArrayGPU[iterFlag*singleCalcPeriods],durationBoolArrayGPU[(iterFlag+1)*singleCalcPeriods])

        durationsBoolGrid = durationsGridCollectionGPU[iterFlag].get()
        singleDurations = durations[durationsBoolGrid]
        single_lc_arr = lc_arr[durationsBoolGrid]
        single_lc_cache_overview = lc_cache_overview[durationsBoolGrid]
        overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        lowestResidualsGPU = cp.empty((singleCalcPeriods,len(singleDurations),tSize),dtype=cp.float32)

        # Phase fold
        phasesGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.float64)
        sortIndexGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.int32)
        tGPU = cp.asarray(t).astype(cp.float64)
        periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
        tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1),(blockSize,),
                        (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

        patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)

        lc_arr_max_len = np.array([np.max(singleDurations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])

        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
        
        edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)

        inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
        periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        durationsGPU = cp.asarray(singleDurations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(singleDurations)])).astype(cp.int32)
        datapointsGPU = cp.array([len(y)]).astype(cp.int32)
        transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

        #GPU variables for the loop
        fullSumGPU = cp.empty((singleCalcPeriods,len(singleDurations)),dtype=cp.float32)
        cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
        ootrGPU = cp.empty((singleCalcPeriods,len(singleDurations),(tSize)),dtype=cp.float32)

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

        #calculate patched data
        patchDataGPU = module.get_function('patchData')
        blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
        patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
        (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
        maxDurationGPU,yGPU,dyGPU,tSizeGPU))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
        (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,maxDurationGPU,periodSizeGPU,))
        
        for i in range(singleCalcPeriods):
            # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
            cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

        calcAllFullSumGPU = module.get_function('calcAllFullSum')
        blockSize,gridSizeX = calcGridBlockSize(len(singleDurations))
        calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
        periodSizeGPU,))

        calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
        inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,tSizeGPU,))

        ootrGPU = cp.cumsum(ootrGPU,axis=-1)
        calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,
        durationsSizeGPU,patchedDatasSizeGPU,
        durationsGPU,tSizeGPU,fullSumGPU,))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUBNoSkipTemp')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        calcAllLowestResidualsGPU((gridSizeX,len(singleDurations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,
        durationsGPU,durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
        transitDepthMinGPU
        ))

        start_idx = iterFlag * singleCalcPeriods
        end_idx = start_idx + singleCalcPeriods
        valid_range = min(end_idx, len(periods)) - start_idx
        valid_lowest_residuals = lowestResidualsGPU[:valid_range]
        flattened_residuals = valid_lowest_residuals.reshape(valid_range, -1)
        min_indices = cp.argmin(flattened_residuals, axis=-1)
        min_values = cp.min(flattened_residuals, axis=-1)
        locationGPU[start_idx:start_idx + valid_range] = min_indices
        LowestResidualsEachPeriodGPU[start_idx:start_idx + valid_range] = min_values

        iterFlagGPU = iterFlagGPU + 1

    chi2 = LowestResidualsEachPeriodGPU.get()

    return chi2

# This function is used for refind Duration and T0 since we skip some points in "search_multi_periods"
def search_single_periods(
    period,
    t,
    y,
    dy,
    transit_depth_min,
    lc_arr,
    lc_cache_overview,
    GPUDeviceID = 0
):

    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    GPUCode = GPUFun.getGPUCode()
    module = cp.RawModule(code=GPUCode)

    durations = np.unique(lc_cache_overview["width_in_samples"])
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    singleCalcPeriods = 1

    #Initialize the variables
    periodsGPU = cp.asarray(np.array([period])).astype(cp.float64)

    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)

    SinglePeriods = np.array([period])
    durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
    durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

    # Phase fold
    phasesGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.float64)
    sortIndexGPU = cp.empty((singleCalcPeriods,tSize),dtype=cp.int32)
    tGPU = cp.asarray(t).astype(cp.float64)
    periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

    durationMax = durationsMaxGPU.get().item()
    durationMin = durationsMinGPU.get().item()
    durationsBoolList = np.logical_and(durations <= durationMax, durations >= durationMin)
    durations = durations[durationsBoolList]
    single_lc_arr = lc_arr[durationsBoolList]
    single_lc_cache_overview = lc_cache_overview[durationsBoolList]

    lowestResidualsGPU = cp.empty((len(durations),tSize),dtype=cp.float32)

    fastFoldGPU = module.get_function('foldFast')
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
    # if singleCalcPeriods > 100:
    i_max = 10
    for i in range(1,i_max + 1):
        sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

    patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)

    lc_arr_max_len = np.array([np.max(durations)]).astype(np.int32)
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in single_lc_arr])

    lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
    lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)

    edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
    
    inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    durationsGPU = cp.asarray(durations).astype(cp.int32)
    durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)

    overshootGPU = cp.array(single_lc_cache_overview["overshoot"]).astype(cp.float32)
    datapointsGPU = cp.array([len(y)]).astype(cp.int32)

    ## depth variable is not needed anymore because we trapezoid fit the transit to get the depth
    ## This method is much more faster and accurate than producing a depth from the transit model
    # depthsEachPeriodGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
    transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

    #GPU variables for the loop
    fullSumGPU = cp.empty((singleCalcPeriods,len(durations)),dtype=cp.float32)
    cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
    
    # resultArrayXAxisSizeGPU is the maximum size of the result of a function possible use to restore the result of a patched light curve
    # resultArrayXAxisSizeGPU = cp.asarray(np.array([int(patchedDatasSize) - (np.min(durations)) + 1])).astype(cp.int32)
    # resultArrayXAxisSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)

    # ootrGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    ootrGPU = cp.empty((singleCalcPeriods,len(durations),(tSize)),dtype=cp.float32)

    #calculate patched data
    patchDataGPU = module.get_function('patchData')
    blockSize,gridSizeX = calcGridBlockSize(tSize + maxDuration)
    patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
    (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
    maxDurationGPU,yGPU,dyGPU,tSizeGPU))

    calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
    blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
    calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
    (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

    calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
    blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
    calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
    (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,maxDurationGPU,periodSizeGPU,))
    
    for i in range(singleCalcPeriods):
        # if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
        cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])

    calcAllFullSumGPU = module.get_function('calcAllFullSum')
    blockSize,gridSizeX = calcGridBlockSize(len(durations))
    calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
    (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
    periodSizeGPU,))

    calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
    # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
    inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,tSizeGPU,))

    # ootrGPU = np.cumsum(ootrGPU,axis=-1)
    ootrGPU = cp.cumsum(ootrGPU,axis=-1)
    calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
    # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(ootrGPU,
    durationsSizeGPU,patchedDatasSizeGPU,
    durationsGPU,tSizeGPU,fullSumGPU,))

    calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUBNoSkip')
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
    patchedDatasGPU,patchedDatasSizeGPU,
    durationsGPU,durationsSizeGPU,
    lcArrFullLengthGPU,
    lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
    overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
    durationsMaxGPU,durationsMinGPU,transitDepthMinGPU
    ))

    bestLocation = lowestResidualsGPU.argmin().get()

    durationIndex = np.floor(bestLocation / (tSize)).astype(int)    
    durationPointsNum = durations[durationIndex]

    find = np.where(single_lc_cache_overview["width_in_samples"] == durationPointsNum)[0]
    if len(find) > 1:
        find = find[0]
    bestRow = find.item()
    rawDuration = single_lc_cache_overview['duration'][bestRow]

    bestTime,bestFlux,bestFluxDy = foldCPU(t,y,dy,period)
    bestFlux = np.concatenate((bestFlux,bestFlux[:maxDuration]))
    bestFluxDy = np.concatenate((bestFluxDy,bestFluxDy[:maxDuration]))

    bestRowT0 = bestLocation % (tSize)

    transitMean = bestFlux[bestRowT0:bestRowT0+durationPointsNum].mean()

    # Transit Depth
    overshoot = single_lc_cache_overview["overshoot"][durationIndex]
    transitDepth =  ((1-transitMean) * overshoot).item()

    dataOutTransit = np.concatenate((bestFlux[0:bestRowT0],bestFlux[bestRowT0+durationPointsNum:]))

    if bestRowT0 > tSize - 1:
        bestRowT0 = bestRowT0 - tSize 

    snrFit = (1 - transitDepth)*(durationPointsNum ** 0.5)/np.std(dataOutTransit)
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durationPointsNum:] - DataCumsum[:-durationPointsNum])/durationPointsNum
    redNoise = np.std(DataSlideAvg)

    Tx = bestTime[bestRowT0]
    T0 = Tx - int((Tx-min(t)) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = (1 - transitDepth)/((np.std(dataOutTransit)**2/(durationPointsNum)) + (redNoise**2/(len(transit_times))))**0.5

    # if legacy:
    #     #Raw TLS Calculate transit duration(days) Method
    #     transit_duration_in_days = calculate_transit_duration_in_days(
    #         t, period, transit_times, rawDuration
    #     )
    # else:
    ## Alternative TLS Calculate transit duration(days) Method
    transit_duration_in_days = calcDurationDays(t, period, T0, rawDuration)
    
    T0 = T0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if(T0 < min(t)):
        T0 = T0 + period
    else:
        T0 = T0

    #SNR
    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
    t, y, transit_times, transit_duration_in_days
    )
    snr_per_transit, snr_pink_per_transit = snr_stats(
        t=t,
        y=y,
        period=period,
        duration=rawDuration,
        T0=T0,
        transit_times=transit_times,
        transit_duration_in_days=transit_duration_in_days,
        per_transit_count=per_transit_count,
    )

    all_flux_intransit = np.concatenate(
        [all_flux_intransit_odd, all_flux_intransit_even]
    )
    intransit = transit_mask(t, period, 2 * rawDuration, T0)
    flux_ootr = y[~intransit]
    depth_mean = np.mean(all_flux_intransit)
    # depth_mean_std = np.std(all_flux_intransit) / np.sum(
    #     per_transit_count
    # ) ** (0.5)
    snr = ((1 - depth_mean) / np.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)

    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times)**(0.5))
    
    return rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,transit_times,snr,snr_pink,snrFit,snrFitPink