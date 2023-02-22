import numpy
import numpy as np
# from transitleastsquares.grid import T14
# from transitleastsquares.helpers import running_mean
# import transitleastsquares.tls_constants as tls_constants
import time
import cupy as cp
from tqdm import tqdm

import gtls.cupyFun as cupyFun
from gtls.stats import spectra,all_transit_times,calculate_transit_duration_in_days

def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    # MAX_BLOCK_SIZE = 256
    # MAX_BLOCK_SIZE = 1024
    blockSize = size
    if blockSize > MAX_BLOCK_SIZE:
        blockSize = MAX_BLOCK_SIZE
    gridSizeX = int((size / blockSize) + 1)
    return blockSize,gridSizeX

def foldfastCPU(time, period):
    """Fast phase folding with T0=0 hardcoded"""
    return time / period - numpy.floor(time / period)

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
    show_progress_bar,
    oversampling_factor
):
    #PreProcess
    singleCalcPeriods = 130
    start = time.time()
    print('Running PreProcess')
    try:
        with open ('cupyFun0.cu', 'r') as myfile:
            cupyCode=myfile.read()
    except IOError:
        cupyCode = cupyFun.getCuPyFun()
 
    module = cp.RawModule(code=cupyCode)

    periods = np.sort(periods)
    periods_arr = np.array(periods).astype(numpy.float32)
    periodsGPU = cp.asarray(periods_arr)

    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxWidthInSamples = int(max(durations))
    if maxWidthInSamples % 2 != 0:
        maxWidthInSamples = maxWidthInSamples + 1
    
    durations = numpy.sort(durations)

    # Phase fold
    mempool = cp.get_default_memory_pool()
    phasesGPU = cp.empty((len(periods),len(t)),dtype=cp.float32)
    sortIndexGPU = cp.empty((len(periods),len(t)),dtype=cp.int32)
    tGPU = cp.asarray(t).astype(cp.float32)
    periodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([len(t)])).astype(cp.int32)

    fastFoldGPU = module.get_function('foldFast')
    blockSize,gridSizeX = calcGridBlockSize(len(t))
    fastFoldGPU((gridSizeX,len(periods),),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))

    #To control the GPU memory usage, sort the data in serveral parts
    i_max = 10
    for i in range(1,i_max + 1):
        sortIndexGPU[(i-1)*len(periods)/i_max:i*len(periods)/i_max] = phasesGPU[(i-1)*len(periods)/i_max:i*len(periods)/i_max].argsort()
    
    del phasesGPU

    patchedDatasGPU = cp.empty((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    patchedDysGPU = cp.empty((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    patchedDatasSizeGPU = cp.asarray(np.array([len(t) + maxWidthInSamples])).astype(cp.int32)
    maxWidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
    patchedDatasSize = int(len(t) + maxWidthInSamples)
    tSizeGPU = cp.asarray(np.array([len(t)])).astype(cp.int32)
    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)
    
    fastFoldGPU = module.get_function('patchData')
    blockSize,gridSizeX = calcGridBlockSize(len(t) + maxWidthInSamples)
    fastFoldGPU((gridSizeX,len(periods),),(blockSize,),
    (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
    maxWidthInSamplesGPU,yGPU,dyGPU,tSizeGPU,periodsSizeGPU))

    del yGPU, dyGPU, sortIndexGPU

    #Other GPU variables, decalre here to save time.
    inverseSquaredPatchedDysGPU = cp.empty((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    edgeEffectCorrectionsGPU = cp.empty((len(periods)),dtype=cp.float32)
    maxwidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
    periodSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    singleCalcPeriodsGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
    meanXSizeGPU = cp.asarray(np.array([int(patchedDatasSize) - (np.min(durations)) + 1])).astype(cp.int32)
    durationsGPU = cp.asarray(durations).astype(cp.int32)
    durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    iterFlagGPU = cp.asarray(np.array([0])).astype(cp.int32)
    fullSumGPU = cp.empty((singleCalcPeriods,len(durations)),dtype=cp.float32)
    ootrGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    lowestResidualsGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    depthsGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    overshootGPU = cp.array(lc_cache_overview["overshoot"]).astype(cp.float32)
    datapointsGPU = cp.array([len(y)]).astype(cp.int32)
    meanSizeGPU = cp.array([(patchedDatasSize - x + 1) for x in durations]).astype(np.int32)
    LowestResidualsEachPeriodGPU = cp.empty((len(periods)),dtype=cp.float32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)

    # # ##calculate durations
    # # Not apply the duration filter for now
    # duration_max = T14(R_s=R_star_max, M_s=M_star_max, P=periods[-1], small=False)
    # duration_min = T14(R_s=R_star_min, M_s=M_star_min, P=periods[-1], small=True)
    # ## Fractional transit duration can be longer than this.
    # ## Example: Data length 11 days, 2 transits at 0.5 days and 10.5 days
    # length = max(t) - min(t)
    # no_of_transits_naive = length / periods[-1]
    # no_of_transits_worst = no_of_transits_naive + 1
    # correction_factor = no_of_transits_worst / no_of_transits_naive
    # duration_min_in_samples = int(numpy.floor(duration_min * len(y)))
    # duration_max_in_samples = int(numpy.ceil(duration_max * len(y) * correction_factor))
    # durations = durations[durations >= duration_min_in_samples]
    # durations = durations[durations <= duration_max_in_samples]

    remain_index = []
    for duration in durations:
        remain_index.append(np.where(lc_cache_overview["width_in_samples"] == duration)[0][0])
    raw_lc_cache_overview = lc_cache_overview
    raw_lc_arr = lc_arr
    lc_arr = lc_arr[remain_index]
    lc_cache_overview = lc_cache_overview[remain_index]
    overshoot = np.array(lc_cache_overview["overshoot"]).astype(np.float32)

    lc_arr_len = np.array([len(x) for x in lc_arr]).astype(np.int32)
    lc_arr_max_len = np.array([np.max(lc_arr_len)]).astype(np.int32)
    lc_arr_full_length = np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr])
    lcArrLenGPU = cp.asarray(lc_arr_len).astype(cp.int32)
    lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
    lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)

    calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
    blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
    calcInverseSquaredPatchedDyGPU((gridSizeX,len(periods),1),(blockSize,1,1),
    (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

    calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
    (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,maxwidthInSamplesGPU,periodSizeGPU,))

    #From now on, due to memory limitation, GPU can only do several periods(about 100-1000) at a time.
    # start_calc = time.time()
    min_locations = []
    min_residuals = []
    TotalIterNum = int(np.ceil(len(periods) / singleCalcPeriods))
    
    if(show_progress_bar):
        bar_format = "{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} periods | {elapsed}<{remaining}"
        pbar = tqdm(total=TotalIterNum, smoothing=0.3, bar_format=bar_format)

    for iterFlag in range(TotalIterNum):
    # for iterFlag in range(1):
   
        for i in range(singleCalcPeriods):
            if((iterFlag * singleCalcPeriods + i) < len(periods)):
                cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i + iterFlag * singleCalcPeriods])

        calcAllFullSumGPU = module.get_function('calcAllFullSum')
        blockSize,gridSizeX = calcGridBlockSize(len(durations))
        calcAllFullSumGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (fullSumGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,durationsGPU,durationsSizeGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))

        calcAllOutOfTransitResiduals_step1_2GPU = module.get_function('calcAllOutOfTransitResiduals_step1_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        calcAllOutOfTransitResiduals_step1_2GPU((gridSizeX,len(durations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,patchedDatasGPU,durationsGPU,durationsSizeGPU,
        inverseSquaredPatchedDysGPU,patchedDatasSizeGPU,meanXSizeGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))

        ootrGPU = np.cumsum(ootrGPU,axis=-1)
        calcAllOutOfTransitResiduals_step2_2GPU = module.get_function('calcAllOutOfTransitResiduals_step2_2GPU')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        calcAllOutOfTransitResiduals_step2_2GPU((gridSizeX,len(durations),singleCalcPeriods),
        (blockSize,1,1),(ootrGPU,patchedDatasGPU,
        inverseSquaredPatchedDysGPU,durationsGPU,durationsSizeGPU,patchedDatasSizeGPU,
        meanSizeGPU,meanXSizeGPU,fullSumGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPU')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        
        calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,depthsGPU,meanSizeGPU,
        meanXSizeGPU,patchedDatasGPU,patchedDatasSizeGPU,durationsGPU,
        durationsSizeGPU,lcArrFullLengthGPU,lcArrLenGPU,lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,#meanGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))

        # locationGPU = lowestResidualsGPU.argmin()
        # minValue = lowestResidualsGPU.min()
        # tempResultIndexsGPU[iterFlag] = locationGPU.item()
        # tempResultResidualsGPU[iterFlag] = minValue
        # sub_location = cp.unravel_index(locationGPU, lowestResidualsGPU.shape)
        # tempResultDepthsGPU[iterFlag] = depthsGPU[sub_location[0]][sub_location[1]][sub_location[2]]

        for i in range(singleCalcPeriods):
            if(iterFlag*singleCalcPeriods + i < len(periods)):
                locationGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].argmin()

        if((iterFlag+1)*singleCalcPeriods < len(periods)):
            LowestResidualsEachPeriodGPU[(iterFlag)*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods] = lowestResidualsGPU.min(axis=(1,2))
        else:
            LowestResidualsEachPeriodGPU[(iterFlag)*singleCalcPeriods:] = lowestResidualsGPU[:len(periods) - (iterFlag)*singleCalcPeriods].min(axis=(1,2))

        iterFlagGPU = iterFlagGPU + 1
        if show_progress_bar:
            pbar.update(1)

    if show_progress_bar:
        pbar.close()
    # minRes = tempResultResidualsGPU.min()
    # print('minRes',minRes)
    
    #TODO: FIX if T0 at the edge of the data
    chi2 = LowestResidualsEachPeriodGPU.get()
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    index_highest_power = numpy.argmax(power)
    period = periods[index_highest_power]
    bestLocation = locationGPU[index_highest_power].item()
    bestRow = np.floor(bestLocation / (int(patchedDatasSize) - (np.min(durations)) + 1)).astype(int)
    bestRowT0 = bestLocation % (int(patchedDatasSize) - (np.min(durations)) + 1)
    phases = np.sort(foldfastCPU(t, period))
    bestT0Phase = phases[bestRowT0]
    bestT0 = np.min(t) + bestT0Phase * period

    rawDuration = lc_cache_overview["duration"][bestRow]
    transit_times = all_transit_times(bestT0, t, period)
    transit_duration_in_days = calculate_transit_duration_in_days(
        t, period, transit_times, rawDuration
    )
    bestPeriod = period
    bestDuration = transit_duration_in_days
    bestDepth = None
    return bestPeriod,bestDuration,bestDepth,bestT0,SDE