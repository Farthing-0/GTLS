import numpy
import numpy as np
import cupy as cp
from .stats import spectra,all_transit_times,calculate_transit_duration_in_days,intransit_stats,snr_stats
from .helpers import transit_mask
from . import GPUFun
import pynvml

def calcGridBlockSize(size):
    MAX_BLOCK_SIZE = 128
    blockSize = size
    if blockSize > MAX_BLOCK_SIZE:
        blockSize = MAX_BLOCK_SIZE
    gridSizeX = int((size / blockSize) + 1)
    return blockSize,gridSizeX

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
    lc_arr_grazing,
    lc_arr_box,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose
):
    # T0_fit_margin is not used for now, because T0_fit_margin is used to skip
    # some points in the search to reduce time in CPU TLS, but GPU is fast enough to search all points.
    # Use TESS data as an example, if we skip 99/100 points,(if the duration is longer than 100 points),
    # the search time will reduce about 0.5s to 0.005s, which is not significant.
    # Maybe we can provide a "Fast" mode in the future.

    # singleCalcPeriods = 130

    # with open ('GPUFun.cu', 'r') as myfile:
    #     myCode=myfile.read()
    GPUCode = GPUFun.getGPUCode()
    module = cp.RawModule(code=GPUCode)

    periodsGPU = cp.array(periods, dtype=cp.float64)
    durationsMaxGPU = cp.array(periods, dtype=cp.int32)
    durationsMinGPU = cp.array(periods, dtype=cp.int32)

    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxWidthInSamples = int(max(durations))
    if maxWidthInSamples % 2 != 0:
        maxWidthInSamples = maxWidthInSamples + 1
    durations = numpy.sort(durations)

    # Phase fold
    
    phasesGPU = cp.empty((len(periods),len(t)),dtype=cp.float64)
    sortIndexGPU = cp.empty((len(periods),len(t)),dtype=cp.int32)
    tGPU = cp.asarray(t).astype(cp.float64)
    periodsSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    tSizeGPU = cp.asarray(np.array([len(t)])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

    durationsGridGPU = module.get_function('durationsGrid')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    durationsGridGPU((gridSizeX,1,1),(blockSize,),
                    (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

    fastFoldGPU = module.get_function('foldFast')
    blockSize,gridSizeX = calcGridBlockSize(len(t))
    fastFoldGPU((gridSizeX,len(periods),),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
    i_max = 10
    for i in range(1,i_max + 1):
        sortIndexGPU[(i-1)*len(periods)/i_max:i*len(periods)/i_max] = phasesGPU[(i-1)*len(periods)/i_max:i*len(periods)/i_max].argsort()

    patchedDatasGPU = cp.zeros((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    patchedDysGPU = cp.empty((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    patchedDatasSizeGPU = cp.asarray(np.array([len(t) + maxWidthInSamples])).astype(cp.int32)
    maxWidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
    patchedDatasSize = int(len(t) + maxWidthInSamples)
    yGPU = cp.asarray(y).astype(cp.float32)
    dyGPU = cp.asarray(dy).astype(cp.float32)


    lc_arr_len = np.array([len(x) for x in lc_arr]).astype(np.int32)
    lc_arr_max_len = np.array([np.max(lc_arr_len)]).astype(np.int32)
    lc_arr_full_length = np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr])

    lc_arr_grazing_full_length = np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr_grazing])
    lc_arr_box_full_length = np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr_box])

    lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
    lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
    lcArrGrazingFullLengthGPU = cp.asarray(lc_arr_grazing_full_length).astype(cp.float32)
    lcArrBoxFullLengthGPU = cp.asarray(lc_arr_box_full_length).astype(cp.float32)

    #Other GPU variables, declare here to save time.
    inverseSquaredPatchedDysGPU = cp.empty((len(periods),len(t) + maxWidthInSamples),dtype=cp.float32)
    edgeEffectCorrectionsGPU = cp.empty((len(periods)),dtype=cp.float32)
    maxwidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
    periodSizeGPU = cp.asarray(np.array([len(periods)])).astype(cp.int32)
    durationsGPU = cp.asarray(durations).astype(cp.int32)
    durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
    iterFlagGPU = cp.asarray(np.array([0])).astype(cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)

    overshootGPU = cp.array(lc_cache_overview["overshoot"]).astype(cp.float32)
    datapointsGPU = cp.array([len(y)]).astype(cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty((len(periods)),dtype=cp.float32)
    depthsEachPeriodGPU = cp.empty((len(periods)),dtype=cp.float32)
    transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = (nvmlinfo.free) / (10*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*3 + 2*len(durations)))
    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods)]))

    #GPU variables for the loop
    singleCalcPeriodsGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    fullSumGPU = cp.empty((singleCalcPeriods,len(durations)),dtype=cp.float32)
    cumsumGPU = cp.empty((singleCalcPeriods,patchedDatasSize),dtype=cp.float32)
    meanXSizeGPU = cp.asarray(np.array([int(patchedDatasSize) - (np.min(durations)) + 1])).astype(cp.int32)
    ootrGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    lowestResidualsGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    # depthsGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
    # lowestResidualsTypeGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.int32)
    meanSizeGPU = cp.array([(patchedDatasSize - x + 1) for x in durations]).astype(np.int32)
    
    #calculate patched data
    patchDataGPU = module.get_function('patchData')
    blockSize,gridSizeX = calcGridBlockSize(len(t) + maxWidthInSamples)
    patchDataGPU((gridSizeX,len(periods),),(blockSize,),
    (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
    maxWidthInSamplesGPU,yGPU,dyGPU,tSizeGPU,periodsSizeGPU))

    # import matplotlib.pyplot as plt
    # plt.plot(patchedDatasGPU[1].get(),'.')
    # plt.savefig('patchedDatasGPU.png')
    # plt.close()
    # exit()

    calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
    blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
    calcInverseSquaredPatchedDyGPU((gridSizeX,len(periods),1),(blockSize,1,1),
    (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

    #calculate edge_effect_correction
    calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
    blockSize,gridSizeX = calcGridBlockSize(len(periods))
    calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
    (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
    patchedDatasSizeGPU,maxwidthInSamplesGPU,periodSizeGPU,))

    # print('gpu memory usage:',cp.get_default_memory_pool().used_bytes() / 1024 / 1024,'MB')

    #From now on, due to GPU memory size limitation, GPU can only do several periods(about 100-1000) at a time.
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))
    if verbose:
        print('TotalIter',TotalIter)
    for iterFlag in range(TotalIter):
        
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
        (blockSize,1,1),(ootrGPU,
        durationsSizeGPU,patchedDatasSizeGPU,
        meanSizeGPU,meanXSizeGPU,fullSumGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPU')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        
        #About LowestResidualsTypeGPU, 0 means standard transit, 1 means grazing transit, 2 means box transit,
        #3 means no transit(Or not detect at all)
        #TODO:export all transit types output, distinguish them in the next step? or just use the lowest one? 
        calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,#depthsGPU,
        meanSizeGPU,meanXSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,durationsGPU,
        durationsSizeGPU,lcArrFullLengthGPU,lcArrGrazingFullLengthGPU,lcArrBoxFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,#meanGPU,
        durationsMaxGPU,durationsMinGPU,transitDepthMinGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))
        
        #find best fit
        for i in range(singleCalcPeriods):
            if(iterFlag*singleCalcPeriods + i < len(periods)):
                locationGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].argmin()
                LowestResidualsEachPeriodGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].min()
                # depthsEachPeriodGPU[iterFlag*singleCalcPeriods + i] = depthsGPU[i][int(locationGPU[iterFlag*singleCalcPeriods + i] / lowestResidualsGPU.shape[2])][locationGPU[iterFlag*singleCalcPeriods + i] % lowestResidualsGPU.shape[2]]

        iterFlagGPU = iterFlagGPU + 1

        if verbose:
            if((iterFlag + 1) % 10 == 0):
                print((iterFlag + 1),'/',TotalIter,'bulk periods calculated')

    chi2 = LowestResidualsEachPeriodGPU.get()
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)

    #Self Defined metrics
    HighestPowerIndex = numpy.argmax(power)
    # Depth = depthsEachPeriodGPU[HighestPowerIndex].item()    
    period = periods[HighestPowerIndex]

    # BestChi2Index = numpy.argmin(chi2)
    bestLocation = locationGPU[HighestPowerIndex].item()
    durationIndex = np.floor(bestLocation / (int(patchedDatasSize) - (np.min(durations)) + 1)).astype(int)
    bestRow = np.where(lc_cache_overview["width_in_samples"] == durations[durationIndex])[0].item()
    rawDuration = lc_cache_overview['duration'][bestRow]

    bestRowT0 = bestLocation % (int(patchedDatasSize) - (np.min(durations)) + 1)
    # print('Period, Duration and T0 index:',HighestPowerIndex,durationIndex,bestRowT0)
    transitMean = patchedDatasGPU[HighestPowerIndex][bestRowT0:bestRowT0+durations[durationIndex]].mean()
    # print('transitMean:',type(transitMean.item()))
    # print('transitMean:',type(transitMean))
    # print('rawDuration:',durations[durationIndex])
    # print('transit:',patchedDatasGPU[HighestPowerIndex][bestRowT0:bestRowT0+durations[durationIndex]])

    #Technically, the "real" trapezoidFitSize = 2 * trapezoidFitSize
    trapezoidFitSize = 100
    trapezoidFitResultGPU = cp.empty((trapezoidFitSize,durations[durationIndex]),dtype=cp.float32)
    # print(type(durations[durationIndex]))
    # exit()
    trapezoidFitGPU = module.get_function('trapezoidFit')
    blockSize,gridSizeX = calcGridBlockSize(trapezoidFitSize)
    trapezoidFitGPU((gridSizeX,1,1),(blockSize,1,1),(trapezoidFitResultGPU,
    patchedDatasGPU[HighestPowerIndex],inverseSquaredPatchedDysGPU[HighestPowerIndex],
    cp.int32(durations[durationIndex]),cp.int32(bestRowT0),transitMean,cp.int32(trapezoidFitSize)))
    bestFitTid = cp.sum(trapezoidFitResultGPU,axis=-1).argmin()
    BestFitDepth = (trapezoidFitSize * (transitMean) - 0.5*bestFitTid)/(trapezoidFitSize - 0.5*bestFitTid)
    dataOutTransit = np.concatenate((patchedDatasGPU[HighestPowerIndex][0:bestRowT0].get(),patchedDatasGPU[HighestPowerIndex][bestRowT0+durations[durationIndex]:].get()))
    # snrFit = (1 - BestFitDepth)*(durations[durationIndex] ** 0.5)/cp.std(trapezoidFitResultGPU[bestFitTid])   
    snrFit = (1 - BestFitDepth)*(durations[durationIndex] ** 0.5)/np.std(dataOutTransit)
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durations[durationIndex]:] - DataCumsum[:-durations[durationIndex]])/durations[durationIndex]
    redNoise = np.std(DataSlideAvg)
    # print('redNoise:',redNoise)
    # print('redNoisestd',np.std(DataSlideAvg))
    # print('BestFitDepth:',BestFitDepth)
    # print('FirstDepth:',1-Depth)
    # print('fitSNR:',fitSNR)


    # print(trapezoidFitResultGPU)
    # print(cp.sum(trapezoidFitResultGPU,axis=-1))
    # import matplotlib.pyplot as plt
    # plt.plot(trapezoidFitResultGPU[23].get(),'.r')
    # plt.plot(patchedDatasGPU[HighestPowerIndex][bestRowT0:bestRowT0+durations[durationIndex]].get(),'.b')
    # plt.plot(cp.sum(trapezoidFitResultGPU,axis=-1).get(),'.r')
    # plt.plot(cp.std(trapezoidFitResultGPU,axis=-1).get(),'.b')
    # plt.savefig('transit.png')
    # plt.close()
    # print('transitMean:',transitMean)
    # exit()


    if bestRowT0 > len(t) - 1:
        bestRowT0 = bestRowT0 - len(t)
    bestSortIndex = sortIndexGPU[HighestPowerIndex]
    tIndex = bestSortIndex[bestRowT0]
    Tx = t[tIndex.get()]
    T0 = Tx - int((Tx-min(t)) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = (1 - BestFitDepth)/((np.std(dataOutTransit)**2/(durations[durationIndex])) + (redNoise**2/(len(transit_times))))**0.5

    transit_duration_in_days = calculate_transit_duration_in_days(
        t, period, transit_times, rawDuration
    )
    T0 = T0 + transit_duration_in_days / 2
    transit_times = transit_times + transit_duration_in_days / 2
    if(T0 < min(t)):
        T0 = T0 + period
    else:
        T0 = T0

    #SNR
    # print('GPU transit_times',transit_times)
    # print('GPU transit_duration_in_days',transit_duration_in_days)
    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
    t, y, transit_times, transit_duration_in_days
    )
    # all_flux_intransit = numpy.concatenate(
    #     [all_flux_intransit_odd, all_flux_intransit_even]
    # )
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
    # print('GPU snr_per_transit',snr_per_transit)
    # print('GPU snr_pink_per_transit',snr_pink_per_transit)
    # intransit = transit_mask(t, period, 2 * rawDuration, T0)
    # flux_ootr = y[~intransit]
    
    # # print('GPU all_flux_intransit',all_flux_intransit)
    # depth_mean = numpy.mean(all_flux_intransit)
    # print('GPU depth_mean',depth_mean)
    # depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
    #     per_transit_count
    # ) ** (0.5)
    # snr = ((1 - depth_mean) / numpy.std(flux_ootr)) * len(
    #     all_flux_intransit
    # ) ** (0.5)

    #Fold N times, SNRFold = SNR * sqrt(N)
    #Reference: https://dsp.stackexchange.com/questions/26366/how-to-derive-the-results-that-averaging-n-signals-yields-a-sqrtn-fold-in
    snr = np.mean(snr_per_transit) * (len(transit_times)**(0.5))
    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times)**(0.5))
    return periods,period,transit_duration_in_days,BestFitDepth,T0,SDE,chi2,transit_times,power,snr,snr_pink,snrFit,snrFitPink