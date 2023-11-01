import numpy
import numpy as np
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

    # singleCalcPeriods = 130

    # with open ('GPUFun.cu', 'r') as myfile:
    #     myCode=myfile.read()
    # options = ('-rdc=true',)
    # if not useLocalPTXCUBIN:
    GPUCode = GPUFun.getGPUCode()
    GPUCode = GPUCode.replace('#define SKIP_POINT 8','#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
    # module = cp.RawModule(code=GPUCode,options=options)
    module = cp.RawModule(code=GPUCode)
    module.compile()
    # else :
    #     import os.path
    #     # options = {}
    #     # options['rdc'] = 'True'

    #     if os.path.isfile('./GTLS.ptx'):
    #         module = cp.RawModule(path='./GTLS.ptx',options=options)
    #     else:
    #         module = cp.RawModule(path='./GTLS.cubin',options=options)

    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = numpy.sort(durations)
    
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))

    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods)]))
    # print(singleCalcPeriods)
    # print('size: ',len(durations)*patchedDatasSize*singleCalcPeriods / 1024 /1024)
    # exit()

    if singleCalcPeriods < 15:
        singleCalcPeriods = int(singleCalcPeriods / 1.1)
        # singleCalcPeriods = singleCalcPeriods - 1
    # # exit()
    #From now on, due to GPU memory size limitation, GPU can only do several periods(about 100-1000) at a time.
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    pbar = tqdm.tqdm(total=TotalIter,position=bar_location)

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    # iterFlagGPU = cp.asarray(np.array([0])).astype(cp.int32)
    iterFlagGPU = cp.int32(0)

    # phasesGPU = cp.empty((len(periods),tSize),dtype=cp.float64)
    # sortIndexGPU = cp.empty((len(periods),tSize),dtype=cp.int32)
    # tGPU = cp.asarray(t).astype(cp.float64)


    # For GPU memory limitaion, we can only calculate about
    for iterFlag in range(TotalIter):

        # lowestResidualsGPU = cp.empty((singleCalcPeriods,len(durations),(int(patchedDatasSize) - (np.min(durations)) + 1)),dtype=cp.float32)
        lowestResidualsGPU = cp.empty((singleCalcPeriods,len(durations),tSize),dtype=cp.float32)

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            # enlarge the SinglePeriods array to the same size as singleCalcPeriods
            SinglePeriods = np.append(SinglePeriods,np.zeros((singleCalcPeriods - len(SinglePeriods),)))
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
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

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
        # if singleCalcPeriods > 100:
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()
        # else:
        #     for i in range(singleCalcPeriods):
        #         sortIndexGPU[i] = phasesGPU[i].argsort()

        patchedDatasGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        patchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        yGPU = cp.asarray(y).astype(cp.float32)
        dyGPU = cp.asarray(dy).astype(cp.float32)

        lc_arr_max_len = np.array([np.max(durations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr])

        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
        lcArrFullLengthSizeGPU = cp.asarray(np.array([len(lc_arr_full_length)])).astype(cp.int32)

        #Other GPU variables, declare here to save time.
        
        # if SimplifyEdgeEffect == False:
        edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
        # else:
        #     pass

        
        inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
        maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
        periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        durationsGPU = cp.asarray(durations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)

        overshootGPU = cp.array(lc_cache_overview["overshoot"]).astype(cp.float32)
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

        # if SimplifyEdgeEffect == False:
        #calculate edge_effect_correction
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

        # calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUA')
        # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        # calcAllLowestResidualsGPU((gridSizeX,singleCalcPeriods,1),
        # (blockSize,1,1),(lowestResidualsGPU,resultArrayXAxisSizeGPU,
        # patchedDatasGPU,patchedDatasSizeGPU,
        # durationsGPU,durationsSizeGPU,
        # lcArrFullLengthGPU,
        # lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        # overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
        # durationsMaxGPU,durationsMinGPU,transitDepthMinGPU
        # ))

        calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPUB')
        # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        blockSize,gridSizeX = calcGridBlockSize(tSize)
        # calcAllLowestResidualsGPU((gridSizeX,singleCalcPeriods,len(durations)),
        calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
        (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,
        durationsGPU,durationsSizeGPU,
        lcArrFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
        durationsMaxGPU,durationsMinGPU,transitDepthMinGPU
        ))

        #find best fit
        for i in range(singleCalcPeriods):
            if(iterFlag*singleCalcPeriods + i < len(periods)):
                locationGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].argmin()
                LowestResidualsEachPeriodGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].min()

        iterFlagGPU = iterFlagGPU + 1

        if verbose:
            pbar.update(1)

    chi2 = LowestResidualsEachPeriodGPU.get()

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)

    if fast:
        return periods,power

    HighestPowerIndex = numpy.argmax(power)
    period = periods[HighestPowerIndex]

    bestLocation = locationGPU[HighestPowerIndex].item()
    # durationIndex = np.floor(bestLocation / (int(patchedDatasSize) - (np.min(durations)) + 1)).astype(int)
    durationIndex = np.floor(bestLocation / (tSize)).astype(int)    

    durationPointsNum = durations[durationIndex]

    # need to do 
    refindT0 = True
    if refindT0:
        pass

    bestRow = np.where(lc_cache_overview["width_in_samples"] == durationPointsNum)[0].item()
    rawDuration = lc_cache_overview['duration'][bestRow]

    bestTime,bestFlux,bestFluxDy = foldCPU(t,y,dy,period)
    bestFlux = np.concatenate((bestFlux,bestFlux[:maxDuration]))
    bestFluxDy = np.concatenate((bestFluxDy,bestFluxDy[:maxDuration]))

    bestRowT0 = bestLocation % (tSize)

    transitMean = bestFlux[bestRowT0:bestRowT0+durationPointsNum].mean()

    # Transit Depth
    overshoot = lc_cache_overview["overshoot"][durationIndex]
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


    # if legacy:
    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
    t, y, transit_times, transit_duration_in_days
    )
    # print('transit_times, transit_duration_in_days',transit_times, transit_duration_in_days)
    all_flux_intransit = numpy.concatenate(
        [all_flux_intransit_odd, all_flux_intransit_even]
    )
    intransit = transit_mask(t, period, 2 * rawDuration, T0)
    flux_ootr = y[~intransit]
    depth_mean = numpy.mean(all_flux_intransit)
    # depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
    #     per_transit_count
    # ) ** (0.5)
    snr = ((1 - depth_mean) / numpy.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)
    # else:
    #     #Fold N times, SNRFold = SNR * sqrt(N)
    #     #Reference: https://dsp.stackexchange.com/questions/26366/how-to-derive-the-results-that-averaging-n-signals-yields-a-sqrtn-fold-in
    #     snr = np.mean(snr_per_transit) * (len(transit_times)**(0.5))


    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times)**(0.5))

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

    return periods,period,rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,SDE,chi2,transit_times,power,snr,snr_pink,snrFit,snrFitPink

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

    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxDuration = int(max(durations))

    # why?
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = numpy.sort(durations)
    
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
    durations  = np.array([x for x in durations if x <= durationMax and x >= durationMin])
    lc_arr = [x for x in lc_arr if len(x) <= durationMax and len(x) >= durationMin]

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
    lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr])

    lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
    lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)
    lcArrFullLengthSizeGPU = cp.asarray(np.array([len(lc_arr_full_length)])).astype(cp.int32)

    #Other GPU variables, declare here to save time.
    
    # if SimplifyEdgeEffect == False:
    edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
    # else:
    #     pass

    
    inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,tSize + maxDuration),dtype=cp.float32)
    maxDurationGPU = cp.asarray(np.array([maxDuration])).astype(cp.int32)
    periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
    durationsGPU = cp.asarray(durations).astype(cp.int32)
    durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)

    overshootGPU = cp.array(lc_cache_overview["overshoot"]).astype(cp.float32)
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

    # if SimplifyEdgeEffect == False:
    #calculate edge_effect_correction
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
    # blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
    blockSize,gridSizeX = calcGridBlockSize(tSize)
    # calcAllLowestResidualsGPU((gridSizeX,singleCalcPeriods,len(durations)),
    calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
    (blockSize,1,1),(lowestResidualsGPU,tSizeGPU,
    patchedDatasGPU,patchedDatasSizeGPU,
    durationsGPU,durationsSizeGPU,
    lcArrFullLengthGPU,
    lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
    overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,
    durationsMaxGPU,durationsMinGPU,transitDepthMinGPU
    ))

    # #find best fit
    # for i in range(singleCalcPeriods):
    # print(lowestResidualsGPU)
    # exit()
    bestLocation = lowestResidualsGPU.argmin().get()
    # print('bestLocation',bestLocation)
        # LowestResidualsEachPeriodGPU[iterFlag*singleCalcPeriods + i] = lowestResidualsGPU[i].min()

    durationIndex = np.floor(bestLocation / (tSize)).astype(int)    
    durationPointsNum = durations[durationIndex]

    bestRow = np.where(lc_cache_overview["width_in_samples"] == durationPointsNum)[0].item()
    rawDuration = lc_cache_overview['duration'][bestRow]

    bestTime,bestFlux,bestFluxDy = foldCPU(t,y,dy,period)
    bestFlux = np.concatenate((bestFlux,bestFlux[:maxDuration]))
    bestFluxDy = np.concatenate((bestFluxDy,bestFluxDy[:maxDuration]))

    bestRowT0 = bestLocation % (tSize)

    transitMean = bestFlux[bestRowT0:bestRowT0+durationPointsNum].mean()

    # Transit Depth
    overshoot = lc_cache_overview["overshoot"][durationIndex]
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

    # if legacy:
    depth_mean_odd, depth_mean_even, depth_mean_odd_std, depth_mean_even_std, all_flux_intransit_odd, all_flux_intransit_even, per_transit_count, transit_depths, transit_depths_uncertainties = intransit_stats(
    t, y, transit_times, transit_duration_in_days
    )
    # print('transit_times, transit_duration_in_days',transit_times, transit_duration_in_days)
    all_flux_intransit = numpy.concatenate(
        [all_flux_intransit_odd, all_flux_intransit_even]
    )
    intransit = transit_mask(t, period, 2 * rawDuration, T0)
    flux_ootr = y[~intransit]
    depth_mean = numpy.mean(all_flux_intransit)
    # depth_mean_std = numpy.std(all_flux_intransit) / numpy.sum(
    #     per_transit_count
    # ) ** (0.5)
    snr = ((1 - depth_mean) / numpy.std(flux_ootr)) * len(all_flux_intransit) ** (0.5)
    # else:
    #     #Fold N times, SNRFold = SNR * sqrt(N)
    #     #Reference: https://dsp.stackexchange.com/questions/26366/how-to-derive-the-results-that-averaging-n-signals-yields-a-sqrtn-fold-in
    #     snr = np.mean(snr_per_transit) * (len(transit_times)**(0.5))

    snr_pink = np.mean(snr_pink_per_transit) * (len(transit_times)**(0.5))
    
    return rawDuration,durationPointsNum,transit_duration_in_days,transitDepth,T0,transit_times,snr,snr_pink,snrFit,snrFitPink