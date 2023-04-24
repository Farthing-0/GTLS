import numpy
import numpy as np
import cupy as cp
from .stats import spectra,all_transit_times,calculate_transit_duration_in_days,intransit_stats,snr_stats,calcDurationDays
from .helpers import transit_mask
from .transit import mutipleTransitFit
from . import GPUFun
import pynvml
import tqdm

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
    # lc_arr_grazing,
    # lc_arr_box,
    lc_cache_overview,
    T0_fit_margin,
    oversampling_factor,
    verbose,
    useLocalPTXCUBIN = False,
    GPUDeviceID = 0,

    #legacy: Skip-points search, like the original TLS.
    legacy = False
):
    
    # Choose the GPU device
    set_cuda_device(GPUDeviceID)

    # T0_fit_margin is not used for now, because T0_fit_margin is used to skip
    # some points in the search to reduce time in CPU TLS, but GPU is fast enough to search all points.
    # Use TESS data as an example, if we skip 99/100 points,(if the duration is longer than 100 points),
    # the search time will reduce about 0.5s to 0.005s, which is not significant.
    # Maybe we can provide a "Fast" mode in the future.

    # singleCalcPeriods = 130

    # with open ('GPUFun.cu', 'r') as myfile:
    #     myCode=myfile.read()
    options = ('-rdc=true',)
    if not useLocalPTXCUBIN:
        GPUCode = GPUFun.getGPUCode()
        module = cp.RawModule(code=GPUCode,options=options)
    else :
        import os.path
        # options = {}
        # options['rdc'] = 'True'

        if os.path.isfile('./GTLS.ptx'):
            module = cp.RawModule(path='./GTLS.ptx',options=options)
        else:
            module = cp.RawModule(path='./GTLS.cubin',options=options)

    durations = numpy.unique(lc_cache_overview["width_in_samples"])
    maxWidthInSamples = int(max(durations))
    if maxWidthInSamples % 2 != 0:
        maxWidthInSamples = maxWidthInSamples + 1
    durations = numpy.sort(durations)
    
    maxWidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
    patchedDatasSize = int(len(t) + maxWidthInSamples)
    patchedDatasSizeGPU = cp.asarray(np.array([patchedDatasSize])).astype(cp.int32)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    singleCalcPeriods_max = (nvmlinfo.free) / (5*(patchedDatasSize * 2 + 2 + len(durations)*patchedDatasSize*4 + 2*len(durations)))
    singleCalcPeriods = int(np.min([np.floor(singleCalcPeriods_max),len(periods)]))
    print('singleCalcPeriods',singleCalcPeriods)

    print('len period',len(periods))
    print('len duration',len(durations))

    #From now on, due to GPU memory size limitation, GPU can only do several periods(about 100-1000) at a time.
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))

    if TotalIter > 10:
        pbar = tqdm.tqdm(total=TotalIter)

    if verbose:
        print('TotalIter',TotalIter)

    #Initialize the variables
    periodsGPU = cp.empty((singleCalcPeriods,),dtype=cp.float64)
    durationsMaxGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    durationsMinGPU = cp.empty((singleCalcPeriods,),dtype=cp.int32)
    locationGPU = cp.empty(len(periods),dtype=cp.int32)
    LowestResidualsEachPeriodGPU = cp.empty(len(periods),dtype=cp.float32)

    # phasesGPU = cp.empty((len(periods),len(t)),dtype=cp.float64)
    # sortIndexGPU = cp.empty((len(periods),len(t)),dtype=cp.int32)
    # tGPU = cp.asarray(t).astype(cp.float64)


    # For GPU memory limitaion, we can only calculate about
    for iterFlag in range(TotalIter):

        if iterFlag == TotalIter - 1:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:]
            # enlarge the SinglePeriods array to the same size as singleCalcPeriods
            print(len(SinglePeriods))
            SinglePeriods = np.append(SinglePeriods,np.zeros((singleCalcPeriods - len(SinglePeriods),)))
            print('SinglePeriods',SinglePeriods.shape)
        else:
            SinglePeriods = periods[iterFlag*singleCalcPeriods:(iterFlag+1)*singleCalcPeriods]
            print('SinglePeriods',SinglePeriods.shape)
        # continue

        periodsGPU = cp.asarray(SinglePeriods).astype(cp.float64)
        durationsMaxGPU = cp.asarray(SinglePeriods).astype(cp.int32)
        durationsMinGPU = cp.asarray(SinglePeriods).astype(cp.int32)

        # Phase fold
        phasesGPU = cp.empty((singleCalcPeriods,len(t)),dtype=cp.float64)
        sortIndexGPU = cp.empty((singleCalcPeriods,len(t)),dtype=cp.int32)
        tGPU = cp.asarray(t).astype(cp.float64)
        periodsSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        tSizeGPU = cp.asarray(np.array([len(t)])).astype(cp.int32)
        tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

        durationsGridGPU = module.get_function('durationsGrid')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        durationsGridGPU((gridSizeX,1,1),(blockSize,),
                        (periodsGPU,durationsMaxGPU, durationsMinGPU,tLengthGPU,tSizeGPU, periodsSizeGPU))

        fastFoldGPU = module.get_function('foldFast')
        blockSize,gridSizeX = calcGridBlockSize(len(t))
        fastFoldGPU((gridSizeX,singleCalcPeriods,),(blockSize,), (tGPU, periodsGPU,phasesGPU,periodsSizeGPU,tSizeGPU))
        i_max = 10
        for i in range(1,i_max + 1):
            sortIndexGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max] = phasesGPU[(i-1)*singleCalcPeriods/i_max:i*singleCalcPeriods/i_max].argsort()

        patchedDatasGPU = cp.zeros((singleCalcPeriods,len(t) + maxWidthInSamples),dtype=cp.float32)
        patchedDysGPU = cp.empty((singleCalcPeriods,len(t) + maxWidthInSamples),dtype=cp.float32)
        yGPU = cp.asarray(y).astype(cp.float32)
        dyGPU = cp.asarray(dy).astype(cp.float32)

        lc_arr_max_len = np.array([np.max(durations)]).astype(np.int32)
        lc_arr_full_length = 1 - np.array([np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') for x in lc_arr])

        lcArrMaxLenGPU = cp.asarray(lc_arr_max_len).astype(cp.int32)
        lcArrFullLengthGPU = cp.asarray(lc_arr_full_length).astype(cp.float32)

        #Other GPU variables, declare here to save time.
        inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods,len(t) + maxWidthInSamples),dtype=cp.float32)
        edgeEffectCorrectionsGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
        maxwidthInSamplesGPU = cp.asarray(np.array([maxWidthInSamples])).astype(cp.int32)
        periodSizeGPU = cp.asarray(np.array([singleCalcPeriods])).astype(cp.int32)
        durationsGPU = cp.asarray(durations).astype(cp.int32)
        durationsSizeGPU = cp.asarray(np.array([len(durations)])).astype(cp.int32)
        iterFlagGPU = cp.asarray(np.array([0])).astype(cp.int32)

        overshootGPU = cp.array(lc_cache_overview["overshoot"]).astype(cp.float32)
        datapointsGPU = cp.array([len(y)]).astype(cp.int32)

        ## depth variable is not needed anymore because we trapezoid fit the transit to get the depth
        ## This method is much more faster and accurate than producing a depth from the transit model
        # depthsEachPeriodGPU = cp.empty((singleCalcPeriods),dtype=cp.float32)
        transitDepthMinGPU = cp.array([transit_depth_min]).astype(cp.float32)

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
        patchDataGPU((gridSizeX,singleCalcPeriods,),(blockSize,),
        (patchedDatasGPU,patchedDysGPU,patchedDatasSizeGPU,sortIndexGPU,
        maxWidthInSamplesGPU,yGPU,dyGPU,tSizeGPU,periodsSizeGPU))

        calcInverseSquaredPatchedDyGPU = module.get_function('calcInverseSquaredPatchedDy')
        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize)
        calcInverseSquaredPatchedDyGPU((gridSizeX,singleCalcPeriods,1),(blockSize,1,1),
        (inverseSquaredPatchedDysGPU,patchedDysGPU,patchedDatasSizeGPU,))

        #calculate edge_effect_correction
        calcEdgeEffectCorrectionsGPU = module.get_function('calcEdgeEffectCorrections')
        blockSize,gridSizeX = calcGridBlockSize(singleCalcPeriods)
        calcEdgeEffectCorrectionsGPU((gridSizeX,1,1),(blockSize,1,1),
        (edgeEffectCorrectionsGPU,patchedDatasGPU,inverseSquaredPatchedDysGPU,
        patchedDatasSizeGPU,maxwidthInSamplesGPU,periodSizeGPU,))

        
        for i in range(singleCalcPeriods):
            if((iterFlag * singleCalcPeriods + i) < singleCalcPeriods):
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

        if legacy:
            calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsCompatibleGPU')
        else:
            calcAllLowestResidualsGPU = module.get_function('calcAllLowestResidualsGPU')

        blockSize,gridSizeX = calcGridBlockSize(patchedDatasSize - (np.min(durations)) + 1)
        
        #About LowestResidualsTypeGPU, 0 means standard transit, 1 means grazing transit, 2 means box transit,
        #3 means no transit(Or not detect at all)
        #TODO:export all transit types output, distinguish them in the next step? or just use the lowest one? 
        # calcAllLowestResidualsGPU((gridSizeX,len(durations),singleCalcPeriods),
        calcAllLowestResidualsGPU((gridSizeX,singleCalcPeriods,1),
        (blockSize,1,1),(lowestResidualsGPU,#depthsGPU,
        meanSizeGPU,meanXSizeGPU,
        patchedDatasGPU,patchedDatasSizeGPU,durationsGPU,
        durationsSizeGPU,lcArrFullLengthGPU,#lcArrGrazingFullLengthGPU,lcArrBoxFullLengthGPU,
        lcArrMaxLenGPU,inverseSquaredPatchedDysGPU,
        overshootGPU,ootrGPU,fullSumGPU,edgeEffectCorrectionsGPU,datapointsGPU,cumsumGPU,#meanGPU,
        durationsMaxGPU,durationsMinGPU,transitDepthMinGPU,
        iterFlagGPU,singleCalcPeriodsGPU,periodSizeGPU,))
        
        print('iterFlag',iterFlag)
        print('lowestResidualsGPU',edgeEffectCorrectionsGPU)
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
    print('chi2',chi2)
    import matplotlib.pyplot as plt
    # plt.plot(periods,chi2,'.')
    plt.plot(chi2,'.')
    plt.savefig('chi2.png')
    plt.close()

    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)

    #Self Defined metrics
    HighestPowerIndex = numpy.argmax(power)
    # Depth = depthsEachPeriodGPU[HighestPowerIndex].item()    
    period = periods[HighestPowerIndex]

    # BestChi2Index = numpy.argmin(chi2)
    print('locationGPU',locationGPU)
    bestLocation = locationGPU[HighestPowerIndex].item()
    print('bestLocation',bestLocation)
    durationIndex = np.floor(bestLocation / (int(patchedDatasSize) - (np.min(durations)) + 1)).astype(int)
    print('durationIndex',durationIndex)

    bestRow = np.where(lc_cache_overview["width_in_samples"] == durations[durationIndex])[0].item()
    rawDuration = lc_cache_overview['duration'][bestRow]

    bestRowT0 = bestLocation % (int(patchedDatasSize) - (np.min(durations)) + 1)
    transitMean = patchedDatasGPU[HighestPowerIndex][bestRowT0:bestRowT0+durations[durationIndex]].mean()

    # Transit Depth
    overshoot = lc_cache_overview["overshoot"][durationIndex]
    transitDepth =  ((1-transitMean) * overshoot).item()

    #Technically, the "real" trapezoidFitSize = 2 * trapezoidFitSize
    trapezoidFitSize = 100
    trapezoidFitResultGPU = cp.empty((trapezoidFitSize,durations[durationIndex]),dtype=cp.float32)
    trapezoidFitGPU = module.get_function('trapezoidFitAtom')
    blockSize,gridSizeX = calcGridBlockSize(durations[durationIndex])
    trapezoidFitGPU((gridSizeX,trapezoidFitSize,1),(blockSize,1,1),(trapezoidFitResultGPU,
    patchedDatasGPU[HighestPowerIndex],inverseSquaredPatchedDysGPU[HighestPowerIndex],
    cp.int32(durations[durationIndex]),cp.int32(bestRowT0),transitMean,cp.int32(trapezoidFitSize)))

    bestFitTid = cp.int32(cp.sum(trapezoidFitResultGPU,axis=-1).argmin().get())
    BestFitDepthGPU = (trapezoidFitSize * (transitMean) - 0.5*bestFitTid)/(trapezoidFitSize - 0.5*bestFitTid)
    BestFitDepth = BestFitDepthGPU.item()
    dataOutTransit = np.concatenate((patchedDatasGPU[HighestPowerIndex][0:bestRowT0].get(),patchedDatasGPU[HighestPowerIndex][bestRowT0+durations[durationIndex]:].get()))

    # Generate Trapezoid Fit
    bestTrapezoidFitGPU = cp.empty(durations[durationIndex],dtype=cp.float32)
    generateTrapezoidFitGPU = module.get_function('generateTrapezoidFit')
    blockSize,gridSizeX = calcGridBlockSize(durations[durationIndex])
    generateTrapezoidFitGPU((gridSizeX,1,1),(blockSize,1,1),(bestTrapezoidFitGPU,
    bestFitTid,cp.int32(durations[durationIndex]),cp.int32(trapezoidFitSize),cp.float32(BestFitDepth)))

    if bestRowT0 > len(t) - 1:
        bestRowT0 = bestRowT0 - len(t) 

    # -- post Transit Fit
    # print('durations[durationIndex]',durations[durationIndex])
    # transitArr = mutipleTransitFit(durations[durationIndex],transitDepth)
    # transitArrGPU = cp.array(transitArr,dtype=cp.float32)
    # # Find the best transit fit
    # # __global__ void postTransitFitAtom(float *results,
    # # float *inData, float *idealTransit, float *inInverseSquaredDys,
    # # int duration, int inDataSize, int idealTransitSize){    
    # pointResultGPU = cp.empty((durations[durationIndex]),dtype=cp.float32)
    # postTransitFitResultGPU = cp.empty((len(transitArr),len(y)),dtype=cp.float32)
    # postTransitFitGPU = module.get_function('postTransitFitAtom')
    # blockSize,gridSizeX = calcGridBlockSize(len(t))
    # postTransitFitGPU((gridSizeX,len(transitArr),1),(blockSize,1,1),(postTransitFitResultGPU,
    # patchedDatasGPU[HighestPowerIndex],transitArrGPU,inverseSquaredPatchedDysGPU[HighestPowerIndex],
    # cp.int32(durations[durationIndex]),cp.int32(len(t)),cp.int32(len(transitArr)),pointResultGPU))
    # # print('T0FitTest',postTransitFitResultGPU[:,bestRowT0])
    # # print('T0FitTest',postTransitFitResultGPU)
    # bestFitIndex = postTransitFitResultGPU[:,bestRowT0].argmin().item()
    # # print('bestFitIndex',bestFitIndex)
    # idealTransitFitLoss = postTransitFitResultGPU[bestFitIndex].get()
    # # print('idealTransitFitLoss',idealTransitFitLoss)
    # # print('idealTransitFitLoss-T0',idealTransitFitLoss[bestRowT0])
    # lossSDE = abs(idealTransitFitLoss[bestRowT0] - np.mean(idealTransitFitLoss)) / np.std(idealTransitFitLoss)
    # # print('lossSDE',lossSDE)
    # # lossSDE = 
    # # print('T0FitTest',postTransitFitResultGPU[0].shape)

    # # Normalize Trapezoid Fit and patched data
    # patchDataMin = patchedDatasGPU[HighestPowerIndex].min()
    # bestFoldedDataGPU = (patchedDatasGPU[HighestPowerIndex] - patchDataMin)/(1 - patchDataMin)
    # bestTrapezoidFitGPU = (bestTrapezoidFitGPU - patchDataMin)/(1 - patchDataMin)

    # Find loss for each point
    # lossGPU = cp.empty(len(t),dtype=cp.float32)
    # trapezoidSNRlossGPU = module.get_function('trapezoidSNRloss')
    # blockSize,gridSizeX = calcGridBlockSize(len(t))
    # trapezoidSNRlossGPU((gridSizeX,1,1),(blockSize,1,1),(lossGPU,cp.int32(len(t)),patchedDatasGPU[HighestPowerIndex],
    # inverseSquaredPatchedDysGPU[HighestPowerIndex],cp.int32(durations[durationIndex]),bestTrapezoidFitGPU))
    # T0loss = lossGPU[bestRowT0]
    # lossSDE = cp.abs(T0loss - cp.mean(lossGPU))/cp.std(lossGPU)

    # --debug ----
    # lossAtomGPU = cp.empty((len(t),durations[durationIndex]),dtype=cp.float32)
    # trapezoidSNRlossAtomGPU = module.get_function('trapezoidSNRlossAtom')
    # blockSize,gridSizeX = calcGridBlockSize(len(t))
    # trapezoidSNRlossAtomGPU((durations[durationIndex],gridSizeX,1),(1,blockSize,1),(lossAtomGPU,cp.int32(len(t)),patchedDatasGPU[HighestPowerIndex],
    # inverseSquaredPatchedDysGPU[HighestPowerIndex],cp.int32(durations[durationIndex]),bestTrapezoidFitGPU))

    # lossStd = cp.std(lossAtomGPU,axis=-1)
    # T0loss = lossStd[bestRowT0]
    # lossSDE = cp.abs(T0loss - cp.mean(lossStd))/cp.std(lossStd)
    # print('lossSDE',lossSDE.get())
    
    # import matplotlib.pyplot as plt
    # print('bestTrapezoidFitGPU',bestTrapezoidFitGPU.shape)

    # # plt.plot(bestTrapezoidFitGPU.get())
    # plt.plot(lossGPU.get())
    # # plt.plot(lossStd.get())
    # # plt.ylim(0,10**-6)
    # plt.axvline(bestRowT0)
    # plt.savefig('lossGPU.png')
    # plt.close()

    # -- debug end ----

    # dft = cp.abs(cp.fft.rfft(patchedDatasGPU[HighestPowerIndex]))[50:]
    # line = dft[100:].mean() + 3 * dft[100:].std()
    # outlineValue = len([x for x in dft[100:] if x > line])

    # snrFit = (1 - BestFitDepth)*(durations[durationIndex] ** 0.5)/cp.std(trapezoidFitResultGPU[bestFitTid])
    snrFit = (1 - BestFitDepth)*(durations[durationIndex] ** 0.5)/np.std(dataOutTransit)
    DataCumsum = np.cumsum(dataOutTransit)
    DataSlideAvg = (DataCumsum[durations[durationIndex]:] - DataCumsum[:-durations[durationIndex]])/durations[durationIndex]
    redNoise = np.std(DataSlideAvg)

    bestSortIndex = sortIndexGPU[HighestPowerIndex]
    tIndex = bestSortIndex[bestRowT0]
    Tx = t[tIndex.get()]
    T0 = Tx - int((Tx-min(t)) / period) * period - period
    transit_times = all_transit_times(T0, t, period)

    snrFitPink = (1 - BestFitDepth)/((np.std(dataOutTransit)**2/(durations[durationIndex])) + (redNoise**2/(len(transit_times))))**0.5

    # if legacy:
    #     #Raw TLS Calculate transit duration(days) Method
    #     transit_duration_in_days = calculate_transit_duration_in_days(
    #         t, period, transit_times, rawDuration
    #     )
    # else:
    ## Alternative TLS Calculate transit duration(days) Method
    transit_duration_in_days = calcDurationDays(t, period, T0, rawDuration)
    # transit_duration_in_days = calcDurationDays(t, period, T0, durations[durationIndex])
    

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
    
    # cp.cuda.runtime.deviceSynchronize()
    # print('After main search, time used:',time.time() - start,'s')
    # outlineValue = None

    # # KLoss and KLossStd, can be optimized
    # def centerFold(time, period, T0):
    #     """Normal phase folding"""
    #     T0 = T0 + period/2
    #     return (time - T0) / period - np.floor((time - T0) / period)

    # phases = centerFold(t, period, T0)
    # phasesIndex = np.argsort(phases)
    # phasesSorted = phases[phasesIndex]
    # fluxesSorted = y[phasesIndex]
    # def chunks(lst, n):
    #     """Yield successive n-sized chunks from lst."""
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]

    # # KLoss and KLossStd
    # polyFitSize = int(durations[durationIndex] / 2)
    # leftLimit = int(len(t)/2 - durations[durationIndex]/2)
    # rightLimit = int(len(t)/2 + durations[durationIndex]/2)
    # splitFluxes = list(chunks(fluxesSorted[0:leftLimit],polyFitSize))
    # splitFluxes = splitFluxes + (list(chunks(fluxesSorted[rightLimit:],polyFitSize)))
    # para = []
    # for flux in splitFluxes:
    #     if len(flux) < 2:
    #         continue
    #     p = np.polyfit(range(len(flux)),flux,1)
    #     para.append(p)

    # standardK = (2*(1-BestFitDepth))/durations[durationIndex]
    # # fluxK = (np.array(para)[:,0])**2
    # # KLossStd = np.std(fluxK) / standardK**2
    # # KLossMean = np.sum(fluxK) / standardK**2 / len(para)

    lossSDE = None
    KLossStd = None
    KLossMean = None

    # # print('durationsGPU',durationsGPU.get())
    return periods,period,rawDuration,durations[durationIndex],transit_duration_in_days,transitDepth,T0,SDE,chi2,transit_times,power,snr,snr_pink,snrFit,snrFitPink,lossSDE,KLossMean,KLossStd