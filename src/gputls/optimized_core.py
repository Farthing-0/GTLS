import numpy as np
import cupy as cp
from .stats import spectra, all_transit_times, calculate_transit_duration_in_days
from .stats import intransit_stats, snr_stats, calcDurationDays, alignDifferentBinChi2, findPossibleFitPeriods
from .helpers import transit_mask
from .transit import mutipleTransitFit
from . import GPUFun
from .memory_manager import GTLSMemoryManager
from .stream_manager import PipelinedGTLSProcessor
import pynvml
import tqdm
import logging

def optimized_search_multi_periods(
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
    GPUDeviceID=0,
    fast=False,
    legacy=False,
    SimplifyEdgeEffect=True,
    bar_location=0
):
    """
    优化的多周期搜索函数，集成内存池管理和流处理
    """
    
    # 初始化优化组件
    memory_manager = GTLSMemoryManager(GPUDeviceID)
    pipeline_processor = PipelinedGTLSProcessor(GPUDeviceID, num_streams=4)
    
    # 设置GPU设备
    cp.cuda.Device(GPUDeviceID).use()
    
    # 编译GPU代码
    GPUCode = GPUFun.getGPUCode()
    if T0_fit_margin == 0:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + '0x7f800000')
    else:
        GPUCode = GPUCode.replace('#define SKIP_POINT 8', '#define SKIP_POINT ' + str(int(1/T0_fit_margin)))
    
    module = cp.RawModule(code=GPUCode)
    module.compile()
    
    # 预处理持续时间数据
    durations, indices = np.unique(lc_cache_overview["width_in_samples"], return_index=True)
    lc_arr = lc_arr[indices]
    lc_cache_overview = lc_cache_overview[indices]
    maxDuration = int(max(durations))
    
    if maxDuration % 2 != 0:
        maxDuration = maxDuration + 1
    
    durations = np.sort(durations)
    tSize = len(t)
    patchedDatasSize = int(tSize + maxDuration)
    
    # 计算最优批处理大小
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)
    nvmlinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # 更精确的内存估算
    memory_per_period = (
        patchedDatasSize * 6 * 4 +  # 主要数组 (float32)
        tSize * 4 * 4 +             # 索引和相位数组
        len(durations) * (patchedDatasSize + tSize) * 4  # 结果数组
    )
    
    # 保留30%内存作为缓冲
    available_memory = nvmlinfo.free * 0.7
    singleCalcPeriods = max(15, int(available_memory / memory_per_period))
    singleCalcPeriods = min(singleCalcPeriods, len(periods))
    
    if verbose:
        logging.info(f"Optimized batch size: {singleCalcPeriods} periods per iteration")
        logging.info(f"Estimated memory per period: {memory_per_period / 1024**2:.2f} MB")
    
    TotalIter = int(np.ceil(len(periods) / singleCalcPeriods))
    
    if verbose:
        pbar = tqdm.tqdm(total=TotalIter, position=bar_location)
    
    # 分配持久化数组
    locationGPU = memory_manager.allocate_persistent(
        'locationGPU', (len(periods),), cp.int32)
    LowestResidualsEachPeriodGPU = memory_manager.allocate_persistent(
        'LowestResidualsEachPeriodGPU', (len(periods),), cp.float32)
    
    # 预处理输入数据到GPU
    tGPU = memory_manager.allocate_persistent('tGPU', (tSize,), cp.float64)
    yGPU = memory_manager.allocate_persistent('yGPU', (tSize,), cp.float32)
    dyGPU = memory_manager.allocate_persistent('dyGPU', (tSize,), cp.float32)
    
    # 异步传输输入数据
    transfer_event1 = pipeline_processor.async_data_transfer_and_compute(
        t, tGPU, lambda: None)
    transfer_event2 = pipeline_processor.async_data_transfer_and_compute(
        y, yGPU, lambda: None)
    transfer_event3 = pipeline_processor.async_data_transfer_and_compute(
        dy, dyGPU, lambda: None)
    
    # 等待数据传输完成
    transfer_event1.synchronize()
    transfer_event2.synchronize()
    transfer_event3.synchronize()
    
    # 定义单次迭代处理函数
    def process_single_iteration(period_batch, actual_size, iter_flag):
        """处理单次迭代的函数"""
        
        # 分配工作空间内存
        workspace = memory_manager.allocate_workspace(
            singleCalcPeriods, tSize, maxDuration, len(durations))
        
        try:
            # 设置周期数据
            workspace['periodsGPU'][:actual_size] = cp.asarray(period_batch[:actual_size])
            
            # 常量参数
            periodsSizeGPU = cp.array([singleCalcPeriods], dtype=cp.int32)
            tSizeGPU = cp.array([tSize], dtype=cp.int32)
            tLengthGPU = cp.array([max(t) - min(t)], dtype=cp.float32)
            patchedDatasSizeGPU = cp.array([patchedDatasSize], dtype=cp.int32)
            maxDurationGPU = cp.array([maxDuration], dtype=cp.int32)
            periodSizeGPU = cp.array([singleCalcPeriods], dtype=cp.int32)
            durationsGPU = cp.asarray(durations, dtype=cp.int32)
            durationsSizeGPU = cp.array([len(durations)], dtype=cp.int32)
            
            # 核心函数执行序列
            kernels_and_args = [
                # 1. 计算持续时间网格
                (module.get_function('durationsGrid'), 
                 (workspace['periodsGPU'], workspace['durationsMaxGPU'], workspace['durationsMinGPU'],
                  tLengthGPU, tSizeGPU, periodsSizeGPU), 0),
                
                # 2. 相位折叠
                (module.get_function('foldFast'),
                 (tGPU, workspace['periodsGPU'], workspace['phasesGPU'], 
                  periodsSizeGPU, tSizeGPU), 1),
            ]
            
            # 执行前两个核心函数
            pipeline_processor.multi_stream_kernel_execution(kernels_and_args[:2])
            
            # 计算排序索引（需要在主机上完成）
            phases_host = workspace['phasesGPU'].get()
            for i in range(actual_size):
                sort_indices = np.argsort(phases_host[i])
                workspace['sortIndexGPU'][i] = cp.asarray(sort_indices)
            
            # 继续执行其他核心函数
            remaining_kernels = [
                # 3. 数据补丁
                (module.get_function('patchData'),
                 (workspace['patchedDatasGPU'], workspace['patchedDysGPU'], 
                  patchedDatasSizeGPU, workspace['sortIndexGPU'], maxDurationGPU,
                  yGPU, dyGPU, tSizeGPU), 0),
                
                # 4. 计算逆平方误差
                (module.get_function('calcInverseSquaredPatchedDy'),
                 (workspace['inverseSquaredPatchedDysGPU'], workspace['patchedDysGPU'], 
                  patchedDatasSizeGPU), 1),
                
                # 5. 边缘效应校正
                (module.get_function('calcEdgeEffectCorrections'),
                 (workspace['edgeEffectCorrectionsGPU'], workspace['patchedDatasGPU'],
                  workspace['inverseSquaredPatchedDysGPU'], patchedDatasSizeGPU,
                  maxDurationGPU, periodSizeGPU), 2),
            ]
            
            # 定义依赖关系：3依赖2完成，4依赖3完成，5依赖3和4完成
            dependencies = [[], [0], [1], [2]]
            
            pipeline_processor.multi_stream_kernel_execution(
                remaining_kernels, dependencies)
            
            # 计算累积和（在CPU上更高效）
            patched_data_host = workspace['patchedDatasGPU'].get()
            for i in range(actual_size):
                workspace['cumsumGPU'][i] = cp.cumsum(workspace['patchedDatasGPU'][i])
            
            # 最终计算核心函数
            final_kernels = [
                (module.get_function('calcAllFullSum'),
                 (workspace['fullSumGPU'], workspace['patchedDatasGPU'],
                  workspace['inverseSquaredPatchedDysGPU'], patchedDatasSizeGPU,
                  durationsGPU, durationsSizeGPU, periodSizeGPU), 0),
                
                (module.get_function('calcAllOutOfTransitResiduals_step1_2GPU'),
                 (workspace['ootrGPU'], workspace['patchedDatasGPU'], durationsGPU,
                  durationsSizeGPU, workspace['inverseSquaredPatchedDysGPU'],
                  patchedDatasSizeGPU, tSizeGPU), 1),
            ]
            
            pipeline_processor.multi_stream_kernel_execution(final_kernels)
            
            # 累积和OOTR
            workspace['ootrGPU'] = cp.cumsum(workspace['ootrGPU'], axis=-1)
            
            # 最终残差计算
            overshootGPU = cp.array(lc_cache_overview["overshoot"], dtype=cp.float32)
            datapointsGPU = cp.array([len(y)], dtype=cp.int32)
            transitDepthMinGPU = cp.array([transit_depth_min], dtype=cp.float32)
            
            # 预处理LC数据
            lc_arr_max_len = np.array([np.max(durations)], dtype=np.int32)
            lc_arr_full_length = 1 - np.array([
                np.pad(x, (0, lc_arr_max_len[0] - len(x)), 'constant') 
                for x in lc_arr
            ])
            
            lcArrMaxLenGPU = cp.asarray(lc_arr_max_len, dtype=cp.int32)
            lcArrFullLengthGPU = cp.asarray(lc_arr_full_length, dtype=cp.float32)
            
            # 执行最终残差计算
            with pipeline_processor.computation_stream:
                module.get_function('calcAllLowestResidualsGPUB')(
                    (calcGridBlockSize(tSize)[1], len(durations), singleCalcPeriods),
                    (calcGridBlockSize(tSize)[0], 1, 1),
                    (workspace['lowestResidualsGPU'], tSizeGPU,
                     workspace['patchedDatasGPU'], patchedDatasSizeGPU,
                     durationsGPU, durationsSizeGPU, lcArrFullLengthGPU,
                     lcArrMaxLenGPU, workspace['inverseSquaredPatchedDysGPU'],
                     overshootGPU, workspace['ootrGPU'], workspace['fullSumGPU'],
                     workspace['edgeEffectCorrectionsGPU'], datapointsGPU,
                     workspace['cumsumGPU'], workspace['durationsMaxGPU'],
                     workspace['durationsMinGPU'], transitDepthMinGPU))
            
            # 同步并获取结果
            cp.cuda.Stream.null.synchronize()
            
            # 提取结果
            start_idx = iter_flag * singleCalcPeriods
            end_idx = min(start_idx + actual_size, len(periods))
            
            locationGPU[start_idx:end_idx] = workspace['lowestResidualsGPU'][:actual_size].argmin(axis=(1, 2))
            LowestResidualsEachPeriodGPU[start_idx:end_idx] = workspace['lowestResidualsGPU'][:actual_size].min(axis=(1, 2))
            
            return True
            
        finally:
            # 释放工作空间内存
            memory_manager.deallocate_workspace(workspace)
    
    # 使用流水线处理所有迭代
    def process_batch_wrapper(period_batch, actual_size):
        return process_single_iteration(period_batch, actual_size, 0)
    
    # 手动迭代处理（更好的内存控制）
    for iter_flag in range(TotalIter):
        start_idx = iter_flag * singleCalcPeriods
        end_idx = min((iter_flag + 1) * singleCalcPeriods, len(periods))
        actual_size = end_idx - start_idx
        
        period_batch = periods[start_idx:end_idx]
        if len(period_batch) < singleCalcPeriods:
            period_batch = np.append(period_batch, 
                                   np.zeros(singleCalcPeriods - len(period_batch)))
        
        process_single_iteration(period_batch, actual_size, iter_flag)
        
        if verbose:
            pbar.update(1)
    
    # 获取最终结果
    chi2 = LowestResidualsEachPeriodGPU.get()
    
    # 计算谱特征
    SR, power_raw, power, SDE_raw, SDE = spectra(chi2, oversampling_factor)
    HighestPowerIndex = np.argmax(power)
    period = periods[HighestPowerIndex]
    
    # 清理持久化内存
    memory_manager.clear_persistent()
    
    if verbose:
        pbar.close()
        logging.info("Optimized search completed")
    
    if fast:
        return periods, power
    else:
        # 返回完整结果（需要实现其他计算）
        # 这里简化返回
        return periods, period, None, None, None, None, None, SDE, chi2, None, power, None, None, None, None


def calcGridBlockSize(size):
    """计算网格和线程块大小"""
    MAX_BLOCK_SIZE = 256  # 优化：增加到256
    blockSize = min(size, MAX_BLOCK_SIZE)
    gridSizeX = int((size + blockSize - 1) // blockSize)
    return blockSize, gridSizeX