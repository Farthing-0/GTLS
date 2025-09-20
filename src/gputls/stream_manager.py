import cupy as cp
import numpy as np
from typing import List, Optional, Callable, Any
import threading
import queue
import time

class CUDAStreamManager:
    """CUDA流管理器，实现计算和数据传输的重叠"""
    
    def __init__(self, num_streams: int = 4, device_id: int = 0):
        """
        初始化CUDA流管理器
        
        Args:
            num_streams: 并行流的数量
            device_id: GPU设备ID
        """
        self.device_id = device_id
        self.num_streams = num_streams
        
        # 设置GPU设备
        cp.cuda.Device(device_id).use()
        
        # 创建CUDA流
        self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
        self.stream_queue = queue.Queue()
        
        # 初始化流队列
        for stream in self.streams:
            self.stream_queue.put(stream)
        
        # 事件用于同步
        self.events = [cp.cuda.Event() for _ in range(num_streams)]
        self.event_queue = queue.Queue()
        for event in self.events:
            self.event_queue.put(event)
    
    def get_stream(self) -> cp.cuda.Stream:
        """获取可用的CUDA流"""
        return self.stream_queue.get()
    
    def return_stream(self, stream: cp.cuda.Stream):
        """归还CUDA流"""
        self.stream_queue.put(stream)
    
    def get_event(self) -> cp.cuda.Event:
        """获取可用的事件"""
        return self.event_queue.get()
    
    def return_event(self, event: cp.cuda.Event):
        """归还事件"""
        self.event_queue.put(event)
    
    def async_memory_copy(self, src: np.ndarray, dst: cp.ndarray, 
                         stream: Optional[cp.cuda.Stream] = None) -> cp.cuda.Stream:
        """
        异步内存拷贝
        
        Args:
            src: 源数组（CPU）
            dst: 目标数组（GPU）
            stream: CUDA流，如果为None则获取新流
            
        Returns:
            使用的CUDA流
        """
        if stream is None:
            stream = self.get_stream()
        
        with stream:
            cp.asarray(src, dtype=dst.dtype, order='C').copy_to(dst)
        
        return stream
    
    def synchronize_all(self):
        """同步所有流"""
        for stream in self.streams:
            stream.synchronize()


class PipelinedGTLSProcessor:
    """流水线化的GTLS处理器，实现计算和数据传输重叠"""
    
    def __init__(self, device_id: int = 0, num_streams: int = 4):
        self.device_id = device_id
        self.stream_manager = CUDAStreamManager(num_streams, device_id)
        self.computation_stream = self.stream_manager.get_stream()
        self.transfer_streams = [self.stream_manager.get_stream() 
                               for _ in range(num_streams - 1)]
    
    def pipeline_process_periods(self, periods: np.ndarray, singleCalcPeriods: int,
                               process_func: Callable, **kwargs) -> List[Any]:
        """
        流水线处理周期数据
        
        Args:
            periods: 周期数组
            singleCalcPeriods: 单次计算的周期数
            process_func: 处理函数
            **kwargs: 传递给处理函数的额外参数
            
        Returns:
            处理结果列表
        """
        total_iters = int(np.ceil(len(periods) / singleCalcPeriods))
        results = []
        
        # 创建流水线阶段
        stages = {
            'data_prep': queue.Queue(maxsize=2),
            'computation': queue.Queue(maxsize=2),
            'result_collection': queue.Queue(maxsize=2)
        }
        
        # 数据准备线程
        def data_preparation_thread():
            for iter_flag in range(total_iters):
                start_idx = iter_flag * singleCalcPeriods
                end_idx = min((iter_flag + 1) * singleCalcPeriods, len(periods))
                period_batch = periods[start_idx:end_idx]
                
                # 如果不足，填充到singleCalcPeriods大小
                if len(period_batch) < singleCalcPeriods:
                    period_batch = np.append(period_batch, 
                                           np.zeros(singleCalcPeriods - len(period_batch)))
                
                stages['data_prep'].put((iter_flag, period_batch, end_idx - start_idx))
        
        # 计算线程
        def computation_thread():
            while True:
                try:
                    iter_flag, period_batch, actual_size = stages['data_prep'].get(timeout=5)
                    
                    # 使用计算流进行处理
                    with self.computation_stream:
                        result = process_func(period_batch, actual_size, **kwargs)
                    
                    stages['computation'].put((iter_flag, result))
                    stages['data_prep'].task_done()
                    
                except queue.Empty:
                    break
        
        # 结果收集线程
        def result_collection_thread():
            collected_results = {}
            while len(collected_results) < total_iters:
                try:
                    iter_flag, result = stages['computation'].get(timeout=5)
                    collected_results[iter_flag] = result
                    stages['computation'].task_done()
                except queue.Empty:
                    continue
            
            # 按顺序整理结果
            for i in range(total_iters):
                results.append(collected_results[i])
        
        # 启动线程
        threads = [
            threading.Thread(target=data_preparation_thread),
            threading.Thread(target=computation_thread),
            threading.Thread(target=result_collection_thread)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        return results
    
    def async_data_transfer_and_compute(self, host_data: np.ndarray, 
                                      device_buffer: cp.ndarray,
                                      compute_func: Callable,
                                      compute_args: tuple = ()) -> cp.cuda.Event:
        """
        异步数据传输和计算
        
        Args:
            host_data: CPU端数据
            device_buffer: GPU端缓冲区
            compute_func: 计算函数
            compute_args: 计算函数参数
            
        Returns:
            计算完成事件
        """
        transfer_stream = self.transfer_streams[0]
        compute_stream = self.computation_stream
        
        # 异步传输数据
        with transfer_stream:
            device_buffer.set(host_data)
        
        # 创建事件标记传输完成
        transfer_event = self.stream_manager.get_event()
        transfer_event.record(transfer_stream)
        
        # 等待传输完成后开始计算
        compute_stream.wait_event(transfer_event)
        
        # 异步执行计算
        with compute_stream:
            result = compute_func(*compute_args)
        
        # 创建计算完成事件
        compute_event = self.stream_manager.get_event()
        compute_event.record(compute_stream)
        
        # 归还传输事件
        self.stream_manager.return_event(transfer_event)
        
        return compute_event
    
    def multi_stream_kernel_execution(self, kernels_and_args: List[tuple], 
                                    dependencies: Optional[List[List[int]]] = None):
        """
        多流并行执行多个核心函数
        
        Args:
            kernels_and_args: [(kernel_func, args, stream_id), ...]
            dependencies: 依赖关系，dependencies[i]表示第i个核心依赖的核心索引
        """
        events = []
        
        for i, (kernel_func, args, stream_id) in enumerate(kernels_and_args):
            stream = self.transfer_streams[stream_id % len(self.transfer_streams)]
            
            # 等待依赖的核心完成
            if dependencies and i < len(dependencies):
                for dep_idx in dependencies[i]:
                    if dep_idx < len(events):
                        stream.wait_event(events[dep_idx])
            
            # 执行核心函数
            with stream:
                kernel_func(*args)
            
            # 记录完成事件
            event = self.stream_manager.get_event()
            event.record(stream)
            events.append(event)
        
        # 等待所有核心完成
        for event in events:
            event.synchronize()
            self.stream_manager.return_event(event)
    
    def __del__(self):
        """清理资源"""
        self.stream_manager.synchronize_all()
        self.stream_manager.return_stream(self.computation_stream)
        for stream in self.transfer_streams:
            self.stream_manager.return_stream(stream)