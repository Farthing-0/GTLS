import cupy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
import logging

class GPUMemoryPool:
    """GPU内存池管理器，避免重复分配和释放GPU内存"""
    
    def __init__(self, device_id: int = 0, alignment: int = 256):
        """
        初始化GPU内存池
        
        Args:
            device_id: GPU设备ID
            alignment: 内存对齐字节数，通常为256字节
        """
        self.device_id = device_id
        self.alignment = alignment
        self.pools: Dict[str, List[cp.ndarray]] = {}
        self.allocated_sizes: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # 设置GPU设备
        cp.cuda.Device(device_id).use()
        
        # 预分配的内存类型和大小
        self.pool_types = {
            'float32': cp.float32,
            'float64': cp.float64,
            'int32': cp.int32,
            'int64': cp.int64
        }
        
        # 初始化每种类型的内存池
        for dtype_str in self.pool_types:
            self.pools[dtype_str] = []
            self.allocated_sizes[dtype_str] = 0
    
    def _align_size(self, size: int) -> int:
        """将大小对齐到指定字节边界"""
        return ((size + self.alignment - 1) // self.alignment) * self.alignment
    
    def _get_pool_key(self, shape: Tuple[int, ...], dtype: cp.dtype) -> str:
        """生成内存池键值"""
        dtype_str = str(dtype).split('.')[-1]
        return f"{dtype_str}_{np.prod(shape)}"
    
    def allocate(self, shape: Tuple[int, ...], dtype: cp.dtype) -> cp.ndarray:
        """
        从内存池分配内存
        
        Args:
            shape: 数组形状
            dtype: 数据类型
            
        Returns:
            分配的CuPy数组
        """
        with self.lock:
            dtype_str = str(dtype).split('.')[-1]
            total_size = np.prod(shape)
            aligned_size = self._align_size(total_size)
            
            # 查找合适的内存块
            pool = self.pools.get(dtype_str, [])
            for i, array in enumerate(pool):
                if array.size >= aligned_size:
                    # 找到合适的内存块，从池中移除并返回
                    allocated_array = pool.pop(i)
                    # 返回所需大小的视图
                    return allocated_array[:total_size].reshape(shape)
            
            # 没有找到合适的内存块，分配新的
            try:
                new_array = cp.empty(aligned_size, dtype=dtype)
                self.allocated_sizes[dtype_str] += aligned_size * dtype().itemsize
                logging.info(f"Allocated new GPU memory: {aligned_size * dtype().itemsize / 1024**2:.2f} MB")
                return new_array[:total_size].reshape(shape)
            except cp.cuda.memory.OutOfMemoryError:
                # 内存不足，清理池并重试
                self.clear_pool(dtype_str)
                new_array = cp.empty(aligned_size, dtype=dtype)
                return new_array[:total_size].reshape(shape)
    
    def deallocate(self, array: cp.ndarray):
        """
        将内存块返回到内存池
        
        Args:
            array: 要释放的CuPy数组
        """
        with self.lock:
            dtype_str = str(array.dtype).split('.')[-1]
            if dtype_str in self.pools:
                # 将扁平化的数组添加到池中
                flattened = array.ravel()
                self.pools[dtype_str].append(flattened)
    
    def clear_pool(self, dtype_str: Optional[str] = None):
        """
        清空指定类型的内存池
        
        Args:
            dtype_str: 数据类型字符串，None表示清空所有池
        """
        with self.lock:
            if dtype_str:
                if dtype_str in self.pools:
                    del self.pools[dtype_str][:]
                    self.allocated_sizes[dtype_str] = 0
            else:
                for key in self.pools:
                    del self.pools[key][:]
                    self.allocated_sizes[key] = 0
    
    def get_memory_info(self) -> Dict[str, int]:
        """获取内存使用信息"""
        return self.allocated_sizes.copy()
    
    def __del__(self):
        """析构函数，清理所有内存"""
        self.clear_pool()


class GTLSMemoryManager:
    """GTLS专用内存管理器"""
    
    def __init__(self, device_id: int = 0):
        self.pool = GPUMemoryPool(device_id)
        self.persistent_arrays = {}
        
    def allocate_workspace(self, singleCalcPeriods: int, tSize: int, 
                          maxDuration: int, num_durations: int) -> Dict[str, cp.ndarray]:
        """
        为GTLS工作空间分配内存
        
        Args:
            singleCalcPeriods: 单次计算的周期数
            tSize: 时间序列长度
            maxDuration: 最大持续时间
            num_durations: 持续时间数量
            
        Returns:
            分配的GPU数组字典
        """
        patchedDatasSize = tSize + maxDuration
        
        arrays = {
            'periodsGPU': self.pool.allocate((singleCalcPeriods,), cp.float64),
            'durationsMaxGPU': self.pool.allocate((singleCalcPeriods,), cp.int32),
            'durationsMinGPU': self.pool.allocate((singleCalcPeriods,), cp.int32),
            'phasesGPU': self.pool.allocate((singleCalcPeriods, tSize), cp.float64),
            'sortIndexGPU': self.pool.allocate((singleCalcPeriods, tSize), cp.int32),
            'patchedDatasGPU': self.pool.allocate((singleCalcPeriods, patchedDatasSize), cp.float32),
            'patchedDysGPU': self.pool.allocate((singleCalcPeriods, patchedDatasSize), cp.float32),
            'inverseSquaredPatchedDysGPU': self.pool.allocate((singleCalcPeriods, patchedDatasSize), cp.float32),
            'fullSumGPU': self.pool.allocate((singleCalcPeriods, num_durations), cp.float32),
            'cumsumGPU': self.pool.allocate((singleCalcPeriods, patchedDatasSize), cp.float32),
            'ootrGPU': self.pool.allocate((singleCalcPeriods, num_durations, tSize), cp.float32),
            'lowestResidualsGPU': self.pool.allocate((singleCalcPeriods, num_durations, tSize), cp.float32),
            'edgeEffectCorrectionsGPU': self.pool.allocate((singleCalcPeriods,), cp.float32)
        }
        
        return arrays
    
    def deallocate_workspace(self, arrays: Dict[str, cp.ndarray]):
        """释放工作空间内存"""
        for array in arrays.values():
            self.pool.deallocate(array)
    
    def allocate_persistent(self, name: str, shape: Tuple[int, ...], dtype: cp.dtype) -> cp.ndarray:
        """分配持久化数组（在整个计算过程中保持）"""
        if name not in self.persistent_arrays:
            self.persistent_arrays[name] = self.pool.allocate(shape, dtype)
        return self.persistent_arrays[name]
    
    def clear_persistent(self):
        """清理持久化数组"""
        for array in self.persistent_arrays.values():
            self.pool.deallocate(array)
        self.persistent_arrays.clear()