"""
测试实际场景中的显存使用
"""
import cupy as cp
import numpy as np

# 获取当前 GPU 显存信息
mempool = cp.get_default_memory_pool()
print(f"初始显存使用: {mempool.used_bytes() / 1024 / 1024:.1f} MB")

# 模拟实际场景的数据规模
singleCalcPeriods = 16
tSize = 63168
maxDuration = 2000
patchedDatasSize = tSize + maxDuration
durations_count = 35

print(f"\n数据规模:")
print(f"  singleCalcPeriods: {singleCalcPeriods}")
print(f"  tSize: {tSize}")
print(f"  patchedDatasSize: {patchedDatasSize}")
print(f"  durations_count: {durations_count}")

# 分配实际场景中的所有大数组
print(f"\n分配主要数组...")

# 这些是实际 core.py 中同时存在的数组
lowestResidualsGPU = cp.empty((singleCalcPeriods, durations_count, tSize), dtype=cp.float32)
print(f"  lowestResidualsGPU: {lowestResidualsGPU.nbytes / 1024 / 1024:.1f} MB")

phasesGPU = cp.empty((singleCalcPeriods, tSize), dtype=cp.float64)
print(f"  phasesGPU: {phasesGPU.nbytes / 1024 / 1024:.1f} MB")

sortIndexGPU = cp.empty((singleCalcPeriods, tSize), dtype=cp.int32)
print(f"  sortIndexGPU: {sortIndexGPU.nbytes / 1024 / 1024:.1f} MB")

patchedDatasGPU = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  patchedDatasGPU: {patchedDatasGPU.nbytes / 1024 / 1024:.1f} MB")

patchedDysGPU = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  patchedDysGPU: {patchedDysGPU.nbytes / 1024 / 1024:.1f} MB")

inverseSquaredPatchedDysGPU = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  inverseSquaredPatchedDysGPU: {inverseSquaredPatchedDysGPU.nbytes / 1024 / 1024:.1f} MB")

cumsumGPU = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  cumsumGPU: {cumsumGPU.nbytes / 1024 / 1024:.1f} MB")

ootrGPU = cp.empty((singleCalcPeriods, durations_count, tSize), dtype=cp.float32)
print(f"  ootrGPU: {ootrGPU.nbytes / 1024 / 1024:.1f} MB")

fullSumGPU = cp.empty((singleCalcPeriods, durations_count), dtype=cp.float32)
print(f"  fullSumGPU: {fullSumGPU.nbytes / 1024 / 1024:.1f} MB")

base_error = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  base_error: {base_error.nbytes / 1024 / 1024:.1f} MB")

error_prefix_sum = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
print(f"  error_prefix_sum: {error_prefix_sum.nbytes / 1024 / 1024:.1f} MB")

print(f"\n当前显存使用: {mempool.used_bytes() / 1024 / 1024:.1f} MB")

# 填充测试数据
patchedDatasGPU[:] = cp.random.rand(singleCalcPeriods, patchedDatasSize).astype(cp.float32)
cp.cuda.Stream.null.synchronize()

print(f"\n测试 cumsum 操作...")

# 方案1: for 循环
print("方案1 - for循环:")
for i in range(singleCalcPeriods):
    cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])
cp.cuda.Stream.null.synchronize()
print(f"  成功! 显存: {mempool.used_bytes() / 1024 / 1024:.1f} MB")

# 方案2: 批量 cumsum
print("方案2 - 批量 cumsum:")
try:
    # CuPy cumsum 的临时内存
    temp_result = cp.cumsum(patchedDatasGPU, axis=1)
    cumsumGPU[:] = temp_result
    del temp_result
    cp.cuda.Stream.null.synchronize()
    print(f"  成功! 显存: {mempool.used_bytes() / 1024 / 1024:.1f} MB")
except cp.cuda.memory.OutOfMemoryError as e:
    print(f"  显存溢出!")

# 方案3: 使用 out 参数 (如果支持)
print("方案3 - 使用 out 参数:")
try:
    cp.cumsum(patchedDatasGPU, axis=1, out=cumsumGPU)
    cp.cuda.Stream.null.synchronize()
    print(f"  成功! 显存: {mempool.used_bytes() / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"  失败: {e}")

# 方案4: 分批处理
print("方案4 - 分批(batch=4):")
for start_idx in range(0, singleCalcPeriods, 4):
    end_idx = min(start_idx + 4, singleCalcPeriods)
    cumsumGPU[start_idx:end_idx] = cp.cumsum(patchedDatasGPU[start_idx:end_idx], axis=1)
cp.cuda.Stream.null.synchronize()
print(f"  成功! 显存: {mempool.used_bytes() / 1024 / 1024:.1f} MB")

print(f"\n最终显存使用: {mempool.used_bytes() / 1024 / 1024:.1f} MB")
