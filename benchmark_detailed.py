"""
core.py 深度性能分析
测量各个操作的实际耗时
"""
import cupy as cp
import numpy as np
import time

# 模拟实际数据规模
singleCalcPeriods = 16
tSize = 63168
maxDuration = 2000
patchedDatasSize = tSize + maxDuration
durations_count = 35

print("=" * 60)
print("GTLS core.py 性能瓶颈分析")
print("=" * 60)
print(f"数据规模: singleCalcPeriods={singleCalcPeriods}, tSize={tSize}")
print()

# 创建测试数据
phasesGPU = cp.random.rand(singleCalcPeriods, tSize).astype(cp.float64)
patchedDatasGPU = cp.random.rand(singleCalcPeriods, patchedDatasSize).astype(cp.float32)
durationBoolArrayGPU = cp.random.rand(500, durations_count) > 0.5  # 500 periods
cumsumGPU = cp.empty((singleCalcPeriods, patchedDatasSize), dtype=cp.float32)
sortIndexGPU = cp.empty((singleCalcPeriods, tSize), dtype=cp.int32)

# 预热
cp.cuda.Stream.null.synchronize()

n_iter = 20
results = {}

# ============================================================
# 1. argsort 操作分析
# ============================================================
print("1. argsort 操作:")

# 方案A: 原始分批 for 循环
start = time.perf_counter()
for _ in range(n_iter):
    i_max = 10
    for i in range(1, i_max + 1):
        start_slice = int((i-1)*singleCalcPeriods/i_max)
        end_slice = int(i*singleCalcPeriods/i_max)
        sortIndexGPU[start_slice:end_slice] = phasesGPU[start_slice:end_slice].argsort()
cp.cuda.Stream.null.synchronize()
argsort_loop_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   原始分批循环(10批): {argsort_loop_time:.3f} ms")

# 方案B: 直接批量 argsort
start = time.perf_counter()
for _ in range(n_iter):
    sortIndexGPU = phasesGPU.argsort(axis=1)
cp.cuda.Stream.null.synchronize()
argsort_batch_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   批量 argsort:       {argsort_batch_time:.3f} ms")
print(f"   潜在加速: {argsort_loop_time/argsort_batch_time:.2f}x")

results['argsort'] = {'loop': argsort_loop_time, 'batch': argsort_batch_time}

# ============================================================
# 2. cumsum 操作分析 (已知问题)
# ============================================================
print("\n2. cumsum 操作:")

# 方案A: for 循环
start = time.perf_counter()
for _ in range(n_iter):
    for i in range(singleCalcPeriods):
        cumsumGPU[i] = cp.cumsum(patchedDatasGPU[i])
cp.cuda.Stream.null.synchronize()
cumsum_loop_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   for循环:            {cumsum_loop_time:.3f} ms")

# 方案B: 批量
start = time.perf_counter()
for _ in range(n_iter):
    cumsumGPU = cp.cumsum(patchedDatasGPU, axis=1)
cp.cuda.Stream.null.synchronize()
cumsum_batch_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   批量 cumsum:        {cumsum_batch_time:.3f} ms")
print(f"   潜在加速: {cumsum_loop_time/cumsum_batch_time:.2f}x")

results['cumsum'] = {'loop': cumsum_loop_time, 'batch': cumsum_batch_time}

# ============================================================
# 3. logical_or 循环分析
# ============================================================
print("\n3. logical_or 操作 (合并 duration bool):")

start_idx = 0
end_idx = singleCalcPeriods

# 方案A: for 循环
start = time.perf_counter()
for _ in range(n_iter):
    temp_bool = durationBoolArrayGPU[start_idx]
    for i in range(start_idx + 1, end_idx):
        temp_bool = cp.logical_or(temp_bool, durationBoolArrayGPU[i])
cp.cuda.Stream.null.synchronize()
logical_or_loop_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   for循环:            {logical_or_loop_time:.3f} ms")

# 方案B: 使用 any (等价于 reduce OR)
start = time.perf_counter()
for _ in range(n_iter):
    temp_bool = cp.any(durationBoolArrayGPU[start_idx:end_idx], axis=0)
cp.cuda.Stream.null.synchronize()
logical_or_any_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   cp.any():           {logical_or_any_time:.3f} ms")
print(f"   潜在加速: {logical_or_loop_time/logical_or_any_time:.2f}x")

results['logical_or'] = {'loop': logical_or_loop_time, 'batch': logical_or_any_time}

# ============================================================
# 4. GPU->CPU 数据传输 (.get())
# ============================================================
print("\n4. GPU->CPU 数据传输 (.get()):")

test_array = cp.random.rand(durations_count).astype(cp.bool_)
start = time.perf_counter()
for _ in range(n_iter * 10):
    _ = test_array.get()
cp.cuda.Stream.null.synchronize()
get_small_time = (time.perf_counter() - start) / (n_iter * 10) * 1000
print(f"   小数组 ({durations_count} 元素): {get_small_time:.4f} ms")

test_array_large = cp.random.rand(tSize).astype(cp.float32)
start = time.perf_counter()
for _ in range(n_iter):
    _ = test_array_large.get()
cp.cuda.Stream.null.synchronize()
get_large_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   大数组 ({tSize} 元素): {get_large_time:.3f} ms")

results['get'] = {'small': get_small_time, 'large': get_large_time}

# ============================================================
# 5. 重复创建 GPU 变量开销
# ============================================================
print("\n5. 重复创建 GPU 变量开销:")

t = np.random.rand(tSize).astype(np.float64)

start = time.perf_counter()
for _ in range(n_iter):
    tGPU = cp.asarray(t).astype(cp.float64)
    tSizeGPU = cp.asarray(np.array([tSize])).astype(cp.int32)
    tLengthGPU = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)
cp.cuda.Stream.null.synchronize()
create_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   每次循环创建 tGPU 等: {create_time:.3f} ms")

# 预先创建
tGPU_cached = cp.asarray(t).astype(cp.float64)
tSizeGPU_cached = cp.asarray(np.array([tSize])).astype(cp.int32)
tLengthGPU_cached = cp.asarray(np.array([max(t) - min(t)])).astype(cp.float32)

start = time.perf_counter()
for _ in range(n_iter):
    # 直接使用缓存
    _ = tGPU_cached
    _ = tSizeGPU_cached  
    _ = tLengthGPU_cached
cp.cuda.Stream.null.synchronize()
cached_time = (time.perf_counter() - start) / n_iter * 1000
print(f"   使用缓存变量:       {cached_time:.4f} ms")
print(f"   潜在加速: {create_time/max(cached_time, 0.001):.0f}x (每次循环)")

results['create_vars'] = {'create': create_time, 'cached': cached_time}

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("优化建议总结 (按收益排序)")
print("=" * 60)

optimizations = [
    ("argsort 分批→批量", results['argsort']['loop'] - results['argsort']['batch'], 
     results['argsort']['loop'] / results['argsort']['batch'], "高"),
    ("cumsum for循环→批量", results['cumsum']['loop'] - results['cumsum']['batch'],
     results['cumsum']['loop'] / results['cumsum']['batch'], "中"),
    ("logical_or 循环→any", results['logical_or']['loop'] - results['logical_or']['batch'],
     results['logical_or']['loop'] / results['logical_or']['batch'], "低"),
    ("GPU变量缓存", results['create_vars']['create'], 
     results['create_vars']['create'] / max(results['create_vars']['cached'], 0.001), "中"),
]

optimizations.sort(key=lambda x: x[1], reverse=True)

for name, save_ms, speedup, difficulty in optimizations:
    print(f"\n{name}:")
    print(f"   节省时间: {save_ms:.3f} ms/批次")
    print(f"   加速比: {speedup:.2f}x")
    print(f"   实现难度: {difficulty}")

# 估算整体影响 (32 批次)
total_batches = 32
total_save = sum([opt[1] for opt in optimizations]) * total_batches
print(f"\n估算整体优化 ({total_batches}批次):")
print(f"   累计节省: ~{total_save:.1f} ms")
print(f"   原始搜索约 1.2s = 1200ms")
print(f"   预计提升: ~{total_save/1200*100:.1f}%")
