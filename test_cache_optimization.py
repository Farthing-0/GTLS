"""
GPU变量缓存优化 - 前后对比测试
"""
import numpy as np
import time
import sys

# 模拟 KeplerLongCurveSingleTest.py 的测试
from astropy.io import fits
from gputls import gtls

# 加载测试数据
hdulist = fits.open('/home/farthing/GTLS_workspace/CUDA_test/GTLS/tess2022302161335-s0058-0000000021132157-0247-s_lc.fits')
time_data = hdulist[1].data['TIME']
flux = hdulist[1].data['PDCSAP_FLUX']
flux_err = hdulist[1].data['PDCSAP_FLUX_ERR']
hdulist.close()

# 过滤 NaN 值
mask = ~np.isnan(flux) & ~np.isnan(time_data) & ~np.isnan(flux_err)
time_data = time_data[mask]
flux = flux[mask]
flux_err = flux_err[mask]

# 归一化
flux_median = np.median(flux)
flux = flux / flux_median
flux_err = flux_err / flux_median

print("=" * 60)
print("GPU变量缓存优化测试")
print("=" * 60)
print(f"数据点数: {len(time_data)}")
print(f"周期范围: 100-150 天")
print()

# 运行多次取平均
n_runs = 3
times_list = []

for i in range(n_runs):
    print(f"运行 {i+1}/{n_runs}...")
    start = time.perf_counter()
    
    model = gtls(t=time_data, y=flux, dy=flux_err)
    results = model.power(
        periods=np.linspace(100, 150, 500),
        bar_location=0,
        GPUDeviceID=1,
        T0_fit_margin=0.125,
        verbose=False,
        fast=True  # 快速模式
    )
    
    elapsed = time.perf_counter() - start
    times_list.append(elapsed)
    print(f"  耗时: {elapsed:.3f} 秒")

avg_time = np.mean(times_list)
std_time = np.std(times_list)

print()
print("=" * 60)
print(f"平均耗时: {avg_time:.3f} ± {std_time:.3f} 秒")
print("=" * 60)

# 验证结果正确性
periods, power = results
print(f"结果验证:")
print(f"  周期数: {len(periods)}")
print(f"  power 范围: [{np.nanmin(power):.6f}, {np.nanmax(power):.6f}]")
print(f"  NaN 数量: {np.sum(np.isnan(power))}")

# 保存基准结果用于对比
np.save('/tmp/baseline_power.npy', np.array(power))
print(f"\n基准结果已保存到 /tmp/baseline_power.npy")
