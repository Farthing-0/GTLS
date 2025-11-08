import numpy as np
import sys
sys.path.insert(0, 'src')
from gputls import gtls

# 创建简单的测试数据
np.random.seed(42)
t = np.linspace(0, 27, 1000)
y = np.ones_like(t) + np.random.normal(0, 0.01, len(t))
dy = np.ones_like(t) * 0.01

# 测试自定义periods
print("Testing custom periods...")
GTLSmodel = gtls(t=t, y=y, dy=dy)

# 测试500个periods的情况
periods = np.linspace(100, 150, 500)
print(f"Total periods: {len(periods)}")

try:
    gtlsResult = GTLSmodel.power(
        periods=periods,
        bar_location=0,
        GPUDeviceID=0,
        T0_fit_margin=0.125,
        verbose=True
    )
    
    raw_chi2 = gtlsResult.raw_chi2
    nan_count = np.sum(np.isnan(raw_chi2))
    valid_count = np.sum(~np.isnan(raw_chi2))
    
    print(f"\nResults:")
    print(f"Total chi2 values: {len(raw_chi2)}")
    print(f"Valid values: {valid_count}")
    print(f"NaN values: {nan_count}")
    
    if nan_count > 0:
        print(f"\nNaN positions: {np.where(np.isnan(raw_chi2))[0]}")
        print(f"First few valid values: {raw_chi2[~np.isnan(raw_chi2)][:10]}")
        print(f"Last few values: {raw_chi2[-10:]}")
    else:
        print("\n✓ No NaN values found! Fix successful!")
        print(f"Chi2 range: [{np.min(raw_chi2):.6f}, {np.max(raw_chi2):.6f}]")
        print(f"Best period: {gtlsResult.period:.3f} days")
        
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
