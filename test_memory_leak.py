#!/usr/bin/env python
"""
Test script to verify GPU memory leak issue.
Runs multiple calculations and monitors GPU memory usage.
"""
import numpy as np
from astropy.io import fits
import sys
sys.path.insert(0, 'src')
from gputls import gtls
import pynvml

def get_gpu_memory_info(device_id=0):
    """Get GPU memory usage in MB."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_mb = info.used / 1024 / 1024
    free_mb = info.free / 1024 / 1024
    total_mb = info.total / 1024 / 1024
    pynvml.nvmlShutdown()
    return used_mb, free_mb, total_mb

def print_gpu_memory(label, device_ids=[0, 1]):
    """Print GPU memory for multiple devices."""
    print(f"\n{label}")
    for gpu_id in device_ids:
        try:
            used, free, total = get_gpu_memory_info(gpu_id)
            print(f"  GPU {gpu_id}: {used:.0f} MB used, {free:.0f} MB free (total: {total:.0f} MB)")
        except Exception as e:
            print(f"  GPU {gpu_id}: Error getting memory info: {e}")

def load_test_data():
    """Load test light curve data."""
    fitsFile = fits.open("tess2022302161335-s0058-0000000021132157-0247-s_lc.fits")
    t = fitsFile[1].data['TIME']
    y = fitsFile[1].data['PDCSAP_FLUX']
    dy = fitsFile[1].data['PDCSAP_FLUX_ERR']
    
    mask = ~np.isnan(t) & ~np.isnan(y) & ~np.isnan(dy)
    t = t[mask]
    y = y[mask]
    dy = dy[mask]
    
    y = y / np.median(y)
    dy = dy / np.median(y)
    
    return t, y, dy

def run_single_gpu_test(t, y, dy, iteration):
    """Run a single GPU test."""
    print(f"\n--- Single GPU Test #{iteration} ---")
    print_gpu_memory("Before computation:")
    
    model = gtls(t, y, dy, verbose=False)
    results = model.power()
    
    print_gpu_memory("After computation:")
    print(f"  Result: period={results.period:.4f}, SDE={results.SDE:.2f}")
    
    return results

def run_dual_gpu_test(t, y, dy, iteration):
    """Run a dual GPU test."""
    print(f"\n--- Dual GPU Test #{iteration} ---")
    print_gpu_memory("Before computation:")
    
    model = gtls(t, y, dy, verbose=False)
    results = model.power(GPUDeviceIDs=[0, 1])
    
    print_gpu_memory("After computation:")
    print(f"  Result: period={results.period:.4f}, SDE={results.SDE:.2f}")
    
    return results

def main():
    print("="*70)
    print("GPU Memory Leak Test")
    print("="*70)
    
    print_gpu_memory("Initial GPU memory state:")
    
    # Load data
    print("\nLoading test data...")
    t, y, dy = load_test_data()
    print(f"Data points: {len(t)}")
    
    print_gpu_memory("After loading data:")
    
    num_iterations = 5
    
    # Test single GPU multiple times
    print("\n" + "="*70)
    print("SINGLE GPU TESTS")
    print("="*70)
    
    for i in range(1, num_iterations + 1):
        run_single_gpu_test(t, y, dy, i)
    
    print_gpu_memory("\nFinal state after single GPU tests:")
    
    # Force cleanup
    print("\n--- Forcing GPU memory cleanup ---")
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print_gpu_memory("After forced cleanup:")
    
    # Test dual GPU multiple times
    print("\n" + "="*70)
    print("DUAL GPU TESTS")
    print("="*70)
    
    for i in range(1, num_iterations + 1):
        run_dual_gpu_test(t, y, dy, i)
    
    print_gpu_memory("\nFinal state after dual GPU tests:")
    
    # Force cleanup again
    print("\n--- Forcing GPU memory cleanup ---")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print_gpu_memory("After forced cleanup:")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
