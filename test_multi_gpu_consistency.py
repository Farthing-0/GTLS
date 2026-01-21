#!/usr/bin/env python
"""
Test script to verify that single-GPU and multi-GPU results are consistent.
Compares chi2 values between single GPU and dual GPU execution.
"""
import numpy as np
from astropy.io import fits

# Import gputls
import sys
sys.path.insert(0, 'src')
from gputls import gtls

def load_test_data():
    """Load test light curve data."""
    fitsFile = fits.open("tess2022302161335-s0058-0000000021132157-0247-s_lc.fits")
    t = fitsFile[1].data['TIME']
    y = fitsFile[1].data['PDCSAP_FLUX']
    dy = fitsFile[1].data['PDCSAP_FLUX_ERR']
    
    # Clean NaN values
    mask = ~np.isnan(t) & ~np.isnan(y) & ~np.isnan(dy)
    t = t[mask]
    y = y[mask]
    dy = dy[mask]
    
    # Normalize flux
    y = y / np.median(y)
    dy = dy / np.median(y)
    
    return t, y, dy

def run_single_gpu(t, y, dy):
    """Run GTLS with single GPU."""
    print("\n" + "="*60)
    print("Running with SINGLE GPU (GPU 0)")
    print("="*60)
    
    model = gtls(t, y, dy, verbose=True)
    results = model.power()
    
    return results

def run_dual_gpu(t, y, dy):
    """Run GTLS with dual GPU."""
    print("\n" + "="*60)
    print("Running with DUAL GPU (GPU 0 and 1)")
    print("="*60)
    
    model = gtls(t, y, dy, verbose=True)
    results = model.power(GPUDeviceIDs=[0, 1])
    
    return results

def compare_results(single_results, dual_results):
    """Compare chi2 results between single and dual GPU runs."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    chi2_single = single_results.chi2
    chi2_dual = dual_results.chi2
    
    # Basic shape check
    print(f"\nSingle GPU chi2 shape: {chi2_single.shape}")
    print(f"Dual GPU chi2 shape:   {chi2_dual.shape}")
    
    if chi2_single.shape != chi2_dual.shape:
        print("\n❌ FAILED: Shapes do not match!")
        return False
    
    # Check for exact equality
    exact_match = np.array_equal(chi2_single, chi2_dual)
    print(f"\nExact match: {exact_match}")
    
    # Check for NaN positions
    nan_single = np.isnan(chi2_single)
    nan_dual = np.isnan(chi2_dual)
    nan_match = np.array_equal(nan_single, nan_dual)
    print(f"NaN positions match: {nan_match}")
    
    # Compare non-NaN values
    valid_mask = ~nan_single & ~nan_dual
    valid_count = np.sum(valid_mask)
    print(f"\nValid (non-NaN) values: {valid_count} / {len(chi2_single)}")
    
    if valid_count > 0:
        chi2_single_valid = chi2_single[valid_mask]
        chi2_dual_valid = chi2_dual[valid_mask]
        
        # Absolute difference
        abs_diff = np.abs(chi2_single_valid - chi2_dual_valid)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        print(f"\nAbsolute difference:")
        print(f"  Max:  {max_abs_diff:.2e}")
        print(f"  Mean: {mean_abs_diff:.2e}")
        
        # Relative difference (avoid division by zero)
        nonzero_mask = chi2_single_valid != 0
        if np.any(nonzero_mask):
            rel_diff = abs_diff[nonzero_mask] / np.abs(chi2_single_valid[nonzero_mask])
            max_rel_diff = np.max(rel_diff)
            mean_rel_diff = np.mean(rel_diff)
            
            print(f"\nRelative difference:")
            print(f"  Max:  {max_rel_diff:.2e}")
            print(f"  Mean: {mean_rel_diff:.2e}")
        
        # Check if values are close enough (floating point tolerance)
        # Note: GPU floating point operations may have small differences due to
        # different execution order, so we use a slightly relaxed tolerance
        is_close = np.allclose(chi2_single_valid, chi2_dual_valid, rtol=1e-4, atol=1e-7)
        print(f"\nValues are close (rtol=1e-4, atol=1e-7): {is_close}")
        
        # Count exact matches
        exact_matches = np.sum(chi2_single_valid == chi2_dual_valid)
        print(f"Exact matches: {exact_matches} / {valid_count} ({100*exact_matches/valid_count:.2f}%)")
        
        # Show some examples of differences if any
        if not exact_match and max_abs_diff > 0:
            diff_indices = np.where(abs_diff > 0)[0][:5]  # First 5 differences
            print(f"\nFirst few differences (index, single, dual, diff):")
            for idx in diff_indices:
                orig_idx = np.where(valid_mask)[0][idx]
                print(f"  [{orig_idx}]: {chi2_single_valid[idx]:.10f} vs {chi2_dual_valid[idx]:.10f} (diff: {abs_diff[idx]:.2e})")
    
    # Final verdict
    print("\n" + "="*60)
    if exact_match:
        print("✅ PASSED: Single GPU and Dual GPU results are IDENTICAL")
        return True
    elif np.allclose(chi2_single[valid_mask], chi2_dual[valid_mask], rtol=1e-4, atol=1e-7):
        print("✅ PASSED (with tolerance): Results are numerically close")
        print("   (Small differences are expected due to GPU floating point precision)")
        return True
    else:
        print("❌ FAILED: Results differ significantly")
        return False

def main():
    print("="*60)
    print("Multi-GPU Consistency Test")
    print("="*60)
    
    # Load data
    print("\nLoading test data...")
    t, y, dy = load_test_data()
    print(f"Data points: {len(t)}")
    
    # Run single GPU
    single_results = run_single_gpu(t, y, dy)
    
    # Run dual GPU
    dual_results = run_dual_gpu(t, y, dy)
    
    # Compare results
    success = compare_results(single_results, dual_results)
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    print("="*60)
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
