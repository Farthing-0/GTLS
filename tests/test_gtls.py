"""End-to-end GPU tests for GTLS.

Run with ``pytest tests`` or ``python tests/test_gtls.py``.
Select the GPU with the ``GTLS_GPU_DEVICE`` environment variable (default: 0).
"""
import os
import sys
from pathlib import Path
from unittest import SkipTest

import batman
import cupy as cp
import numpy as np

from gputls import gtls
from gputls.helpers import cleaned_array

GPU_DEVICE = int(os.environ.get("GTLS_GPU_DEVICE", 0))
KEPLER_FITS = Path(__file__).resolve().parent / "data" / "kplr011403044-2009350155506_llc.fits"

TRUE_PERIOD = 7.7
TRUE_T0 = 3.3


def synthetic_light_curve(seed=1):
    """100 days of Kepler long-cadence data with an injected transit (Rp/Rs = 0.1)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(1, 101, 100 * 48)
    params = batman.TransitParams()
    params.t0 = TRUE_T0
    params.per = TRUE_PERIOD
    params.rp = 0.1
    params.a = 20
    params.inc = 90
    params.ecc = 0
    params.w = 90
    params.u = [0.4, 0.4]
    params.limb_dark = "quadratic"
    flux = batman.TransitModel(params, t).light_curve(params)
    return t, flux + rng.normal(0, 5e-4, t.size)


def test_synthetic_injection_recovery():
    t, y = synthetic_light_curve()
    result = gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE)

    assert abs(result.period - TRUE_PERIOD) < 0.01
    assert abs(result.T0 - TRUE_T0) < 0.02
    assert 0.008 < result.depth < 0.015
    assert result.SDE > 20

    # The search must be deterministic
    again = gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE)
    np.testing.assert_array_equal(result.raw_chi2, again.raw_chi2)
    assert result.SDE == again.SDE


def test_fast_mode():
    t, y = synthetic_light_curve()
    periods, power = gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE, fast=True)

    assert len(periods) == len(power)
    assert np.all(np.isfinite(power))
    assert abs(periods[np.argmax(power)] - TRUE_PERIOD) < 0.05


def test_kepler_custom_periods():
    """Real Kepler data searched with user periods longer than the time span (no NaN chi2)."""
    from astropy.io import fits

    with fits.open(KEPLER_FITS) as hdul:
        t, y, dy = cleaned_array(
            hdul[1].data["TIME"], hdul[1].data["PDCSAP_FLUX"], hdul[1].data["PDCSAP_FLUX_ERR"])
    mean_flux = np.mean(y)
    periods = np.linspace(100, 150, 500)

    result = gtls(t, y / mean_flux, dy / mean_flux, verbose=False).power(
        periods=periods, GPUDeviceID=GPU_DEVICE)

    assert len(result.raw_chi2) == len(periods)
    assert np.all(np.isfinite(result.raw_chi2))
    assert periods.min() <= result.period <= periods.max()
    # Every period exceeds the 89 d baseline, so each fold holds every event once and the
    # deepest single event gives the same lowest chi2 at all periods: the periodogram is flat
    # (to float32 rounding) and the SDE is undefined.
    np.testing.assert_allclose(result.raw_chi2, result.raw_chi2[0], rtol=1e-6)


def test_gpu_memory_released():
    """Repeated searches must not leave GPU memory allocated."""
    t, y = synthetic_light_curve()
    for _ in range(2):
        gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE)
        with cp.cuda.Device(GPU_DEVICE):
            assert cp.get_default_memory_pool().total_bytes() == 0


def test_multi_gpu_matches_single_gpu():
    """Splitting the search across GPUs must not change the result."""
    if cp.cuda.runtime.getDeviceCount() < 2:
        raise SkipTest("needs 2 GPUs")
    t, y = synthetic_light_curve()
    single = gtls(t, y, verbose=False).power(GPUDeviceID=0)
    multi = gtls(t, y, verbose=False).power(GPUDeviceIDs=[0, 1])

    np.testing.assert_array_equal(single.raw_chi2, multi.raw_chi2)
    np.testing.assert_array_equal(single.chi2, multi.chi2)
    assert single.period == multi.period
    assert single.T0 == multi.T0
    assert single.SDE == multi.SDE

    _, power_single = gtls(t, y, verbose=False).power(GPUDeviceID=0, fast=True)
    _, power_multi = gtls(t, y, verbose=False).power(GPUDeviceIDs=[0, 1], fast=True)
    np.testing.assert_array_equal(power_single, power_multi)


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    failed = skipped = 0
    for test in tests:
        try:
            test()
            print("PASSED", test.__name__)
        except SkipTest as reason:
            skipped += 1
            print("SKIPPED", test.__name__, "-", reason)
        except Exception as error:
            failed += 1
            print("FAILED", test.__name__, "-", repr(error))
    print("%d passed, %d skipped, %d failed" % (len(tests) - failed - skipped, skipped, failed))
    sys.exit(1 if failed else 0)
