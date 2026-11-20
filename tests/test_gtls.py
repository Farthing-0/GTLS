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
import pytest

from gputls import core, gtls
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
    assert np.isfinite(result.snr) and np.isfinite(result.snrPink)
    assert not {'snrFit', 'snrFitPink'} & vars(result).keys()

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


def test_tls2_template_backend_default_and_exact_override(monkeypatch):
    monkeypatch.delenv("TLS_BACKEND", raising=False)
    monkeypatch.delenv("TLS_T0_COARSEN", raising=False)
    t, y = synthetic_light_curve()
    periods = np.linspace(6.8, 8.6, 25)
    model = gtls(t, y, verbose=False)
    default = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True)
    assert model.backend == "fused-pl"
    explicit = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True, backend="fused-pl")
    np.testing.assert_array_equal(default, explicit)

    monkeypatch.setenv("TLS_BACKEND", "fused")
    exact = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True)
    assert model.backend == "fused"
    assert np.max(np.abs(default[1] - exact[1])) > 1e-5
    overridden = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True, backend="fused-pl")
    np.testing.assert_array_equal(default, overridden)
    with pytest.raises(ValueError, match="Unknown backend"):
        model.power(backend="unknown")


def test_tls2_phase_switch_and_one_scan_per_period(monkeypatch):
    monkeypatch.delenv("TLS_BACKEND", raising=False)
    monkeypatch.delenv("TLS_T0_COARSEN", raising=False)
    t, y = synthetic_light_curve()
    periods = np.linspace(6.8, 8.6, 25)
    visits = []
    best_keys = core.GPUSearch.best_keys

    def track(search, trial_periods, *args, **kwargs):
        visits.extend(trial_periods)
        return best_keys(search, trial_periods, *args, **kwargs)

    monkeypatch.setattr(core.GPUSearch, "best_keys", track)
    model = gtls(t, y, verbose=False)
    result = model.power(periods=periods, GPUDeviceID=GPU_DEVICE)
    np.testing.assert_array_equal(np.sort(visits), periods)
    assert model.T0_coarsen == 3
    assert result.period == result.periods[np.argmax(result.power)]
    assert result.possiblePeriods == result.possiblePeriodsIndices == []
    np.testing.assert_array_equal(result.raw_power, result.power)

    monkeypatch.setenv("TLS_T0_COARSEN", "1")
    off = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True)
    assert model.T0_coarsen == 1
    explicit = model.power(periods=periods, GPUDeviceID=GPU_DEVICE, fast=True, T0_coarsen=3)
    np.testing.assert_array_equal(result.power, explicit[1])
    assert np.any(result.power != off[1])
    with pytest.raises(ValueError, match="T0_coarsen"):
        model.power(T0_coarsen=2)


def test_duration_scout_switch(monkeypatch):
    monkeypatch.delenv("TLS_BACKEND", raising=False)
    monkeypatch.delenv("TLS_SCOUT_EVERY", raising=False)
    t, y = synthetic_light_curve()
    model = gtls(t, y, verbose=False)
    options = dict(periods=np.linspace(1.04, 1.08, 12), GPUDeviceID=GPU_DEVICE, fast=True)
    visits = []
    search_scouts = core.GPUSearch._search_duration_scouts

    def track(search, *args):
        visits.append(search.scout_every)
        return search_scouts(search, *args)

    monkeypatch.setattr(core.GPUSearch, "_search_duration_scouts", track)
    default = model.power(**options)
    assert model.scout_every == 4
    assert visits and set(visits) == {4}
    visits.clear()

    model.power(**options, backend="fused")
    assert model.scout_every == 0

    monkeypatch.setenv("TLS_SCOUT_EVERY", "0")
    disabled = model.power(**options)
    assert model.scout_every == 0
    np.testing.assert_array_equal(disabled, model.power(**options, scout_every=1))
    assert not visits
    np.testing.assert_array_equal(default, model.power(**options, scout_every=4))
    with pytest.raises(ValueError, match="scout_every"):
        model.power(scout_every=-1)


def test_duration_scout_mask_uses_allowed_ordinals():
    with cp.cuda.Device(GPU_DEVICE):
        mask = cp.asarray([[True, False, True, True, True, False, True, True],
                           [False, True, False, True, False, True, False, True],
                           [False] * 8])
        states = cp.empty(mask.shape, dtype=cp.int32)
        kernel = core.load_module().get_function("prepareDurationScouts")
        kernel((1,), (128,), (states, mask, np.int32(3), np.int32(8), np.int32(4)))
        np.testing.assert_array_equal(states.get(), [[-1, -3, -2, -2, -2, -3, -1, -1],
                                                     [-3, -1, -3, -1, -3, -1, -3, -1],
                                                     [-3] * 8])


@pytest.mark.parametrize("start", [91, 92])
def test_coarse_phase_kernel_refines_between_coarse_starts(start):
    with cp.cuda.Device(GPU_DEVICE):
        n, width = 240, 70
        template = (0.5 * np.sin(np.linspace(0, np.pi, width))**2).astype(np.float32)
        residual = np.zeros(n + width, dtype=np.float32)
        residual[start:start + width] = 0.02 * template
        cumulative = np.cumsum(np.rint(residual * 2**25), dtype=np.int64)
        best = cp.full(1, core.EMPTY_KEY, dtype=cp.uint64)
        null = np.uint64(0)
        kernel = core.load_module().get_function("searchCoarsePhase")
        kernel((1, 1), (128,), (
            best, cp.asarray([[True]]), cp.asarray(cumulative), null, null, null, null, null,
            cp.asarray([width], dtype=cp.int32), cp.asarray([0.5 / template.mean()], dtype=cp.float32),
            np.int32(1), np.int32(n), np.int32(n + width), np.int32(0), np.int32(0),
            np.float32(2**-25), np.float32(1e-5), np.float64(0), np.float64(0),
            cp.asarray([0], dtype=cp.int32), cp.asarray([1], dtype=cp.int32), cp.asarray(residual),
            cp.asarray(template), np.int32(width), np.int32(1), np.float32(1)))
        _, locations = core.decode_keys(best, 0.)
        assert int(locations[0]) == start


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


class _PeakMemory(cp.cuda.MemoryHook):
    """Peak of the memory held by live CuPy arrays while active."""

    name = "PeakMemory"

    def __init__(self):
        self.live = self.peak = 0
        self.sizes = {}

    def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
        self.sizes[mem_ptr] = mem_size
        self.live += mem_size
        self.peak = max(self.peak, self.live)

    def free_postprocess(self, device_id, mem_size, mem_ptr, pmem_id):
        self.live -= self.sizes.pop(mem_ptr, 0)


def test_batch_memory_model():
    """One batch must stay within bytes_per_period per period, and batches within the grid limit."""
    checked = []
    batch_size = core.MultiGPUSearch.batch_size

    def checked_batch_size(search, n_periods):
        size = batch_size(search, n_periods)
        assert 1 <= size <= min(core.MAX_BATCH, n_periods)
        periods = np.geomspace(0.5, 50, 64)
        search.primary.chi2_batch(periods, len(periods), refine=False)  # compile outside the hook
        hook = _PeakMemory()
        with cp.cuda.Device(GPU_DEVICE), hook:
            search.primary.chi2_batch(periods, len(periods), refine=False)
        assert hook.peak <= len(periods) * search.bytes_per_period()
        checked.append(size)
        return size

    core.MultiGPUSearch.batch_size = checked_batch_size
    try:
        t, y = synthetic_light_curve()
        gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE, fast=True)
    finally:
        core.MultiGPUSearch.batch_size = batch_size
    assert checked


def test_batch_size_does_not_change_chi2():
    """Each period is searched over its own durations, whatever periods share its batch."""
    chi2 = {}
    batch_size = core.MultiGPUSearch.batch_size

    def all_sizes(search, n_periods):
        periods = np.geomspace(0.5, 50, 500)
        for size in (1, 7, 64, 500):
            chi2[size] = search.chi2(periods, size, refine=False)
        return batch_size(search, n_periods)

    core.MultiGPUSearch.batch_size = all_sizes
    try:
        t, y = synthetic_light_curve()
        gtls(t, y, verbose=False).power(GPUDeviceID=GPU_DEVICE, fast=True)
    finally:
        core.MultiGPUSearch.batch_size = batch_size
    for size in (7, 64, 500):
        np.testing.assert_array_equal(chi2[1], chi2[size])


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
