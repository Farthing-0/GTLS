"""Certificates cover every original FP32 candidate, not only the winning period."""
import cupy as cp
import numpy as np
import pytest

from gputls import core
from gputls.screening import certificates, template_certificate


@pytest.mark.parametrize('cheap_bound', [False, True])
@pytest.mark.parametrize('g', [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize('pattern', ['random', 'outlier', 'tie', 'overflow'])
def test_screen_bounds_cover_every_exact_candidate(g, pattern, cheap_bound):
    rng = np.random.default_rng(318)
    n, width = 131, 513
    size = n + width
    integers = rng.integers(-11, 17, n)
    integers[0] = 100
    inv_scale = np.float32(2.0**90 if pattern == 'overflow' else 2.0**-14)
    residual = np.resize(integers, size).astype(np.float32)*inv_scale
    dy = np.exp(rng.uniform(-3, 3, n)).astype(np.float32)
    if pattern == 'outlier':
        dy[-3] = 1e-9  # variance floor; small windows after a very large weighted sample
    weight = np.resize(np.float32(1)/np.maximum(dy*dy, np.float32(1e-10)), size)
    z = weight*residual
    x = np.linspace(-1, 1, width)
    signal = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
    if pattern == 'tie':
        signal.fill(0)
    pos, coef, meta, mass_error = certificates(signal[None], [width], abs(z).max(), weight.max(), size)
    # Every row is the same folded curve. A diagonal selector below evaluates each
    # candidate separately with the original FMA chain and XOR reduction.
    weighted = cp.asarray(np.tile(np.stack((z, weight), axis=-1), (n, 1, 1)))
    cum = cp.asarray(np.tile(np.cumsum(np.resize(integers, size)), (n, 1)))
    prefix = cp.empty((n, size, 4), dtype=cp.float64)
    module = cp.RawModule(code=core.KERNEL_SOURCE.read_text(),
                          options=(f"-DGTLS_CHEAP_SCREEN_BOUNDS={int(cheap_bound)}",))
    module.get_function('scanScreening')((n,), (512,), (weighted, prefix, np.int32(size)))
    widths = cp.asarray([width], dtype=cp.int32)
    selected = cp.asarray([0], dtype=cp.int32)
    offsets = cp.asarray([0, n], dtype=cp.int32)
    strides = cp.asarray([1], dtype=cp.int32)
    overshoot = cp.asarray([1.173], dtype=cp.float32)
    mask = cp.ones((n, 1), dtype=cp.bool_)
    lower = cp.empty((n, n), dtype=cp.float32)
    upper = cp.full(n, core.EMPTY_KEY, dtype=cp.uint64)
    screen_args = (lower, upper, prefix, cum, inv_scale, overshoot, widths, selected,
         offsets, strides, cp.asarray([g], dtype=cp.int32), mask, cp.asarray(pos), cp.asarray(coef),
         cp.asarray(meta), *map(np.float64, mass_error), np.int32(1), np.int32(1),
         np.int32(n), np.int32(size), np.float32(-1))
    module.get_function('screenCandidates')(((n+255)//256, n), (256,), screen_args)
    best = cp.full(n, core.EMPTY_KEY, dtype=cp.uint64)
    args = (best, cum, inv_scale, weighted, cp.asarray(signal), overshoot,
            widths, selected, offsets, strides, mask, np.int32(0), np.int32(1),
            np.int32(1), np.int32(n), np.int32(size), np.int32(width), np.float32(-1))
    diagonal = cp.full((n,n), cp.inf, dtype=cp.float32)
    diagonal[cp.arange(n), cp.arange(n)] = -cp.inf
    zero_upper = cp.full(n, np.uint64(0x8000000000000000), dtype=cp.uint64)
    exact = module.get_function(f'searchChi2ScreenedG{g}')
    grid = ((n*g+255)//256, n)
    exact(grid, (256,), args+(diagonal, zero_upper, np.int32(n)))
    delta = core.decode_keys(best, 0)[0].get()
    bound = lower[0].get()
    threshold = core.decode_keys(upper, 0)[0].get()[0]
    if pattern != 'overflow':
        assert np.all(bound <= delta)
    if np.isnan(threshold):
        assert np.all(upper.get() == core.EMPTY_KEY)
        assert np.all(np.isneginf(bound))
    else:
        assert threshold >= delta.min()
    best.fill(core.EMPTY_KEY)
    module.get_function(f'searchChi2G{g}')(grid, (256,), args)
    expected = best.get()
    best.fill(core.EMPTY_KEY)
    exact(grid, (256,), args+(lower, upper, np.int32(n)))
    np.testing.assert_array_equal(best.get(), expected)
    if pattern != 'overflow':
        # Seed the best exact value as a valid upper bound, so the early-bound
        # path is exercised even when the entire row fits in a single block.
        upper.fill(np.uint64(int(expected[0]) & 0xffffffff00000000))
        module.get_function('screenCandidates')(((n+255)//256, n), (256,), screen_args)
        assert np.all(lower[0].get() <= delta)
        best.fill(core.EMPTY_KEY)
        exact(grid, (256,), args+(lower, upper, np.int32(n)))
        np.testing.assert_array_equal(best.get(), expected)
    if pattern == 'tie':
        assert int(expected[0] & np.uint64(0xffffffff)) == 0


def test_sparse_certificate_reconstructs_target_within_bound():
    rng = np.random.default_rng(991)
    for width in [512, 1031, 8193]:
        x = np.linspace(-1, 1, width)
        s = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
        s += rng.uniform(-1e-5, 1e-5, width).astype(np.float32)
        pos, coef, error, _ = template_certificate(s)
        # Direct definition of the represented discrete template, independent of scans.
        j = np.arange(width, dtype=np.longdouble)
        represented = np.maximum(pos[:,None]-1-j[None], 0).T @ coef.astype(np.longdouble)
        target = np.column_stack((s, s*s))
        assert np.all(np.max(abs(represented-target), axis=0) <= error)


def _screened_curve():
    rng = np.random.default_rng(219)
    t = np.linspace(1, 51, 9000)
    y = 1+rng.normal(0, .001, t.size)
    y[(t % 2.3) < .12] -= .01
    dy = .001*np.exp(rng.normal(0, .7, t.size))
    return t, y, dy


def test_screening_batch_memory_and_full_result(monkeypatch):
    from gputls import gtls
    from test_gtls import _PeakMemory
    original_size = core.MultiGPUSearch.batch_size
    checked = []

    def check(search, n_periods):
        assert search.primary.screening
        periods = np.linspace(2, 2.5, 33)
        results = [search.chi2(periods, size, False) for size in (1, 7, 33)]
        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])
        with _PeakMemory() as hook:
            search.primary.chi2_batch(periods, len(periods), False)
        assert hook.peak <= len(periods)*search.bytes_per_period()
        checked.append(True)
        return original_size(search, n_periods)

    monkeypatch.setattr(core.MultiGPUSearch, 'batch_size', check)
    arrays = _screened_curve()
    settings = dict(periods=np.linspace(2, 2.5, 65), GPUDeviceID=0,
                    T0_fit_margin=.01, T0_refine_margin=.01)
    screened = gtls(*arrays, verbose=False).power(**settings)
    assert checked
    monkeypatch.setattr(core.MultiGPUSearch, 'batch_size', original_size)
    initialize = core.GPUSearch.__init__

    def unscreened(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.screening = False

    monkeypatch.setattr(core.GPUSearch, '__init__', unscreened)
    original = gtls(*arrays, verbose=False).power(**settings)
    for name in vars(original):
        np.testing.assert_array_equal(getattr(screened, name), getattr(original, name), err_msg=name)


def test_screening_multi_gpu_matches_single_gpu():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip('needs 2 GPUs')
    from gputls import gtls
    arrays = _screened_curve()
    settings = dict(periods=np.linspace(2, 2.5, 65), T0_fit_margin=.01, T0_refine_margin=.01)
    single = gtls(*arrays, verbose=False).power(GPUDeviceIDs=[0], **settings)
    multi = gtls(*arrays, verbose=False).power(GPUDeviceIDs=[0, 1], **settings)
    for name in vars(single):
        np.testing.assert_array_equal(getattr(single, name), getattr(multi, name), err_msg=name)
