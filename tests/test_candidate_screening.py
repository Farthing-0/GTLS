"""Certificates cover every original FP32 candidate, not only the winning period."""
import cupy as cp
import numpy as np
import pytest

from gputls import core
from gputls.screening import (certificates, template_certificate, SCREEN_TOLERANCES,
                             uniform_sparse_certificate, sparse_certificate,
                             UNIFORM_PREFIX_STRIDE, GENERAL_PREFIX_STRIDE, prefix_quantization)


def test_screen_schedule_covers_candidates_once_in_time_order():
    offsets = np.array([0, 17, 82, 209, 466], dtype=np.int64)
    strides = np.array([1, 7, 33, 81], dtype=np.int32)
    blocks = core.screen_block_map(offsets, strides)
    starts = (blocks[:, 1]-offsets[blocks[:, 0]])*strides[blocks[:, 0]]
    assert np.all(np.diff(starts) >= 0)
    items = np.concatenate([np.arange(first, min(first+core.SCREEN_THREADS, offsets[d+1]))
                            for d, first in blocks])
    np.testing.assert_array_equal(np.sort(items), np.arange(offsets[-1]))
    assert blocks.dtype == np.int32


@pytest.mark.parametrize('size', [1, 15, 16, 17, 127, 128, 129, 511, 512, 513, 1025, 706406])
def test_chunked_prefixes_fit_high_precision_error_budget(size):
    rng = np.random.default_rng(734)
    weight = np.exp(rng.uniform(-10, 10, size)).astype(np.float32)
    z = (weight*rng.normal(0, .01, size)).astype(np.float32)
    weighted = cp.asarray(np.column_stack((z, weight))[None])
    prefix = cp.empty(24*size, dtype=cp.uint8)
    core.load_module().get_function('scanScreening')(
        (1,), (core.SCREEN_SCAN_THREADS,), (weighted, prefix, np.int32(size)))
    z_long, w_long = z.astype(np.longdouble), weight.astype(np.longdouble)
    expected = np.stack((
        np.column_stack((np.cumsum(np.cumsum(z_long)), np.cumsum(np.cumsum(w_long)))),
        np.column_stack((np.cumsum(abs(z_long)), np.cumsum(w_long)))))
    steps = 2*((size+511)//512+48)
    gamma = steps*2.0**-53/(1-steps*2.0**-53)
    maxima = np.array([abs(z_long).max(), w_long.max()])
    budgets = 4*gamma*np.array([size*size, size])[:, None]*maxima
    actual_d = prefix[:16*size].view(cp.float64).reshape(size, 2).get().astype(np.longdouble)
    mass_bits = prefix[16*size:].view(cp.uint32).reshape(size, 2).get()
    actual_mass = (mass_bits.astype(np.uint64) << 32).view(np.float64).astype(np.longdouble)
    assert np.all(np.max(abs(actual_d-expected[0]), axis=0) <= budgets[0])
    storage_radius = 2.0**-19*abs(actual_mass)+2.0**-1020
    assert np.all(abs(actual_mass-expected[1]) <= budgets[1]+storage_radius)
    uniform = cp.empty(12*size, dtype=cp.uint8)
    core.load_module().get_function('scanScreeningUniform')(
        (1,), (core.SCREEN_SCAN_THREADS,), (weighted, uniform, np.int32(size)))
    assert uniform[:8*size].view(cp.float64).get().tobytes() == actual_d[:, 0].astype(np.float64).tobytes()
    np.testing.assert_array_equal(uniform[8*size:].view(cp.uint32).get(), mass_bits[:, 0])
    packed_prefix = cp.empty_like(uniform)
    core.load_module().get_function('scanScreeningUniformPacked')(
        (1,), (core.SCREEN_SCAN_THREADS,), (cp.asarray(z), packed_prefix, np.int32(size)))
    assert packed_prefix.get().tobytes() == uniform.get().tobytes()
    stride = UNIFORM_PREFIX_STRIDE
    padded_size = (size+stride-1)//stride*stride
    prefix_size = padded_size//stride+1
    sparse = cp.empty(12*prefix_size, dtype=cp.uint8)
    core.load_module().get_function('scanScreeningUniformSparse')(
        (1,), (core.SCREEN_SCAN_THREADS,), (cp.asarray(z), sparse, np.int32(size)))
    sparse_d = sparse[:8*prefix_size].view(cp.float64).get()
    sparse_mass = sparse[8*prefix_size:].view(cp.uint32).get()
    indices = np.arange(stride, size+1, stride)-1
    assert sparse_d[1:len(indices)+1].tobytes() == actual_d[indices, 0].astype(np.float64).tobytes()
    np.testing.assert_array_equal(sparse_mass[1:len(indices)+1], mass_bits[indices, 0])
    padded = np.zeros(padded_size, np.longdouble)
    padded[:size] = z
    exact_d = np.r_[np.longdouble(0), np.cumsum(np.cumsum(padded))]
    steps = 2*((padded_size+511)//512+48)
    gamma = steps*2.**-53/(1-steps*2.**-53)
    prefix_budget = 4*gamma*padded_size**2*maxima[0]
    at = np.arange(padded_size+1)
    cells, fraction = at//stride, (at%stride).astype(np.longdouble)/stride
    interpolated = (1-fraction)*sparse_d[cells] + fraction*sparse_d[np.minimum(cells+1, prefix_size-1)]
    assert np.all(abs(interpolated-exact_d) <= prefix_budget+(stride*stride/8)*maxima[0])


@pytest.mark.parametrize('size', [1, 7, 8, 9, 127, 128, 129, 511, 512, 513, 1025, 706406])
def test_general_sparse_prefix_grid_bits_and_interpolation_bound(size):
    stride = GENERAL_PREFIX_STRIDE
    rng = np.random.default_rng(size)
    w = np.exp(rng.uniform(-10, 10, size)).astype(np.float32)
    z = (w*rng.normal(0, .01, size)).astype(np.float32)
    inputs = np.column_stack((z, w))
    weighted = cp.asarray(inputs)
    padded_size = (size+stride-1)//stride*stride
    prefix_size = padded_size//stride+1
    full = cp.empty(24*size, cp.uint8)
    sparse = cp.empty(24*prefix_size, cp.uint8)
    module = core.load_module()
    module.get_function('scanScreening')((1,), (core.SCREEN_SCAN_THREADS,),
                                         (weighted, full, np.int32(size)))
    module.get_function('scanScreeningSparse')((1,), (core.SCREEN_SCAN_THREADS,),
                                               (weighted, sparse, np.int32(size)))
    d = sparse[:16*prefix_size].view(cp.float64).reshape(prefix_size, 2).get()
    mass_bits = sparse[16*prefix_size:].view(cp.uint32).reshape(prefix_size, 2).get()
    mass = (mass_bits.astype(np.uint64) << 32).view(np.float64).astype(np.longdouble)
    indices = np.arange(stride, size+1, stride)-1
    assert d[1:len(indices)+1].tobytes() == full[:16*size].view(cp.float64).reshape(size, 2).get()[indices].tobytes()
    np.testing.assert_array_equal(mass_bits[1:len(indices)+1], full[16*size:].view(cp.uint32).reshape(size, 2).get()[indices])
    padded = np.zeros((padded_size, 2), np.longdouble)
    padded[:size] = inputs
    zero = np.zeros((1, 2), np.longdouble)
    expected_d = np.vstack((zero, np.cumsum(np.cumsum(padded, axis=0), axis=0)))
    expected_mass = np.vstack((zero, np.cumsum(abs(padded), axis=0)))
    steps = 2*((padded_size+511)//512+48)
    gamma = steps*2.**-53/(1-steps*2.**-53)
    maxima = np.max(abs(padded), axis=0)
    prefix_budget = 4*gamma*padded_size**2*maxima
    nodes = np.arange(prefix_size)*stride
    assert np.all(abs(d.astype(np.longdouble)-expected_d[nodes]) <= prefix_budget)
    assert np.all(abs(mass-expected_mass[nodes]) <= 4*gamma*padded_size*maxima+2.**-19*abs(mass)+2.**-1020)
    at = np.arange(padded_size+1)
    cells = at//stride
    fraction = (at%stride).astype(np.longdouble)[:, None]/stride
    interpolated = (1-fraction)*d[cells]+fraction*d[np.minimum(cells+1, prefix_size-1)]
    assert np.all(abs(interpolated-expected_d) <= prefix_budget+(stride*stride/8)*maxima)


@pytest.mark.parametrize('g', [1, 2, 4, 8, 16, 32])
def test_uniform_square_uses_original_fma_and_reduction_order(g):
    widths = np.array([1, 7, 513, 1025], dtype=np.int32)
    rng = np.random.default_rng(909)
    signal = rng.uniform(-.5, .5, (4, widths.max())).astype(np.float32)
    signal[0] = 0
    weight = np.float32(.27)
    size = int(widths.max()+1)
    module = core.load_module()
    widths_gpu, selected = cp.asarray(widths), cp.arange(4, dtype=cp.int32)
    signal_gpu, square = cp.asarray(signal), cp.empty(4, cp.float32)
    module.get_function('prepareUniformSquare')((4,), (32,),
        (signal_gpu, widths_gpu, selected, cp.full(4, g, cp.int32), np.int32(widths.max()), weight, square))
    # With zero AR and unit amplitude, the unchanged exact search returns A2.
    weighted = cp.zeros((4, size, 2), cp.float32)
    weighted[:, :, 1] = weight
    best = cp.full(4, core.EMPTY_KEY, cp.uint64)
    args = (best, cp.tile(cp.arange(1, size+1, dtype=cp.int64), (4, 1)), np.float32(1),
            weighted, signal_gpu, cp.full(4, .5, cp.float32), widths_gpu, selected,
            cp.arange(5, dtype=cp.int32), cp.ones(4, cp.int32), cp.eye(4, dtype=cp.bool_),
            np.int32(0), np.int32(4), np.int32(4), np.int32(1), np.int32(size),
            np.int32(widths.max()), np.float32(0))
    module.get_function(f'searchChi2G{g}')(((4*g+255)//256, 4), (256,), args)
    assert square.get().tobytes() == core.decode_keys(best, 0)[0].get().tobytes()


@pytest.mark.parametrize('cheap_bound', [False, True])
@pytest.mark.parametrize('g', [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize('pattern', ['random', 'outlier', 'tie', 'overflow', 'uniform',
                                     'sparse', 'sparse-outlier', 'sparse-tie', 'sparse-overflow',
                                     'coarse-random', 'coarse-outlier', 'coarse-tie', 'coarse-overflow',
                                     'integer-coarse-random', 'integer-coarse-outlier',
                                     'integer-coarse-tie', 'integer-coarse-overflow',
                                     'integer-sparse', 'integer-sparse-outlier',
                                     'integer-sparse-tie', 'integer-sparse-overflow'])
def test_screen_bounds_cover_every_exact_candidate(g, pattern, cheap_bound):
    integer = pattern.startswith('integer-')
    if integer:
        pattern = pattern[len('integer-'):]
    sparse = pattern.startswith(('sparse', 'coarse'))
    uniform = pattern == 'uniform' or pattern.startswith('sparse')
    pattern = pattern.split('-', 1)[1] if '-' in pattern else pattern
    rng = np.random.default_rng(318)
    n, width = 131, 513
    size = n + width
    integers = rng.integers(-11, 17, n)
    integers[0] = 100
    if uniform and pattern == 'outlier':
        integers[-3] = 1 << 25
    inv_scale = np.float32(2.0**111 if pattern == 'overflow' else 2.0**-14)
    residual = np.resize(integers, size).astype(np.float32)*inv_scale
    dy = np.exp(rng.uniform(-3, 3, n)).astype(np.float32)
    if pattern == 'outlier':
        dy[-3] = 1e-9  # variance floor; small windows after a very large weighted sample
    weight = np.resize(np.float32(1)/np.maximum(dy*dy, np.float32(1e-10)), size)
    if uniform:
        weight.fill(np.float32(.27))
    z = weight*residual
    x = np.linspace(-1, 1, width)
    signal = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
    if pattern == 'tie':
        signal.fill(0)
    quantization = prefix_quantization(size, float(abs(z).max()), float(weight.max())) if integer else None
    point_error = None if quantization is None else quantization[2]
    prefix_maxima = (np.array([abs(z).max(), weight.max()]) if point_error is None else
                     np.nextafter(np.array([abs(z).max(), weight.max()])+point_error, np.inf))
    def make_certificate(tolerance):
        stride = (UNIFORM_PREFIX_STRIDE if uniform else GENERAL_PREFIX_STRIDE) if sparse else 1
        padded_size = (size+stride-1)//stride*stride
        pos, coef, meta, mass_error = certificates(
            signal[None], [width], abs(z).max(), weight.max(), padded_size, tolerance, point_error)
        if sparse:
            pos, coef, meta = sparse_certificate(
                pos, coef, meta, stride, padded_size, *prefix_maxima, uniform)
        return pos, coef, meta, mass_error

    pos, coef, meta, mass_error = make_certificate(SCREEN_TOLERANCES[0])
    # Every row is the same folded curve. A diagonal selector below evaluates each
    # candidate separately with the original FMA chain and XOR reduction.
    weighted = cp.asarray(np.tile(np.stack((z, weight), axis=-1), (n, 1, 1)))
    cum = cp.asarray(np.tile(np.cumsum(np.resize(integers, size)), (n, 1)))
    prefix = cp.empty(n*size*24, dtype=cp.uint8)
    module = cp.RawModule(code=core.KERNEL_SOURCE.read_text(),
                          options=(f"-DGTLS_CHEAP_SCREEN_BOUNDS={int(cheap_bound)}",))
    prefix_kernel = 'scanScreening' + ('Uniform' if uniform else '') + ('Sparse' if sparse else '')
    prefix_input = weighted[:, :, 0].copy() if sparse and uniform else weighted
    prefix_args = (prefix_input, prefix, np.int32(size))
    if integer:
        prefix_kernel = 'scanScreeningUniformInteger' if uniform else 'scanScreeningInteger'
        prefix_args += (*quantization[0], *quantization[1])
    module.get_function(prefix_kernel)((n,), (core.SCREEN_SCAN_THREADS,), prefix_args)
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
    block_map = cp.asarray(core.screen_block_map(np.array([0, n]), np.array([1])))
    budgets = cp.empty((3, 4), dtype=cp.float64)
    budget_kernel = 'prepareScreenBudgets' + (('UniformSparse' if uniform else 'Sparse') if sparse else '')
    module.get_function(budget_kernel)((1,), (64,),
        (widths, selected, screen_args[10], screen_args[14], budgets, np.int32(1), np.int32(1)))
    extra = (block_map, budgets)
    suffix = 'MappedSparse' if sparse else 'Mapped'
    if uniform:
        square = cp.empty(1, cp.float32)
        module.get_function('prepareUniformSquare')((1,), (32,),
            (cp.asarray(signal), widths, selected, screen_args[10], np.int32(width), weight[0], square))
        extra += (square, weight[0])
        suffix = 'UniformSparse' if sparse else 'Uniform'
    module.get_function('screenCandidates'+suffix)((len(block_map), n), (core.SCREEN_THREADS,),
                                                   screen_args+extra)
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
    if uniform:
        packed_args = args[:3] + (weighted[:, :, 0].copy(),) + args[4:]
        best.fill(core.EMPTY_KEY)
        module.get_function(f'searchChi2PackedG{g}')(grid, (256,), packed_args+(weight[0],))
        np.testing.assert_array_equal(best.get(), expected)
        best.fill(core.EMPTY_KEY)
        module.get_function(f'searchChi2PackedScreenedG{g}')(grid, (256,),
            packed_args+(lower, upper, np.int32(n), weight[0]))
        np.testing.assert_array_equal(best.get(), expected)
    best.fill(core.EMPTY_KEY)
    exact(grid, (256,), args+(lower, upper, np.int32(n)))
    np.testing.assert_array_equal(best.get(), expected)
    if pattern != 'overflow':
        # Seed the best exact value as a valid upper bound, so the early-bound
        # path is exercised even when the entire row fits in a single block.
        upper.fill(np.uint64(int(expected[0]) & 0xffffffff00000000))
        module.get_function('screenCandidates'+suffix)((len(block_map), n), (core.SCREEN_THREADS,),
                                                       screen_args+extra)
        assert np.all(lower[0].get() <= delta)
        best.fill(core.EMPTY_KEY)
        exact(grid, (256,), args+(lower, upper, np.int32(n)))
        np.testing.assert_array_equal(best.get(), expected)
    # Each refinement keeps existing pruned bounds and independently certifies
    # the survivors. Check every exact candidate, including overflow and ties.
    for tolerance in SCREEN_TOLERANCES[1:]:
        pos, coef, meta, mass_error = make_certificate(tolerance)
        before_lower, before_upper = lower.copy(), upper.copy()
        refined_args = screen_args[:12] + (cp.asarray(pos), cp.asarray(coef), cp.asarray(meta),
                                           *map(np.float64, mass_error)) + screen_args[17:]
        module.get_function('refineCandidates'+suffix)((len(block_map), n), (core.SCREEN_THREADS,),
                                                       refined_args+extra)
        assert bool(cp.all(lower >= before_lower))
        assert bool(cp.all(upper <= before_upper))
        if pattern != 'overflow':
            assert np.all(lower[0].get() <= delta)
        best.fill(core.EMPTY_KEY)
        exact(grid, (256,), args+(lower, upper, np.int32(n)))
        np.testing.assert_array_equal(best.get(), expected)
    candidates = cp.full((n, n), -1, dtype=cp.int32)
    counts = cp.zeros(n, dtype=cp.int32)
    module.get_function('compactCandidates')(((n+255)//256, n), (256,),
        (lower, upper, candidates, counts, np.int32(0), np.int32(n), np.int32(n), np.int32(n)))
    maximum = int(counts.max())
    best.fill(core.EMPTY_KEY)
    if maximum:
        module.get_function(f'searchChi2CompactedG{g}')(((maximum*g+255)//256, n), (256,),
            args+(lower, upper, np.int32(n), candidates, counts, np.int32(n)))
    np.testing.assert_array_equal(best.get(), expected)
    if pattern == 'tie':
        assert int(expected[0] & np.uint64(0xffffffff)) == 0
    if uniform and maximum:
        best.fill(core.EMPTY_KEY)
        module.get_function(f'searchChi2PackedCompactedG{g}')(((maximum*g+255)//256, n), (256,),
            packed_args+(lower, upper, np.int32(n), candidates, counts, np.int32(n), weight[0]))
        np.testing.assert_array_equal(best.get(), expected)


@pytest.mark.parametrize('tolerance', SCREEN_TOLERANCES)
def test_sparse_certificate_reconstructs_target_within_bound(tolerance):
    rng = np.random.default_rng(991)
    for width in [512, 1031, 8193]:
        x = np.linspace(-1, 1, width)
        s = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
        s += rng.uniform(-1e-5, 1e-5, width).astype(np.float32)
        pos, coef, error, _ = template_certificate(s, tolerance)
        # Direct definition of the represented discrete template, independent of scans.
        j = np.arange(width, dtype=np.longdouble)
        represented = np.maximum(pos[:,None]-1-j[None], 0).T @ coef.astype(np.longdouble)
        target = np.column_stack((s, s*s))
        assert np.all(np.max(abs(represented-target), axis=0) <= error)


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('pattern', ['random', 'constant', 'alternating', 'outlier'])
def test_resampled_certificate_covers_direct_long_double_dot(pattern, uniform):
    stride = UNIFORM_PREFIX_STRIDE if uniform else GENERAL_PREFIX_STRIDE
    width, size = 513, 1408
    rng = np.random.default_rng(2729)
    z = rng.normal(0, .01, size).astype(np.float32)
    if pattern == 'constant':
        z.fill(.01)
    elif pattern == 'alternating':
        z[:] = np.where(np.arange(size) % 2, -.01, .01)
    elif pattern == 'outlier':
        z[0] = 1000.
    x = np.linspace(-1, 1, width)
    signal = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
    w = np.exp(rng.normal(0, 2, size)).astype(np.float32)
    if pattern == 'outlier':
        w[0] = 1e9
    pos, coef, meta, _ = certificates(signal[None], [width], abs(z).max(), w.max(), size)
    pos, coef, meta = sparse_certificate(pos, coef, meta, stride, size,
                                          float(abs(z).max()), float(w.max()), uniform)
    inputs = np.column_stack((z, w)).astype(np.longdouble)
    prefix = np.vstack((np.zeros((1, 2), np.longdouble), np.cumsum(np.cumsum(inputs, axis=0), axis=0)))
    for start in range(2*stride):
        row = meta[start % stride]
        begin, count = map(int, row[:2])
        nodes = (start//stride+pos[begin:begin+count]-2)*stride
        values = prefix[np.maximum(nodes, 0)]
        for column in range(1 if uniform else 2):
            template = signal if column == 0 else signal*signal
            estimate = np.sum(values[:, column]*coef[begin:begin+count, column].astype(np.longdouble))
            target = np.sum(inputs[start:start+width, column]*template.astype(np.longdouble))
            radius = row[2+column]*np.sum(abs(inputs[start:start+width, column]))+row[6+column]
            assert abs(estimate-target) <= radius


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


def test_candidate_compaction_keeps_ties_nonfinite_bounds_and_tail():
    n, begin, end = 801, 7, 786
    rng = np.random.default_rng(818)
    lower = rng.normal(size=(4, n)).astype(np.float32)
    lower[:, 11:15] = [-.5, np.inf, -np.inf, np.nan]
    lower[-1].fill(1)  # an entirely pruned row
    thresholds = np.array([-.5, np.inf, np.nan, -np.inf], np.float32)
    bits = thresholds.view(np.uint32)
    ordered = np.where(bits & 0x80000000, ~bits, bits | 0x80000000).astype(np.uint64)
    upper = cp.asarray(ordered << 32)
    candidates = cp.full((4, end-begin), -1, cp.int32)
    counts = cp.zeros(4, cp.int32)
    core.load_module().get_function('compactCandidates')(((end-begin+255)//256, 4), (256,),
        (cp.asarray(lower), upper, candidates, counts, np.int32(begin), np.int32(end),
         np.int32(n), np.int32(end-begin)))
    candidates, counts = candidates.get(), counts.get()
    for row in range(4):
        expected = np.flatnonzero(~(lower[row, begin:end] > thresholds[row]))+begin
        assert counts[row] == len(expected)
        np.testing.assert_array_equal(np.sort(candidates[row, :counts[row]]), expected)
        assert np.all(candidates[row, counts[row]:] == -1)


@pytest.mark.parametrize('maximum', [0., float(np.finfo(np.float32).smallest_subnormal),
                                    float(np.finfo(np.float32).tiny), .27, 1.e9,
                                    float(np.finfo(np.float32).max)])
@pytest.mark.parametrize('size', [8192, 706406, 10000000])
def test_integer_prefix_scale_range_and_quantization_radius(maximum, size):
    scale, inverse, radius = prefix_quantization(size, maximum, maximum)
    padded = (size+511)//512*512
    limit = (2**61)//(padded*(padded+1)//2)-2
    rng = np.random.default_rng(2189)
    values = (rng.uniform(-1, 1, 2048)*maximum).astype(np.float32)
    values[:3] = [-maximum, maximum, 0.]
    quantized = np.rint((values*scale[0]).astype(np.float64)).astype(np.int64)
    assert np.max(abs(quantized)) <= limit
    assert np.all(abs(quantized.astype(np.longdouble)*inverse[0]-values.astype(np.longdouble)) <= radius[0])
    assert prefix_quantization(2**31, maximum, maximum) is None


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('size', [1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1025, 706406])
def test_integer_screen_prefix_matches_independent_integer_reference(size, uniform):
    stride, columns = (16, 1) if uniform else (8, 2)
    rng = np.random.default_rng(size)
    w = np.exp(rng.uniform(-10, 10, size)).astype(np.float32)
    z = (w*rng.normal(0, .01, size)).astype(np.float32)
    inputs = np.column_stack((z, w))
    padded_size = (size+stride-1)//stride*stride
    prefix_size = padded_size//stride+1
    scale, inverse, radius = prefix_quantization(padded_size, float(abs(z).max()), float(w.max()))
    padded = np.zeros((padded_size, 2), np.float32)
    padded[:size] = inputs
    quantized = np.rint((padded*scale).astype(np.float64)).astype(np.int64)
    assert np.all(abs(quantized.astype(np.longdouble)*inverse-padded.astype(np.longdouble)) <= radius)
    zero = np.zeros((1, 2), np.int64)
    d = np.vstack((zero, np.cumsum(np.cumsum(quantized, axis=0), axis=0)))
    mass = np.vstack((zero, np.cumsum(abs(quantized), axis=0)))
    prefix = cp.empty((12 if uniform else 24)*prefix_size, cp.uint8)
    kernel = 'scanScreeningUniformInteger' if uniform else 'scanScreeningInteger'
    core.load_module().get_function(kernel)((1,), (core.SCREEN_SCAN_THREADS,),
        (cp.asarray(z if uniform else inputs), prefix, np.int32(size), *scale, *inverse))
    actual_d = prefix[:8*columns*prefix_size].view(cp.float64).reshape(prefix_size, columns).get()
    actual_mass = prefix[8*columns*prefix_size:].view(cp.uint32).reshape(prefix_size, columns).get()
    nodes = np.arange(prefix_size)*stride
    expected = d[nodes, :columns].astype(np.float64)*inverse[:columns]
    assert actual_d.tobytes() == expected.tobytes()
    exact_mass = mass[nodes, :columns].astype(np.float64)*inverse[:columns]
    np.testing.assert_array_equal(actual_mass, (exact_mass.view(np.uint64) >> 32).astype(np.uint32))
    original_d = np.vstack((zero, np.cumsum(np.cumsum(padded.astype(np.longdouble), axis=0), axis=0)))
    rounding = 2.**-51*abs(expected.astype(np.longdouble))
    error = radius[:columns]*(padded_size*(padded_size+1)/2)
    assert np.all(abs(actual_d.astype(np.longdouble)-original_d[nodes, :columns]) <= error+rounding)


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('outlier', [False, True])
def test_quantized_certificate_covers_long_curve_dot(uniform, outlier):
    size, width = 706406, 513
    stride, columns = (16, 1) if uniform else (8, 2)
    padded_size = (size+stride-1)//stride*stride
    prefix_size = padded_size//stride+1
    rng = np.random.default_rng(6642)
    w = np.exp(rng.normal(0, 1, size)).astype(np.float32)*np.float32(1.e6)
    z = (w*rng.normal(0, .001, size)).astype(np.float32)
    if outlier:
        w[0], z[0] = 1.e10, 1.e8
    scale, inverse, radius = prefix_quantization(padded_size, float(abs(z).max()), float(w.max()))
    x = np.linspace(-1, 1, width)
    signal = (.5*np.maximum(0, 1-x*x)**.3).astype(np.float32)
    pos, coef, meta, _ = certificates(signal[None], [width], abs(z).max(), w.max(),
                                       padded_size, point_error=radius)
    maxima = np.nextafter(np.array([abs(z).max(), w.max()])+radius, np.inf)
    pos, coef, meta = sparse_certificate(pos, coef, meta, stride, padded_size, *maxima, uniform)
    inputs = np.column_stack((z, w))
    prefix = cp.empty((12 if uniform else 24)*prefix_size, cp.uint8)
    kernel = 'scanScreeningUniformInteger' if uniform else 'scanScreeningInteger'
    core.load_module().get_function(kernel)((1,), (core.SCREEN_SCAN_THREADS,),
        (cp.asarray(z if uniform else inputs), prefix, np.int32(size), *scale, *inverse))
    values = prefix[:8*columns*prefix_size].view(cp.float64).reshape(prefix_size, columns).get().astype(np.longdouble)
    for region in [0, size//2, size-width-2*stride]:
        for start in range(region, region+stride):
            row = meta[start % stride]
            begin, count = map(int, row[:2])
            nodes = start//stride+pos[begin:begin+count]-2
            at = np.maximum(nodes, 0)
            for column in range(columns):
                template = signal if column == 0 else signal*signal
                estimate = np.sum(values[at, column]*coef[begin:begin+count, column].astype(np.longdouble))
                window = inputs[start:start+width, column].astype(np.longdouble)
                target = np.sum(window*template.astype(np.longdouble))
                bound = row[2+column]*(np.sum(abs(window))+width*radius[column])+row[6+column]
                assert abs(estimate-target) <= bound


@pytest.mark.parametrize('g', [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('pattern', ['random', 'tie', 'overflow'])
def test_screen_seed_matches_original_candidate_statistic(g, packed, pattern):
    module = core.load_module()
    rng = np.random.default_rng(113)
    rows, n, width = 33, 131, 513
    size = n+width
    integers = rng.integers(-11, 17, (rows, n))
    integers[:, 0] = 100
    repeated = integers[:, np.arange(size) % n]
    inv = np.float32(2.**111 if pattern == 'overflow' else 2.**-14)
    residual = repeated.astype(np.float32)*inv
    weight = np.exp(rng.uniform(-3, 3, (rows, size))).astype(np.float32)
    if packed:
        weight.fill(.27)
    weighted = cp.asarray(np.stack((residual*weight, weight), axis=-1))
    cumulative = cp.asarray(np.cumsum(repeated, axis=1))
    x = np.linspace(-1, 1, width)
    signal = cp.asarray((.5*np.maximum(0, 1-x*x)**.3).astype(np.float32))
    if pattern == 'tie':
        signal.fill(0)
    overshoot = cp.asarray([1.173], cp.float32)
    widths, selected = cp.asarray([width], cp.int32), cp.asarray([0], cp.int32)
    offsets, strides = cp.asarray([0, n], cp.int32), cp.ones(1, cp.int32)
    lanes = cp.asarray([g], cp.int32)
    mask = cp.ones((rows, 1), cp.bool_)
    mask[0] = False
    upper = cp.full(rows, core.EMPTY_KEY, cp.uint64)
    module.get_function('seedScreenCandidates')((rows,), (256,),
        (upper, cumulative, inv, widths, selected, offsets, strides, mask,
         np.int32(1), np.int32(1), np.int32(n), np.int32(size), np.float32(-1)))
    chosen = upper.get()
    valid = chosen != core.EMPTY_KEY
    starts = (chosen & np.uint64(0xffffffff)).astype(np.int64)
    assert not valid[0]
    module.get_function('evaluateScreenSeedPacked' if packed else 'evaluateScreenSeed')((rows,), (32,),
        (upper, cumulative, inv, weighted[:, :, 0].copy() if packed else weighted,
         signal, overshoot, widths, selected, lanes, np.int32(n), np.int32(size),
         np.int32(width), np.float32(.27)))
    # Evaluate only the chosen original-grid candidate in the unchanged kernel.
    lower = cp.full((rows, n), cp.inf, cp.float32)
    lower[cp.asarray(np.flatnonzero(valid)), cp.asarray(starts[valid])] = -cp.inf
    best = cp.full(rows, core.EMPTY_KEY, cp.uint64)
    zero_upper = cp.full(rows, np.uint64(0x8000000000000000), cp.uint64)
    args = (best, cumulative, inv, weighted, signal, overshoot, widths, selected,
            offsets, strides, mask, np.int32(0), np.int32(1), np.int32(1),
            np.int32(n), np.int32(size), np.int32(width), np.float32(-1), lower,
            zero_upper, np.int32(n))
    module.get_function(f'searchChi2ScreenedG{g}')(((n*g+255)//256, rows), (256,), args)
    np.testing.assert_array_equal(upper.get() >> 32, best.get() >> 32)
