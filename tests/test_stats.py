"""Postprocessing shortcuts preserve strict windows and floating-point results."""
from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest

from gputls import core, stats
from gputls.workspace import BatchWorkspace


@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('duration', [0., -2., 4., np.inf])
def test_transit_windows_preserve_boundaries_duplicates_and_input_order(ordered, duration):
    t = np.array([-np.inf, -3., -2., -2., 0., 1., 2., 2., 4., np.inf])
    y = np.arange(t.size, dtype=np.float64)
    if not ordered:
        index = np.array([4, 0, 7, 2, 8, 1, 9, 5, 3, 6])
        t, y = t[index], y[index]
    times = [-10., -2., 0., 2., 10., np.nan]
    expected = []
    with np.errstate(invalid='ignore'):
        for mid in times:
            expected.append(y[(t > mid - duration / 2) & (t < mid + duration / 2)])
        actual = list(stats._transit_flux(t, y, times, duration))
    assert all(a.tobytes() == e.tobytes() for a, e in zip(actual, expected))
    odd, even, counts = stats.intransit_stats(t, y, times, duration)
    assert odd.tobytes() == np.concatenate(expected[1::2]).tobytes()
    assert even.tobytes() == np.concatenate(expected[::2]).tobytes()
    np.testing.assert_array_equal(counts, [len(x) for x in expected])


@pytest.mark.parametrize('pattern', ['wide', 'flux', 'constant'])
def test_gpu_window_std_matches_numpy_at_pairwise_split_boundaries(pattern):
    rng = np.random.default_rng(73265)
    data = rng.normal(size=4352)
    if pattern == 'wide':
        data *= np.exp(rng.normal(0, 20, data.size))
    elif pattern == 'flux':
        data = 1 + data * 1e-8
    else:
        data.fill(.3)
    source = cp.asarray(data)
    kernel = core.load_module().get_function('windowStd')
    for width in [1, 7, 8, 9, 64, 125, 128, 129, 248, 249, 255, 256,
                  257, 263, 264, 384, 495, 496, 503, 504, 511, 512, 513,
                  520, 584, 1015, 1016, 1023, 1024]:
        count = len(data) - width + 1
        output = cp.empty(count, cp.float64)
        kernel(((count+255)//256,), (256,), (source, output, np.int32(count), np.int32(width)))
        expected = np.std(np.lib.stride_tricks.sliding_window_view(data, width), axis=1)
        assert output.get().tobytes() == expected.tobytes(), (pattern, width)


@pytest.mark.parametrize('width', [125, 257, 584, 1024])
def test_pink_noise_reuses_workspace_and_original_outer_chunk_reductions(width):
    data = 1 + np.random.default_rng(56378).normal(0, .01, 70000)
    # More than two original chunks, including the wider short-cadence windows.
    module = core.load_module()
    module.get_function('windowStd')
    search = SimpleNamespace(tGPU=cp.empty(1), module=module,
                             workspace=BatchWorkspace(1, len(data), len(data)+width, 1, True))

    class Allocations(cp.cuda.MemoryHook):
        name = 'StatsDriverAllocations'
        sizes = []

        def alloc_postprocess(self, device_id, mem_size, mem_ptr):
            self.sizes.append(mem_size)

    calls = []

    def window_std(values, width):
        calls.append(width)
        with Allocations() as allocations:
            result = core.GPUSearch.window_standard_deviations(search, values, width)
        assert not allocations.sizes
        return result

    expected = stats.pink_noise(data, width)
    actual = stats.pink_noise(data, width, window_std)
    assert calls == [width]
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_pink_noise_keeps_cpu_fallback_above_gpu_width_limit():
    data = 1 + np.random.default_rng(81).normal(0, .001, 6000)

    def unsupported(*args):
        pytest.fail('A window wider than 1024 must use the CPU fallback')

    expected = stats.pink_noise(data, 1025)
    actual = stats.pink_noise(data, 1025, unsupported)
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
