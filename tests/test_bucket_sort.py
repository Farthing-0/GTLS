"""The reduced histogram must retain stable phase ordering, including ties."""
import cupy as cp
import numpy as np
import pytest

from gputls import core
from gputls.workspace import bucket_count


def test_phase_fraction_matches_original_gpu_arithmetic():
    rng = np.random.default_rng(42389)
    positive = np.ldexp(rng.uniform(.5, 1, 50000), rng.integers(-1070, 1022, 50000))
    integers = np.ldexp(np.ones(32), np.arange(32))
    values = np.concatenate((positive, -positive, integers,
                             np.nextafter(integers, 0.), np.nextafter(integers, np.inf),
                             [0., -0., np.inf, -np.inf]))
    code = core.KERNEL_SOURCE.read_text() + r'''
extern "C" __global__ void checkFraction(const double* x, double* actual,
                                        double* reference, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        actual[i] = phaseFraction(x[i]);
        reference[i] = x[i] - (int)x[i];
    }
}
'''
    kernel = cp.RawModule(code=code).get_function('checkFraction')
    actual, reference = cp.empty(values.size, cp.float64), cp.empty(values.size, cp.float64)
    kernel(((values.size+255)//256,), (256,),
           (cp.asarray(values), actual, reference, np.int32(values.size)))
    np.testing.assert_array_equal(actual.get().view(np.uint64), reference.get().view(np.uint64))


@pytest.mark.parametrize('n', [1, 2, 3, 101, 1031])
def test_bucket_sort_matches_stable_cpu_order(n):
    rng = np.random.default_rng(217)
    times = rng.uniform(0, 100, n)
    edges = np.array([0, 0, np.nextafter(1., 0.), 1., np.nextafter(1., 2.),
                      .5, 1.5, 2., 1e6])
    times[:min(n, len(edges))] = edges[:min(n, len(edges))]
    times[::3] = 1.0  # Many equal phases exercise stable tie ordering.
    inverse_periods = 1/np.array([1., 2., 2.3, .3])
    periods, buckets = len(inverse_periods), bucket_count(n)
    phases = cp.empty((periods, n), cp.float64)
    counts = cp.zeros((periods, buckets), cp.int32)
    cursor = cp.empty_like(counts)
    indices = cp.empty((periods, n), cp.int32)
    module = core.load_module()
    grid = ((n+255)//256, periods)
    module.get_function('foldBuckets')(
        grid, (256,), (cp.asarray(times), cp.asarray(inverse_periods), phases,
                       counts, np.int32(n), np.int32(buckets)))
    module.get_function('scanCounts')(
        (periods,), (core.SCAN_THREADS,), (counts, cursor, np.int32(buckets)))
    module.get_function('bucketScatter')(
        grid, (256,), (phases, cursor, indices, np.int32(n), np.int32(buckets)))
    module.get_function('sortBuckets')(
        ((buckets+255)//256, periods), (256,),
        (phases, counts, cursor, indices, np.int32(n), np.int32(buckets)))
    expected = np.argsort(phases.get(), axis=1, kind='stable')
    np.testing.assert_array_equal(indices.get(), expected)
    np.testing.assert_array_equal(counts.sum(axis=1).get(), np.full(periods, n))
