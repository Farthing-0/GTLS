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


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('n', [1, 2, 3, 101, 1031])
def test_bucket_sort_matches_stable_cpu_order(n, packed):
    rng = np.random.default_rng(217)
    times = rng.uniform(0, 100, n)
    edges = np.array([0, 0, np.nextafter(1., 0.), 1., np.nextafter(1., 2.),
                      .5, 1.5, 2., 1e6])
    times[:min(n, len(edges))] = edges[:min(n, len(edges))]
    times[::3] = 1.0  # Many equal phases exercise stable tie ordering.
    if n >= 101:
        # Distinct phases sharing a compact key, including its sentinel key.
        times[20:40] = .3 + np.arange(20)*2.**-50
        times[40:60] = 1. - np.arange(1, 21)*2.**-50
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
    expected = np.argsort(phases.get(), axis=1, kind='stable')
    suffix, extra = '', ()
    if packed:
        phases = cp.empty((periods, n), cp.uint32)
        reference_counts = counts.copy()
        counts.fill(0)
        extra = (cp.asarray(times), cp.asarray(inverse_periods))
        module.get_function('foldBucketsPackedPhase')(
            grid, (256,), extra + (phases, counts, np.int32(n), np.int32(buckets)))
        np.testing.assert_array_equal(counts.get(), reference_counts.get())
        suffix = 'PackedPhase'
    module.get_function('scanCounts')(
        (periods,), (core.SCAN_THREADS,), (counts, cursor, np.int32(buckets)))
    module.get_function('bucketScatter'+suffix)(
        grid, (256,), (phases,) + extra + (cursor, indices, np.int32(n), np.int32(buckets)))
    module.get_function('sortBuckets'+suffix)(
        ((buckets+255)//256, periods), (256,),
        (phases,) + extra + (counts, cursor, indices, np.int32(n), np.int32(buckets)))
    np.testing.assert_array_equal(indices.get(), expected)
    np.testing.assert_array_equal(counts.sum(axis=1).get(), np.full(periods, n))


@pytest.mark.parametrize('buckets', [1, 17, 210240, 2**21-1, 2**21+1, 2**24+1, 2**31-1])
def test_packed_phase_bucket_boundaries_match_fp64(buckets):
    rng = np.random.default_rng(73512)
    edges = rng.integers(0, buckets+1, 5000).astype(np.float64)/buckets
    quantized = rng.integers(0, 2**32, 5000).astype(np.float64)/2**32
    values = np.concatenate((rng.random(10000), edges, np.nextafter(edges, 0.),
                             np.nextafter(edges, 1.), quantized, np.nextafter(quantized, 0.),
                             np.nextafter(quantized, 1.), [0., np.nextafter(1., 0.)]))
    values = values[(values >= 0) & (values < 1)]
    code = core.KERNEL_SOURCE.read_text() + r'''
extern "C" __global__ void checkPackedBucket(const double* x, int* actual, int* reference,
                                            int size, int buckets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        actual[i] = packedPhaseBucket(phaseKey(x[i]), x, 1., i, buckets);
        reference[i] = min((int)(x[i] * buckets), buckets - 1);
    }
}
'''
    kernel = cp.RawModule(code=code).get_function('checkPackedBucket')
    actual, reference = cp.empty(values.size, cp.int32), cp.empty(values.size, cp.int32)
    kernel(((values.size+255)//256,), (256,),
           (cp.asarray(values), actual, reference, np.int32(values.size), np.int32(buckets)))
    np.testing.assert_array_equal(actual.get(), reference.get())
