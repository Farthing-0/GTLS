"""Adjacent windows must preserve the original weighted sums and key tie order."""
import cupy as cp
import numpy as np
import pytest

from gputls import core


@pytest.mark.parametrize('g,q', [(8, 2), (16, 2), (32, 4)])
@pytest.mark.parametrize('zero_template', [False, True])
def test_adjacent_keys_match_direct(g, q, zero_template):
    rng = np.random.default_rng(93)
    n = 1031
    widths = np.array([1, 7, 31, 64, 97, 131, 257, 389, 1025], dtype=np.int32)
    strides = np.array([1, 3, 7, 8, 17, 31, 33, 64, 97], dtype=np.int32)
    patched = n + int(widths.max())
    integers = rng.integers(-8, 13, (6, n))
    integers[0] = np.abs(integers[0]) + 1  # all windows active
    integers[1] = -np.abs(integers[1]) - 1  # no windows active
    integers = np.tile(integers, (1, 2))[:, :patched]
    residual = integers.astype(np.float32) * np.float32(2**-14)
    # Include unequal weights and samples on both sides of the variance floor.
    dy = np.exp(rng.uniform(-16, 4, (6, n))).astype(np.float32)
    dy = np.tile(dy, (1, 2))[:, :patched]
    weights = np.float32(1) / np.maximum(dy * dy, np.float32(1e-10))
    weighted = cp.asarray(np.stack((weights * residual, weights), axis=-1))
    signal = rng.uniform(0, .5, (len(widths), widths.max())).astype(np.float32)
    if zero_template:
        signal.fill(0)  # exact ties must choose the first duration and first start
    mask = rng.random((6, len(widths))) > .25
    mask[0] = True
    mask[-1] = False
    starts = (n + strides - 1) // strides
    direct_offsets = np.r_[0, np.cumsum(starts)].astype(np.int32)
    group_offsets = np.r_[0, np.cumsum((starts + q - 1) // q)].astype(np.int32)
    best = cp.full(6, core.EMPTY_KEY, dtype=cp.uint64)
    args = (best, cp.asarray(np.cumsum(integers, axis=1)), np.float32(2**-14), weighted,
            cp.asarray(signal), cp.asarray(rng.uniform(.8, 1.2, len(widths)).astype(np.float32)),
            cp.asarray(widths), cp.arange(len(widths), dtype=cp.int32), None,
            cp.asarray(strides), cp.asarray(mask), np.int32(0), np.int32(len(widths)),
            np.int32(len(widths)), np.int32(n), np.int32(patched), np.int32(widths.max()),
            np.float32(0))
    module = core.load_module()
    expected = None
    for name, offsets in [(f'searchChi2G{g}', direct_offsets),
                          (f'searchChi2AdjacentG{g}', group_offsets)]:
        best.fill(core.EMPTY_KEY)
        module.get_function(name)(((int(offsets[-1]) * g + 255) // 256, 6), (256,),
                                  args[:8] + (cp.asarray(offsets),) + args[9:])
        keys = best.get()
        if expected is None:
            expected = keys
        else:
            np.testing.assert_array_equal(keys, expected)
    assert expected[1] == core.EMPTY_KEY
    assert expected[-1] == core.EMPTY_KEY


def test_tail_uses_only_valid_periods(monkeypatch):
    """A large planned capacity must not launch duplicate periods in a short tail."""
    search = object.__new__(core.GPUSearch)
    search.constant_residual = 1.0
    seen = []

    def duration_mask(periods):
        seen.append(np.asarray(periods).copy())
        return cp.zeros((len(periods), 3), dtype=cp.bool_)

    monkeypatch.setattr(search, 'duration_mask', duration_mask)
    periods = np.array([2.7, 3.1, 4.9])
    result = search.chi2_batch(periods, size=815, refine=False)
    np.testing.assert_array_equal(seen[0], periods)
    np.testing.assert_array_equal(result, np.ones(3, dtype=np.float32))
