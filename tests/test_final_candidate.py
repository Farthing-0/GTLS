"""Final epochs and durations against direct CPU residuals and time geometry."""
import numpy as np
import pytest

from gputls import gtls, stats


def direct_epoch(signal, depth, t, y, dy, period, margin):
    """Slow independent reference: re-fold and sum all residuals at each epoch."""
    width, n = len(signal), len(t)
    count = n if margin == 0 else min(n, int(n / (margin * width)))
    epochs = np.linspace(t.min(), t.min() + period, count)
    model = np.ones(n)
    model[:width] = 1 - (1 - np.asarray(signal, float)) / (.5 / (1 - depth))
    residuals = []
    for epoch in epochs:
        phase = (t - epoch) / period
        phase -= np.floor(phase)
        order = np.roll(np.argsort(phase, kind='stable'), int(width / 2) + 1)
        residuals.append(np.sum(((y[order] - model) / dy[order]) ** 2))
    return epochs[np.argmin(residuals)]


@pytest.mark.parametrize('width', [7, 8, 31])
@pytest.mark.parametrize('margin', [0., .01, .2])
@pytest.mark.parametrize('ordered', [False, True])
def test_final_epoch_matches_direct_weighted_residuals(width, margin, ordered):
    rng = np.random.default_rng(224 + width)
    t = np.sort(rng.uniform(1.2, 35., 127))
    y = 1 + rng.normal(0, .002, len(t))
    dy = rng.uniform(.001, .004, len(t))
    if not ordered:
        indices = rng.permutation(len(t)); t, y, dy = t[indices], y[indices], dy[indices]
    signal = 1 - .5 * np.sin(np.linspace(.1, np.pi-.1, width))
    args = (signal, .997, t, y, dy, 3.731, margin)
    assert stats.final_T0_fit(*args) == direct_epoch(*args)


def test_duration_uses_full_baseline_across_gaps():
    t = np.r_[np.arange(1., 5., .1), np.arange(40., 44., .1)]
    period, epoch, fraction = 7.3, 2.7, .03
    times = stats.all_transit_times(epoch, t, period)
    expected = fraction * (t.max() - t.min()) / len(times)
    assert stats.calculate_transit_duration_in_days(t, period, times, fraction) == pytest.approx(expected)


def test_zero_depth_has_no_preferred_epoch():
    t = np.array([4., 1., 2., 3.])
    assert stats.final_T0_fit(np.array([.5]), 1., t, np.ones(4), np.ones(4), 2., .01) == 1.


@pytest.mark.parametrize('period', [.73, 7.31])
def test_final_candidate_uses_epoch_fit_and_stretched_duration(period):
    rng = np.random.default_rng(224)
    t = np.arange(1.001, 31., .04)
    t = t[~((t > 12) & (t < 15))]
    distance = (t - 1.213 + period/2) % period - period/2
    dy = rng.uniform(.0002, .0006, len(t))
    y = 1 - .005 * (abs(distance) < .05) + rng.normal(size=len(t))*dy
    model = gtls(t, y, dy, verbose=False)
    result = model.power(periods=np.linspace(period*.999, period*1.001, 51),
                         T0_fit_margin=.01, T0_refine_margin=.01)
    row = np.flatnonzero(model.lc_cache_overview['duration'] == result.rawDuration)[0]
    expected = direct_epoch(model.lc_arr[row], 1-result.depth, model.t, model.y, model.dy,
                            result.period, .01)
    assert result.T0 == expected
    centers = stats.all_transit_times(expected, model.t, result.period)
    np.testing.assert_array_equal(result.transitTimes, centers)
    assert result.duration == pytest.approx(result.rawDuration*(t.max()-t.min())/len(centers))
