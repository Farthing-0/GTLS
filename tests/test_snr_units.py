"""End-to-end regression for day vs fractional-period noise-mask durations."""
import numpy as np
import pytest

from gputls import gtls, stats


@pytest.mark.parametrize("period", [0.73, 7.31])
def test_native_noise_statistics_use_duration_in_days(period):
    rng = np.random.default_rng(223)
    t = np.arange(1.001, 81., .02)
    epoch, duration = 1.213, .08
    distance = (t - epoch + period / 2) % period - period / 2
    y = 1 - .005 * (np.abs(distance) < duration / 2)
    y += rng.normal(0, .0003, len(t))
    result = gtls(t, y, np.full(len(t), .0003), verbose=False).power(
        periods=np.linspace(period * .999, period * 1.001, 101),
        T0_fit_margin=.01, T0_refine_margin=.01)

    # Independent masks in time units, and direct per-event sample selection.
    distance = (t - result.T0 + result.period / 2) % result.period - result.period / 2
    oot = y[np.abs(distance) >= result.duration]
    events = [y[(t > mid - result.duration / 2) &
                (t < mid + result.duration / 2)] for mid in result.transitTimes]
    inside = np.concatenate(events[1::2] + events[::2])
    expected = (1 - inside.mean()) / oot.std() * np.sqrt(len(inside))
    assert result.snr == pytest.approx(expected, rel=1e-12)

    # CPU pink_noise is independently fed the correctly excluded OOT sample.
    width = int(np.mean([len(event) for event in events]))
    pink = stats.pink_noise(oot, width)
    per_event = [(1 - event.mean()) / pink if len(event) else 0 for event in events]
    expected_pink = np.mean(per_event) * np.sqrt(len(events))
    assert result.snrPink == pytest.approx(expected_pink, rel=1e-10)
