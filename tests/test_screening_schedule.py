"""Changing a conservative pruning schedule must preserve the complete result."""
from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest

from gputls import core, gtls
from gputls.validate import validate_args
from gputls.screening_policy import BatchScreeningPolicy


@pytest.mark.parametrize('kwargs', [
    {'screening_policy': 'unknown'}, {'screening_policy': None},
    {'screening_policy': 'auto', 'screening_tolerances': ()},
    {'screening_policy': 'auto', 'screening_tolerances': (0.05,)},
])
def test_invalid_screening_policy(kwargs):
    with pytest.raises(ValueError, match='screening_policy'):
        validate_args(SimpleNamespace(), kwargs)


def test_auto_selection_ignores_first_use_spike_and_prefers_near_ties():
    assert BatchScreeningPolicy.choose({1: [9., 1.02], 2: [1., 1.1], 3: [1.4, 1.5]}) == 1
    assert BatchScreeningPolicy.choose({1: [2., 2.1], 2: [1., 1.1], 3: [.7, .8]}) == 3


@pytest.mark.parametrize('value', [0.05, [0], [-0.1], [np.nan], [np.inf],
                                  [0.01, 0.05], [0.05, 0.05], [[0.05]], 'bad'])
def test_invalid_screening_tolerances(value):
    with pytest.raises(ValueError, match='screening_tolerances'):
        validate_args(SimpleNamespace(), {'screening_tolerances': value})


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('cheap_bound', [False, True])
def test_screening_schedules_preserve_all_results_and_skip_short_refinement(monkeypatch, weighted, fused, cheap_bound):
    # Force each execution path independently of workload-size performance gates.
    monkeypatch.setattr(core, 'FUSED_SCREEN_MIN_POINTS', 0 if fused else 2**31)
    monkeypatch.setattr(core, 'FUSED_SCREEN_MIN_PERIODS', 0)
    rng = np.random.default_rng(119)
    t = np.arange(9216, dtype=float) / 720
    phase = (t - 1.1 + 1.85) % 3.7 - 1.85
    shape = np.clip((0.08 - abs(phase)) / 0.02, 0, 1)
    dy = np.full(len(t), 0.0003)
    if weighted:
        dy *= np.exp(rng.normal(0, 0.15, len(t)))
    y = 1 - 0.002 * shape + rng.normal(0, dy)
    periods = np.unique(np.r_[np.geomspace(0.7, 6.3, 48), 3.7])
    # Exercise the complete exact-template cascade, including the longest rows.
    expected = gtls(t, y, dy, verbose=False).power(periods=periods, screening_tolerances=(), backend="fused")
    observed = []
    cascades = []
    def loader():
        return cp.RawModule(code=core.KERNEL_SOURCE.read_text(),
                            options=(f'-DGTLS_CHEAP_SCREEN_BOUNDS={int(cheap_bound)}',))

    class CheckedModule:
        def __init__(self):
            self.module = loader()

        def get_function(self, name):
            kernel = self.module.get_function(name)

            def call(grid, block, args):
                if name.startswith('screenCascade4'):
                    cascades.append(name)
                if name.startswith('refineCandidates'):
                    widths, selected, block_map = args[6], args[7], args[22]
                    assert bool(cp.all(widths[selected[block_map[:, 0]]] >= core.MIN_WIDTH))
                    observed.append(name)
                return kernel(grid, block, args)

            return call

    monkeypatch.setattr(core, 'load_module', CheckedModule)
    for schedule in [None, (0.05,), (0.015, 0.00025), (0.04, 0.012, 0.004, 0.0003)]:
        actual = gtls(t, y, dy, verbose=False).power(periods=periods, screening_tolerances=schedule, backend="fused")
        for name, value in vars(expected).items():
            np.testing.assert_array_equal(np.ma.getdata(getattr(actual, name)), np.ma.getdata(value),
                                          err_msg=f'{name}, schedule={schedule}, weighted={weighted}')
            if np.ma.isMaskedArray(value):
                np.testing.assert_array_equal(np.ma.getmaskarray(getattr(actual, name)),
                                              np.ma.getmaskarray(value))
    assert observed, 'The test must exercise actual refinement kernels'
    assert bool(cascades) == fused


@pytest.mark.parametrize('weighted', [False, True])
def test_auto_matches_fixed_and_exposes_per_search_calibration(monkeypatch, weighted):
    # Exercise online calibration rather than the tiny-batch fallback.
    monkeypatch.setattr(core, 'BATCHES_MIN', 2)
    rng = np.random.default_rng(220)
    t = np.arange(9216, dtype=float) / 720
    dy = np.full(len(t), 0.0003)
    if weighted:
        dy *= np.exp(rng.normal(0, 0.2, len(t)))
    phase = (t-1.1+1.85) % 3.7-1.85
    y = 1 - .002*np.clip((.08-abs(phase))/.02, 0, 1) + rng.normal(0, dy)
    periods = np.unique(np.r_[np.geomspace(.7, 6.3, 48), 3.7])
    expected = gtls(t, y, dy, verbose=False).power(periods=periods)
    model = gtls(t, y, dy, verbose=False)
    actual = model.power(periods=periods, screening_policy='auto')
    for name, value in vars(expected).items():
        np.testing.assert_array_equal(np.ma.getdata(getattr(actual, name)), np.ma.getdata(value))
        if np.ma.isMaskedArray(value):
            np.testing.assert_array_equal(np.ma.getmaskarray(getattr(actual, name)), np.ma.getmaskarray(value))
    device = model.screening_diagnostics['devices'][0]
    assert device['enabled'] and device['calibration_seconds'] > 0
    assert device['batches'] and device['calibrations']
    assert len({c['group'] for c in device['calibrations']}) == len(device['calibrations'])
    assert all(c['redundant_periods'] == 0 for c in device['calibrations'])
    assert all(sum(t['periods'] for t in c['trials']) == c['evaluated_periods']
               for c in device['calibrations'])
    assert all(1 <= b['levels'] <= 4 for b in device['batches'])
    assert cp.get_default_memory_pool().total_bytes() == 0


def test_auto_skips_calibration_when_screening_disabled():
    t = np.arange(480, dtype=float)/48
    y = 1 + np.random.default_rng(41).normal(0, .001, len(t))
    model = gtls(t, y, verbose=False)
    model.power(periods=np.linspace(.7, 4, 40), screening_policy='auto', fast=True)
    device = model.screening_diagnostics['devices'][0]
    assert not device['enabled']
    assert device['calibration_seconds'] == 0 and device['calibrations'] == []


def test_auto_partitions_once_preserves_order_and_reuses_group(monkeypatch):
    search = object.__new__(core.GPUSearch)
    search.auto_screening = BatchScreeningPolicy(4, 65536)
    search.screening = True
    search.screening_levels = 4
    visits = []

    def exact(periods, size, refine):
        visits.append((np.asarray(periods).copy(), search.screening_levels, refine))
        return np.asarray(periods, dtype=np.float32) * 2

    monkeypatch.setattr(search, '_chi2_batch', exact)
    monkeypatch.setattr(search, 'reserve_workspace', lambda capacity: None)
    periods = np.linspace(2, 3, 19)
    np.testing.assert_array_equal(search.chi2_batch(periods, 19, False), periods.astype(np.float32)*2)
    np.testing.assert_array_equal(np.sort(np.concatenate([v[0] for v in visits])), periods)
    assert {v[1] for v in visits} == {1, 2, 3, 4}
    assert search.screening_levels == 4
    search.chi2_batch(periods, 19, False)
    assert len(visits) == 5 and len(search.auto_screening.calibrations) == 1
    # A refined pass in an unmeasured band must not initiate calibration.
    search.chi2_batch(periods + 100, 19, True)
    assert visits[-1][1:] == (1, True)
    assert len(search.auto_screening.calibrations) == 1


def test_auto_tiny_batch_uses_one_level_without_calibration(monkeypatch):
    search = object.__new__(core.GPUSearch)
    search.auto_screening = BatchScreeningPolicy(4, 9216)
    search.screening = True
    search.screening_levels = 4

    def exact(periods, size, refine):
        assert search.screening_levels == 1
        return np.asarray(periods, dtype=np.float32)

    monkeypatch.setattr(search, '_chi2_batch', exact)
    search.chi2_batch(np.array([2., 3.]), 2, False)
    assert search.auto_screening.calibrations == []
    assert search.screening_levels == 4
