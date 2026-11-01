"""GTLS vs TLS 1.x vs TLS 2.x on lightkurve-prepared mock Kepler / TESS pipeline data.

Runs the "quick" suite of the comparison framework once (one search per engine and
scenario) and checks that every engine recovers the injected transit and that GTLS agrees
with both TLS releases. For the full speed benchmark use ``python tests/compare_tls.py``.

TLS releases are installed on first use into ``tests/.tls_envs`` (needs network access);
the tests are skipped if that is not possible. Select the GPU with ``GTLS_GPU_DEVICE``.
"""
import os

import numpy as np
import pytest

lk = pytest.importorskip("lightkurve")
pytest.importorskip("cupy")

from tls_compare import SCENARIOS, SUITES, registry, run_comparison  # noqa: E402
from tls_compare.compare import wrap_phase  # noqa: E402
from tls_compare.engines import prepare  # noqa: E402
from tls_compare.simulate import build_dataset  # noqa: E402

GPU_DEVICE = int(os.environ.get("GTLS_GPU_DEVICE", 0))
ENGINES = ("gtls", "tls1", "tls2")
QUICK = SUITES["quick"]


def _entry(report, scenario):
    return next(e for e in report["scenarios"] if e["name"] == scenario)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    engines = registry()
    for name in ENGINES:
        try:
            prepare(engines[name])
        except Exception as error:
            pytest.skip("cannot install %s: %s" % (engines[name].label, error))
    return run_comparison(QUICK, ENGINES, outdir=tmp_path_factory.mktemp("tls_compare"),
                          repeats=0, gpu=GPU_DEVICE, log=lambda *args: None)


@pytest.mark.parametrize("scenario", ["kepler_lc_1q", "tess_2min_1s"])
def test_mock_products_follow_pipeline_format(tmp_path, scenario):
    scenario = SCENARIOS[scenario]
    data = build_dataset(scenario, tmp_path)
    expected = {"Kepler": lk.KeplerLightCurve, "TESS": lk.TessLightCurve}[scenario.mission.name]

    # Recognised as a pipeline product. The simulator only sets quality bits of lightkurve's
    # default bitmask (and corrupts those cadences), so none of them may reach the search.
    raw = lk.read(data.files[0], quality_bitmask="none")
    assert isinstance(raw, expected)
    flagged = raw.time.value[raw.quality.value != 0]
    assert flagged.size > 0
    assert not np.isin(flagged, data.t).any()

    # Normalized and detrended, with the injected transits preserved
    truth = data.truth
    phase = wrap_phase(data.t - truth["T0"], truth["period"])
    in_transit = np.abs(phase) < 0.25 * truth["duration"]
    out_of_transit = np.abs(phase) > truth["duration"]
    assert abs(np.median(data.y) - 1) < 1e-4
    assert 1 - data.y[in_transit].mean() > 0.4 * truth["depth"]
    assert np.std(data.y[out_of_transit]) < 2 * scenario.noise_ppm * 1e-6


def test_tls_engines_are_isolated(report):
    engines = registry()
    for entry in report["scenarios"]:
        for name in ("tls1", "tls2"):
            result = entry["engines"][name]
            assert result["ok"], result.get("error")
            assert result["version"] == engines[name].tls_version
            assert str(prepare(engines[name])) in result["module_dir"]


@pytest.mark.parametrize("scenario", QUICK)
@pytest.mark.parametrize("engine", ENGINES)
def test_engine_recovers_injected_transit(report, scenario, engine):
    entry = _entry(report, scenario)
    result = entry["engines"][engine]
    assert result["ok"], result.get("error")
    truth, rec = entry["truth"], result["recovery"]

    assert rec["status"] == "recovered", rec
    assert abs(rec["T0_err_min"]) < 0.1 * truth["duration"] * 1440
    assert 0.3 < result["depth"] / truth["depth"] < 1.5
    assert abs(rec["duration_rel_err"]) < 0.5
    assert result["SDE"] > 9


@pytest.mark.parametrize("scenario", QUICK)
@pytest.mark.parametrize("other", ["tls1", "tls2"])
def test_gtls_agrees_with_tls(report, scenario, other):
    entry = _entry(report, scenario)
    pair = next(a for a in entry["agreement"] if a["pair"] == "%s vs gtls" % other)
    gtls = entry["engines"]["gtls"]

    assert pair["same_grid"]
    assert abs(pair["period_rel_diff"]) < 1e-4
    assert abs(pair["T0_diff_min"]) < max(entry["cadence_min"], 0.05 * gtls["duration"] * 1440)
    assert abs(pair["SDE_diff"]) < 0.05 * gtls["SDE"]
    assert abs(pair["depth_rel_diff"]) < 0.05
    assert pair["power_corr"] > 0.99


@pytest.mark.parametrize("scenario", QUICK)
def test_tls_versions_agree(report, scenario):
    pair = next(a for a in _entry(report, scenario)["agreement"] if a["pair"] == "tls2 vs tls1")
    assert pair["same_grid"]
    assert pair["same_peak"]
    assert pair["power_corr"] > 0.99
