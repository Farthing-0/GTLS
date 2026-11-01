"""Orchestration, metrics and reporting of the GTLS / TLS 1.x / TLS 2.x comparison."""
import json
import os
import platform
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from . import engines as engines_mod
from .scenarios import SCENARIOS, search_kwargs
from .simulate import build_dataset

ALIASES = (1 / 3, 1 / 2, 2 / 3, 3 / 2, 2, 3)


def wrap_phase(dt, period):
    """Map a time difference onto [-period/2, period/2)."""
    return (dt + period / 2) % period - period / 2


def cold_warm(timings):
    cold = timings[0]
    warm = float(np.median(timings[1:])) if len(timings) > 1 else cold
    return cold, warm


def recovery(result, truth, rel_tol=0.005):
    """Compare a search result with the injected signal."""
    if truth is None:
        return {"status": "no injection"}
    period, true_period = result["period"], truth["period"]
    if period is None:
        return {"status": "failed"}
    out = {"period_rel_err": (period - true_period) / true_period}
    if abs(out["period_rel_err"]) < rel_tol:
        out["status"] = "recovered"
    else:
        alias = next((r for r in ALIASES if abs(period / (true_period * r) - 1) < rel_tol), None)
        out["status"] = "alias x%s" % ("%.3g" % alias) if alias else "missed"
    if result.get("T0") is not None:
        out["T0_err_min"] = wrap_phase(result["T0"] - truth["T0"], true_period) * 1440
    if result.get("depth") is not None:
        out["depth_rel_err"] = result["depth"] / truth["depth"] - 1
    if result.get("duration") is not None:
        out["duration_rel_err"] = result["duration"] / truth["duration"] - 1
    return out


def _load_periodogram(path):
    with np.load(path) as data:
        return data["periods"], data["power"]


def agreement(a, b):
    """Agreement of engine b with engine a on one light curve."""
    out = {"pair": "%s vs %s" % (b["engine"], a["engine"])}
    if a["period"] is not None and b["period"] is not None:
        out["period_rel_diff"] = (b["period"] - a["period"]) / a["period"]
        out["same_peak"] = abs(out["period_rel_diff"]) < 0.005
        if a.get("T0") is not None and b.get("T0") is not None:
            out["T0_diff_min"] = wrap_phase(b["T0"] - a["T0"], a["period"]) * 1440
    if a.get("SDE") is not None and b.get("SDE") is not None:
        out["SDE_diff"] = b["SDE"] - a["SDE"]
    if a.get("depth") is not None and b.get("depth") is not None and a["depth"] != 0:
        out["depth_rel_diff"] = b["depth"] / a["depth"] - 1
    pa, wa = _load_periodogram(a["arrays"])
    pb, wb = _load_periodogram(b["arrays"])
    out["same_grid"] = bool(pa.size == pb.size and np.allclose(pa, pb, rtol=1e-9, atol=0))
    if not out["same_grid"]:
        order = np.argsort(pb)
        overlap = (pa >= pb.min()) & (pa <= pb.max())
        wb = np.interp(pa[overlap], pb[order], wb[order])
        wa = wa[overlap]
    finite = np.isfinite(wa) & np.isfinite(wb)
    if finite.sum() > 2:
        out["power_corr"] = float(np.corrcoef(wa[finite], wb[finite])[0, 1])
        out["power_max_abs_diff"] = float(np.max(np.abs(wa[finite] - wb[finite])))
    return out


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment(threads):
    cupy = next((_version(p) for p in ("cupy", "cupy-cuda12x", "cupy-cuda11x", "cupy-cuda13x")
                 if _version(p)), None)
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "tls_threads": int(threads or os.cpu_count()),
        "numpy": _version("numpy"),
        "numba": _version("numba"),
        "cupy": cupy,
        "lightkurve": _version("lightkurve"),
        "astropy": _version("astropy"),
    }


def product_name(scenario):
    mission = scenario.mission
    if scenario.real_kepler:
        return "Real " + mission.product
    name = "%d x %s" % (scenario.n_segments, mission.product) if scenario.n_segments > 1 else mission.product
    if scenario.dense_segments:
        segments = [mission.first_segment + i for i in scenario.dense_segments]
        name += ", %s in segments %s" % (scenario.dense_mission.product,
                                         ",".join(str(n) for n in segments))
    return name


def _warmup_data(path):
    """Tiny light curve used once per engine to fill numba / CuPy compilation caches."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 0.02) + 100
    y = 1 + rng.normal(0, 5e-4, t.size)
    y[np.abs(wrap_phase(t - 101.3, 4.1)) < 0.06] -= 3e-3
    np.savez(path, t=t, y=y, dy=np.full(t.size, 5e-4))
    return path


def run_comparison(scenario_names, engine_names=engines_mod.DEFAULT_ENGINES, outdir=None,
                   repeats=1, threads=None, gpu=0, gpus=None, timeout=None, repeat_limit=None,
                   baseline="tls1",
                   tls1_version=engines_mod.TLS1_VERSION, tls2_version=engines_mod.TLS2_VERSION,
                   warmup=True, log=print):
    """Build the datasets, run every engine on every scenario and collect the metrics."""
    available = engines_mod.registry(tls1_version, tls2_version)
    unknown = [e for e in engine_names if e not in available]
    if unknown:
        raise ValueError("Unknown engines %s, choose from %s" % (unknown, list(available)))
    unknown = [s for s in scenario_names if s not in SCENARIOS]
    if unknown:
        raise ValueError("Unknown scenarios %s, choose from %s" % (unknown, list(SCENARIOS)))
    selected = [available[e] for e in engine_names]
    outdir = Path(outdir or engines_mod.TESTS_DIR / ".tls_compare_results"
                  / datetime.now().strftime("%Y%m%d-%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    run_kwargs = dict(threads=threads, gpu=gpu, gpus=gpus, timeout=timeout)

    report = {
        "environment": environment(threads),
        "settings": {"repeats": repeats, "gpu": gpus or gpu, "baseline": baseline, "timeout": timeout,
                     "repeat_limit": repeat_limit,
                     "engines": {e.name: {"label": e.label, "kind": e.kind,
                                          "tls_version": e.tls_version or None,
                                          "options": e.options} for e in selected}},
        "outdir": str(outdir),
        "scenarios": [],
    }

    for engine in selected:
        if engine.kind == "tls":
            log("Preparing %s in %s" % (engine.label, engines_mod.prepare(engine)))
    if warmup:
        data = _warmup_data(outdir / "_warmup.npz")
        for engine in selected:
            start = time.perf_counter()
            res = engines_mod.run(engine, data, outdir / "_warmup", {}, repeats=0, **run_kwargs)
            log("Warm-up %-12s %s (%.1f s)" % (
                engine.name, "ok" if res["ok"] else "FAILED: " + res.get("error", ""),
                time.perf_counter() - start))

    for name in scenario_names:
        scenario = SCENARIOS[name]
        workdir = outdir / name
        dataset = build_dataset(scenario, workdir / "products")
        data_path = dataset.save(workdir / "lightcurve.npz")
        search = search_kwargs(scenario)
        entry = {
            "name": name,
            "description": scenario.description,
            "mission": scenario.mission.name,
            "product": product_name(scenario),
            "lightkurve_class": dataset.lightkurve_class,
            "files": dataset.files,
            "n_raw": dataset.n_raw,
            "n_points": int(dataset.t.size),
            "span": dataset.span,
            "cadence_min": float(np.median(np.diff(dataset.t)) * 1440),
            "noise_ppm": float(np.median(dataset.dy) * 1e6),
            "truth": dataset.truth,
            "search": search,
            "engines": {},
        }
        log("\n[%s] %d points over %.1f d (%s)" % (name, dataset.t.size, dataset.span,
                                                   dataset.lightkurve_class))
        for engine in selected:
            res = engines_mod.run(engine, data_path, workdir, search, repeats=repeats,
                                  repeat_limit=repeat_limit, **run_kwargs)
            if res["ok"]:
                res["cold_s"], res["warm_s"] = cold_warm(res["timings"])
                res["recovery"] = recovery(res, dataset.truth)
                log("  %-12s P=%.5f SDE=%5.1f cold %7.2f s, warm %7.2f s  %s" % (
                    engine.name, res["period"], res["SDE"] or float("nan"), res["cold_s"],
                    res["warm_s"], res["recovery"]["status"]))
            elif res.get("timeout"):
                log("  %-12s TIMEOUT after %g s" % (engine.name, res["timeout"]))
            else:
                log("  %-12s FAILED: %s" % (engine.name, res.get("error")))
            entry["engines"][engine.name] = res
        base = entry["engines"].get(baseline)
        for res in entry["engines"].values():
            if not base:
                continue
            if base["ok"] and res["ok"]:
                res["speedup"] = base["warm_s"] / res["warm_s"]
                res["speedup_cold"] = base["cold_s"] / res["cold_s"]
            elif base.get("timeout") and res["ok"]:
                # The baseline took longer than its time limit: lower bounds
                res["speedup_min"] = base["timeout"] / res["warm_s"]
                res["speedup_cold_min"] = base["timeout"] / res["cold_s"]
        ok = [r for r in entry["engines"].values() if r["ok"]]
        entry["agreement"] = [agreement(a, b) for i, a in enumerate(ok) for b in ok[i + 1:]]
        report["scenarios"].append(entry)

    report["summary"] = summarize(report)
    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    with open(outdir / "report.md", "w") as f:
        f.write(render_markdown(report))
    return report


def summarize(report):
    """Per-engine geometric-mean speedup and recovery count over all scenarios."""
    summary = {}
    for entry in report["scenarios"]:
        for name, res in entry["engines"].items():
            s = summary.setdefault(name, {"runs": 0, "failed": 0, "timeouts": 0, "recovered": 0,
                                          "injected": 0, "speedups": [], "warm_s": []})
            s["runs"] += 1
            if res.get("timeout"):
                s["timeouts"] += 1
                s["warm_s"].append(res["timeout"])
                continue
            if not res["ok"]:
                s["failed"] += 1
                continue
            s["warm_s"].append(res["warm_s"])
            if "speedup" in res:
                s["speedups"].append(res["speedup"])
            if entry["truth"] is not None:
                s["injected"] += 1
                s["recovered"] += res["recovery"]["status"] == "recovered"
    for s in summary.values():
        speedups = s.pop("speedups")
        s["geomean_speedup"] = float(np.exp(np.mean(np.log(speedups)))) if speedups else None
        s["total_warm_s"] = float(np.sum(s.pop("warm_s")))
    return summary


def _fmt(value, spec, scale=1.0):
    return "-" if value is None else format(value * scale, spec)


def _speedup(result, key):
    """Speedup vs the baseline, or its lower bound when the baseline timed out."""
    if key in result:
        return "%.1fx" % result[key]
    if key + "_min" in result:
        return "> %.0fx" % result[key + "_min"]
    return "-"


def render_markdown(report):
    env, settings = report["environment"], report["settings"]
    baseline = settings["baseline"]
    labels = {k: v["label"] for k, v in settings["engines"].items()}
    lines = ["# GTLS vs TLS 1.x vs TLS 2.x", ""]
    lines += ["## Environment", "",
              "| Item | Value |", "| --- | --- |"]
    gpus = sorted({g for e in report["scenarios"] for r in e["engines"].values() if r.get("ok")
                   for g in r.get("hardware", {}).get("gpus", [])})
    versions = {}
    for e in report["scenarios"]:
        for name, r in e["engines"].items():
            if r.get("ok"):
                versions[name] = r["version"]
    rows = [("Date", env["date"]), ("CPU", "%s (%d logical cores)" % (env["cpu"], env["cpu_count"])),
            ("TLS threads (`use_threads`)", env["tls_threads"]), ("GPU", ", ".join(gpus) or "-"),
            ("Python / NumPy / numba", "%s / %s / %s" % (env["python"], env["numpy"], env["numba"])),
            ("CuPy / lightkurve / astropy", "%s / %s / %s" % (env["cupy"], env["lightkurve"], env["astropy"])),
            ("Engines", ", ".join("%s = %s (v%s)" % (k, labels[k], versions.get(k, "?")) for k in labels)),
            ("Timed repeats", "1 cold + %d warm%s" % (
                settings["repeats"],
                ", no repeats for runs > %g s" % settings["repeat_limit"]
                if settings.get("repeat_limit") is not None else "")),
            ("Time limit per engine and scenario",
             "%g s" % settings["timeout"] if settings.get("timeout") else "none")]
    lines += ["| %s | %s |" % row for row in rows] + [""]

    lines += ["## Datasets (lightkurve-prepared)", "",
              "| Scenario | Product | lightkurve class | Cadences raw → searched | Span (d) | "
              "Cadence (min) | σ (ppm) | Injected P (d) | Depth (ppm) | Duration (h) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in report["scenarios"]:
        tr = e["truth"] or {}
        lines.append("| %s | %s | %s | %d → %d | %.1f | %.1f | %.0f | %s | %s | %s |" % (
            e["name"], e["product"], e["lightkurve_class"], e["n_raw"], e["n_points"], e["span"],
            e["cadence_min"], e["noise_ppm"], _fmt(tr.get("period"), ".4f"),
            _fmt(tr.get("depth"), ".0f", 1e6), _fmt(tr.get("duration"), ".2f", 24)))
    lines.append("")

    lines += ["## Speed", "",
              "Cold = first `power()` call in a fresh process (imports done; includes JIT cache "
              "loading, CUDA context / worker-pool start-up). Warm = median of later calls; "
              "(parentheses) = not repeated, the cold time is used. "
              "Speedup = %s warm time / engine warm time." % baseline, "",
              "| Scenario | Engine | Periods | Cold (s) | Warm (s) | Speedup (warm) | Speedup (cold) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in report["scenarios"]:
        for name, r in e["engines"].items():
            if r.get("timeout"):
                lines.append("| %s | %s | - | > %g | > %g | timeout | timeout |" % (
                    e["name"], name, r["timeout"], r["timeout"]))
                continue
            if not r["ok"]:
                lines.append("| %s | %s | FAILED: %s | | | | |" % (e["name"], name, r.get("error", "")[:80]))
                continue
            lines.append("| %s | %s | %d | %.2f | %s | %s | %s |" % (
                e["name"], name, r["n_periods"], r["cold_s"],
                "%.2f" % r["warm_s"] if len(r["timings"]) > 1 else "(%.2f)" % r["warm_s"],
                _speedup(r, "speedup"), _speedup(r, "speedup_cold")))
    lines.append("")

    lines += ["## Search results", "",
              "Depth is the fractional depth (TLS `1 - depth`). Errors are relative to the injected "
              "signal; ΔT0 is wrapped to the true period.", "",
              "| Scenario | Engine | Period (d) | ΔP/P | ΔT0 (min) | Depth (ppm) | Duration (h) | "
              "SDE | SNR | Deterministic | Status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in report["scenarios"]:
        for name, r in e["engines"].items():
            if not r["ok"]:
                continue
            rec = r["recovery"]
            lines.append("| %s | %s | %.5f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                e["name"], name, r["period"], _fmt(rec.get("period_rel_err"), ".1e"),
                _fmt(rec.get("T0_err_min"), ".1f"), _fmt(r["depth"], ".0f", 1e6),
                _fmt(r["duration"], ".2f", 24), _fmt(r["SDE"], ".1f"), _fmt(r["snr"], ".1f"),
                "yes" if r["deterministic"] else "NO", rec["status"]))
    lines.append("")

    lines += ["## Cross-engine agreement", "",
              "| Scenario | Pair | Same period grid | ΔP/P | ΔT0 (min) | ΔSDE | Δdepth | "
              "Periodogram r | max |Δpower| |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in report["scenarios"]:
        for a in e["agreement"]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                e["name"], a["pair"], "yes" if a["same_grid"] else "no",
                _fmt(a.get("period_rel_diff"), ".1e"), _fmt(a.get("T0_diff_min"), ".2f"),
                _fmt(a.get("SDE_diff"), "+.2f"), _fmt(a.get("depth_rel_diff"), "+.1%"),
                _fmt(a.get("power_corr"), ".4f"), _fmt(a.get("power_max_abs_diff"), ".2f")))
    lines.append("")

    lines += ["## Summary", "",
              "| Engine | Runs | Failed | Timeouts | Recovered / injected | Total warm time (s) | "
              "Geometric-mean speedup vs %s |" % baseline,
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in report["summary"].items():
        lines.append("| %s | %d | %d | %d | %d / %d | %s%.2f | %s |" % (
            name, s["runs"], s["failed"], s["timeouts"], s["recovered"], s["injected"],
            "> " if s["timeouts"] else "", s["total_warm_s"],
            _fmt(s["geomean_speedup"], ".1f") + ("x" if s["geomean_speedup"] else "")))
    lines += ["", "Timed-out runs count with their time limit in the total time; the geometric "
              "mean only uses scenarios where both the engine and %s finished." % baseline, ""]
    return "\n".join(lines)


def plot_periodograms(report):
    """One PNG per scenario with the periodogram of every engine."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for e in report["scenarios"]:
        ok = {n: r for n, r in e["engines"].items() if r["ok"]}
        if not ok:
            continue
        fig, axes = plt.subplots(len(ok), 1, figsize=(10, 2.2 * len(ok)), sharex=True, squeeze=False)
        for ax, (name, r) in zip(axes[:, 0], ok.items()):
            periods, power = _load_periodogram(r["arrays"])
            ax.plot(periods, power, lw=0.6, color="k")
            if e["truth"]:
                ax.axvline(e["truth"]["period"], color="tab:red", alpha=0.3, lw=3)
            ax.set_ylabel("SDE")
            ax.set_title("%s: P=%.5f d, SDE=%.1f, warm %.2f s" % (name, r["period"], r["SDE"], r["warm_s"]),
                         fontsize=9, loc="left")
        axes[-1, 0].set_xlabel("Period (days)")
        axes[-1, 0].set_xscale("log")
        fig.suptitle(e["name"])
        fig.tight_layout()
        path = Path(report["outdir"]) / ("%s_periodograms.png" % e["name"])
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths
