"""Run one engine on one light curve in an isolated process.

Usage: python worker.py CONFIG.json

Kept free of framework imports so that the import cost, numba JIT cache loading, CUDA
context creation and worker-pool start-up of each engine are measured in a fresh process,
and so that two versions of ``transitleastsquares`` never share an interpreter.
"""
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

SCALARS = ("period", "period_uncertainty", "T0", "duration", "depth", "SDE", "snr",
           "transit_count", "distinct_transit_count")


def _load_engine(cfg):
    if cfg["kind"] == "tls":
        sys.path.insert(0, cfg["site"])
        import transitleastsquares
        from transitleastsquares import transitleastsquares as Model
        from transitleastsquares import version

        module_dir = os.path.dirname(os.path.abspath(transitleastsquares.__file__))
        if not module_dir.startswith(os.path.abspath(cfg["site"])):
            raise ImportError("transitleastsquares was imported from %s, not %s"
                              % (module_dir, cfg["site"]))
        return Model, version.TLS_VERSIONING, module_dir
    if cfg.get("gtls_src"):
        sys.path.insert(0, cfg["gtls_src"])
    import gputls
    from gputls import gtls as Model

    module_dir = os.path.dirname(os.path.abspath(gputls.__file__))
    if cfg.get("gtls_src") and not module_dir.startswith(os.path.abspath(cfg["gtls_src"])):
        raise ImportError("gputls was imported from %s, not %s" % (module_dir, cfg["gtls_src"]))
    return Model, gputls.__version__, module_dir


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _summary(result, kind):
    out = {key: _float(getattr(result, key, None)) for key in SCALARS}
    # TLS reports the in-transit flux level, GTLS the fractional depth
    if kind == "tls" and out["depth"] is not None:
        out["depth"] = 1.0 - out["depth"]
    return out


def _hardware(cfg):
    info = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    if cfg["kind"] == "gtls":
        import cupy

        devices = cfg["power_kwargs"].get("GPUDeviceIDs") or [cfg["power_kwargs"].get("GPUDeviceID", 0)]
        info["gpus"] = [cupy.cuda.runtime.getDeviceProperties(d)["name"].decode() for d in devices]
    return info


def main(path):
    with open(path) as f:
        cfg = json.load(f)
    out = {"engine": cfg["engine"], "ok": False}
    try:
        t_import = time.perf_counter()
        Model, version, module_dir = _load_engine(cfg)
        out["import_s"] = time.perf_counter() - t_import
        out["version"] = version
        out["module_dir"] = module_dir

        data = np.load(cfg["data"])
        t, y, dy = data["t"], data["y"], data["dy"]
        timings, summaries = [], []
        for _ in range(1 + cfg["repeats"]):
            start = time.perf_counter()
            result = Model(t, y, dy, verbose=False).power(**cfg["power_kwargs"])
            timings.append(time.perf_counter() - start)
            summaries.append(_summary(result, cfg["kind"]))
            # Slow runs are not repeated: their start-up cost is negligible
            limit = cfg.get("repeat_limit")
            if limit is not None and timings[-1] > limit:
                break

        periods = np.ma.filled(np.ma.asarray(result.periods, dtype=float), np.nan)
        power = np.ma.filled(np.ma.asarray(result.power, dtype=float), np.nan)
        np.savez(cfg["arrays"], periods=periods, power=power)
        out.update(summaries[-1])
        out.update({
            "ok": True,
            "timings": timings,
            "n_periods": int(len(periods)),
            "period_grid": [float(np.nanmin(periods)), float(np.nanmax(periods))],
            "deterministic": all(s == summaries[0] for s in summaries),
            "hardware": _hardware(cfg),
        })
    except Exception as error:
        out["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        out["traceback"] = traceback.format_exc()
    with open(cfg["output"], "w") as f:
        json.dump(out, f, indent=1)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
