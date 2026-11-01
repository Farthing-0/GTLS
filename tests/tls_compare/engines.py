"""Engine registry, isolated installation of TLS releases, and subprocess execution."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_SRC = TESTS_DIR.parent / "src"
WORKER = Path(__file__).resolve().parent / "worker.py"

# Last TLS 1.x release on PyPI and the current TLS 2.x release
TLS1_VERSION = os.environ.get("TLS1_VERSION", "1.32")
TLS2_VERSION = os.environ.get("TLS2_VERSION", "2.0.4")
ENV_DIR = Path(os.environ.get("TLS_COMPARE_ENV_DIR", TESTS_DIR / ".tls_envs"))


@dataclass(frozen=True)
class Engine:
    name: str
    kind: str  # "gtls" or "tls"
    label: str
    tls_version: str = ""
    options: dict = field(default_factory=dict)  # engine specific power() keyword arguments


def registry(tls1_version=TLS1_VERSION, tls2_version=TLS2_VERSION):
    """Available engines. ``tls2`` uses the TLS 2 default backend (``fused-pl``)."""
    engines = [
        Engine("gtls", "gtls", "GTLS (GPU)"),
        Engine("tls1", "tls", "TLS %s" % tls1_version, tls1_version),
        Engine("tls2", "tls", "TLS %s" % tls2_version, tls2_version),
        Engine("tls2-fused", "tls", "TLS %s fused" % tls2_version, tls2_version, {"backend": "fused"}),
        Engine("tls2-numba", "tls", "TLS %s numba" % tls2_version, tls2_version, {"backend": "numba"}),
    ]
    return {e.name: e for e in engines}


DEFAULT_ENGINES = ("gtls", "tls1", "tls2")


def install_tls(version, env_dir=ENV_DIR, python=sys.executable):
    """Install transitleastsquares==version into its own directory (without dependencies).

    The runtime dependencies (numpy, numba, tqdm, batman for TLS 1.x) come from the
    current environment. Returns the directory to put on ``sys.path``.
    """
    target = Path(env_dir) / ("transitleastsquares-%s" % version)
    if (target / "transitleastsquares" / "__init__.py").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
    try:
        subprocess.run(
            [python, "-m", "pip", "install", "--quiet", "--no-deps", "--no-compile",
             "--target", str(staging), "transitleastsquares==%s" % version],
            check=True, capture_output=True, text=True)
        staging.rename(target)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Installing transitleastsquares==%s failed:\n%s"
                           % (version, error.stderr.strip())) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def prepare(engine):
    """Return the site directory of a TLS engine (installing it if needed), else None."""
    return install_tls(engine.tls_version) if engine.kind == "tls" else None


def power_kwargs(engine, search, threads=None, gpu=0, gpus=None):
    """Keyword arguments of ``power()``: the shared search parameters plus engine options."""
    kwargs = dict(search)
    if engine.kind == "tls":
        kwargs.update(show_progress_bar=False, use_threads=int(threads or os.cpu_count()))
    elif gpus:
        kwargs["GPUDeviceIDs"] = list(gpus)
    else:
        kwargs["GPUDeviceID"] = int(gpu)
    kwargs.update(engine.options)
    return kwargs


def run(engine, data_path, workdir, search, repeats=1, threads=None, gpu=0, gpus=None,
        timeout=None, gtls_src=REPO_SRC):
    """Run ``engine`` on the light curve stored in ``data_path`` (npz with t, y, dy).

    The first call in the fresh process is reported as the cold time, the following
    ``repeats`` calls as warm times. Returns the worker's result dictionary.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = workdir / engine.name
    cfg = {
        "engine": engine.name,
        "kind": engine.kind,
        "site": str(prepare(engine)) if engine.kind == "tls" else None,
        "gtls_src": str(gtls_src) if gtls_src else None,
        "data": str(data_path),
        "repeats": int(repeats),
        "power_kwargs": power_kwargs(engine, search, threads, gpu, gpus),
        "output": str(stem) + ".json",
        "arrays": str(stem) + "_periodogram.npz",
    }
    with open(str(stem) + "_config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    env = dict(os.environ, PYTHONWARNINGS="ignore", TQDM_DISABLE="1")
    try:
        proc = subprocess.run([sys.executable, str(WORKER), str(stem) + "_config.json"],
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"engine": engine.name, "ok": False, "error": "timeout after %ss" % timeout}
    try:
        with open(cfg["output"]) as f:
            result = json.load(f)
    except (OSError, ValueError):
        result = {"engine": engine.name, "ok": False,
                  "error": "worker crashed (exit %d): %s" % (proc.returncode, proc.stderr[-2000:])}
    result["arrays"] = cfg["arrays"]
    result["power_kwargs"] = cfg["power_kwargs"]
    return result
