"""Engine registry, isolated installation of TLS releases, and subprocess execution."""
import json
import os
import shutil
import signal
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
    git_ref: str = ""  # GTLS from this git revision instead of the working tree


def registry(tls1_version=TLS1_VERSION, tls2_version=TLS2_VERSION):
    """Available engines. ``tls2`` uses the TLS 2 default backend (``fused-pl``)."""
    engines = [
        Engine("gtls", "gtls", "GTLS (GPU)"),
        Engine("gtls-main", "gtls", "GTLS main (GPU)", git_ref="main"),
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


def export_gtls(ref, env_dir=ENV_DIR):
    """Export src/ of GTLS at git revision ref; returns the directory to put on sys.path."""
    repo = str(TESTS_DIR.parent)
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "--verify", ref + "^{commit}"],
                                check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unknown git revision %r" % ref) from error
    target = Path(env_dir) / ("gtls-%s" % commit[:12])
    if (target / "src" / "gputls" / "__init__.py").exists():
        return target / "src"
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=target.name + ".", dir=target.parent))
    try:
        archive = subprocess.run(["git", "-C", repo, "archive", "--format=tar", commit, "src"],
                                 check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", str(staging)], input=archive.stdout, check=True)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target / "src"


def prepare(engine):
    """Directory to import the engine from (installing / exporting it if needed); None for the
    GTLS working tree."""
    if engine.kind == "tls":
        return install_tls(engine.tls_version)
    return export_gtls(engine.git_ref) if engine.git_ref else None


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
        timeout=None, repeat_limit=None, gtls_src=REPO_SRC):
    """Run ``engine`` on the light curve stored in ``data_path`` (npz with t, y, dy).

    The first call in the fresh process is reported as the cold time, the following
    ``repeats`` calls as warm times; a call longer than ``repeat_limit`` seconds is not
    repeated. Returns the worker's result dictionary.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stem = workdir / engine.name
    site = prepare(engine)
    if engine.kind == "gtls" and site is not None:
        gtls_src = site
    cfg = {
        "engine": engine.name,
        "kind": engine.kind,
        "site": str(site) if engine.kind == "tls" else None,
        "gtls_src": str(gtls_src) if gtls_src else None,
        "data": str(data_path),
        "repeats": int(repeats),
        "repeat_limit": repeat_limit,
        "power_kwargs": power_kwargs(engine, search, threads, gpu, gpus),
        "output": str(stem) + ".json",
        "arrays": str(stem) + "_periodogram.npz",
    }
    with open(str(stem) + "_config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    env = dict(os.environ, PYTHONWARNINGS="ignore", TQDM_DISABLE="1")
    if os.path.exists(cfg["output"]):
        os.remove(cfg["output"])
    # Own process group, so a timeout also stops the worker pools of TLS
    proc = subprocess.Popen([sys.executable, str(WORKER), str(stem) + "_config.json"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"engine": engine.name, "ok": False, "timeout": float(timeout),
                "error": "timeout after %g s" % timeout, "power_kwargs": cfg["power_kwargs"]}
    try:
        with open(cfg["output"]) as f:
            result = json.load(f)
    except (OSError, ValueError):
        result = {"engine": engine.name, "ok": False,
                  "error": "worker crashed (exit %d): %s" % (proc.returncode, stderr[-2000:])}
    result["arrays"] = cfg["arrays"]
    result["power_kwargs"] = cfg["power_kwargs"]
    return result
