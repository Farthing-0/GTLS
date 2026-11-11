"""Run gputls on a fixed set of light curves; save the results and per-kernel GPU times, or
compare them with a saved run.

    python bench/window_lanes/regress.py save ref.npz      # e.g. on main
    python bench/window_lanes/regress.py compare ref.npz   # on the branch under test
"""
import pathlib, sys, time
import numpy as np, cupy as cp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gputls import core
from gputls.main import gtls

KERNEL_TIME = {}


class TimedModule:
    """Wrap the RawModule so every kernel launch is timed with CUDA events."""
    def __init__(self, m):
        self.m = m

    def get_function(self, name):
        k = self.m.get_function(name)

        def call(*a, **kw):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record(); k(*a, **kw); e1.record(); e1.synchronize()
            key = "searchChi2*" if name.startswith("searchChi2") else name
            KERNEL_TIME[key] = KERNEL_TIME.get(key, 0) + cp.cuda.get_elapsed_time(e0, e1) / 1e3
        return call


_load = core.load_module
core.load_module = lambda: TimedModule(_load())


def box(days, cadence_min, offset=0.0, seed=0, flux_offset=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, cadence_min / 1440.) + offset
    y = 1 + flux_offset + rng.normal(0, 5e-4, t.size)
    ph = ((t - offset - 2.1 + 7.3 / 2) % 7.3) - 7.3 / 2
    y[np.abs(ph) < 0.06] -= 1e-3
    return t, y, np.full_like(t, 5e-4)


def batman_lc():
    import batman
    rng = np.random.default_rng(1)
    t = np.linspace(1, 101, 100 * 48)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.u, p.limb_dark = 3.3, 7.7, 0.1, 20, 90, 0, 90, [0.4, 0.4], "quadratic"
    y = batman.TransitModel(p, t).light_curve(p) + rng.normal(0, 5e-4, t.size)
    return t, y, None


def kepler_q3():
    from astropy.io import fits
    from gputls.helpers import cleaned_array
    with fits.open(ROOT / "tests" / "data" / "kplr011403044-2009350155506_llc.fits") as h:
        t, y, dy = cleaned_array(h[1].data["TIME"], h[1].data["PDCSAP_FLUX"], h[1].data["PDCSAP_FLUX_ERR"])
    m = np.mean(y)
    return t, y / m, dy / m


CASES = {
    "batman_100d": batman_lc,
    "kepler_q3_real": kepler_q3,
    "tess2min_27d": lambda: box(27, 2),
    "tess2min_54d": lambda: box(54, 2),
    "kepler_4q": lambda: box(360, 29.4),
    "kepler_4q_bjd": lambda: box(360, 29.4, offset=2454833.0),
    "tess20s_27d": lambda: box(27, 1 / 3),
    "kepler_17q": lambda: box(1460, 29.4),
    # badly normalized flux: large residual prefix sums (stress for their precision)
    "kepler_4q_flux-1e-3": lambda: box(360, 29.4, flux_offset=-1e-3),
    "tess20s_flux-1e-2": lambda: box(27, 1 / 3, flux_offset=-1e-2),
}
FIELDS = ["periods", "raw_chi2", "chi2", "power", "period", "T0", "duration", "depth", "SDE", "snr", "snrFit",
          "transitTimes", "durationPoints"]


def run(only=None):
    out = {}
    for name, make in CASES.items():
        if only and name not in only:
            continue
        t, y, dy = make()
        gtls(t, y, dy, verbose=False).power(GPUDeviceIDs=[0], fast=True) if t.size < 50000 else None
        KERNEL_TIME.clear()
        t0 = time.perf_counter()
        r = gtls(t, y, dy, verbose=False).power(GPUDeviceIDs=[0])
        wall = time.perf_counter() - t0
        for f in FIELDS:
            out[f"{name}/{f}"] = np.ma.filled(np.ma.asarray(getattr(r, f), dtype=float), np.nan)
        out[f"{name}/wall"] = np.array(wall)
        for k, v in KERNEL_TIME.items():
            out[f"{name}/k/{k}"] = np.array(v)
        ks = " ".join(f"{k}={v:.3f}" for k, v in sorted(KERNEL_TIME.items(), key=lambda x: -x[1]))
        print(f"{name:15s} N={t.size:7d} wall={wall:6.2f}s  {ks}", flush=True)
    return out


def compare(ref, new):
    names = sorted({k.split("/")[0] for k in new})
    print(f"\n{'case':15s} {'wall ref->new':>16s} {'search':>16s} {'max rel dchi2':>14s} "
          f"{'period':>7s} {'T0':>9s} {'dur':>9s} {'depth':>9s} {'SDE':>9s}")
    for n in names:
        if f"{n}/wall" not in ref:
            continue
        g = lambda d, f: d[f"{n}/{f}"]
        rel = np.max(np.abs(g(new, "raw_chi2") - g(ref, "raw_chi2")) / np.abs(g(ref, "raw_chi2")))
        same = lambda f: "same" if np.array_equal(g(new, f), g(ref, f)) else f"{np.max(np.abs(g(new, f) - g(ref, f))):.1e}"
        ks = lambda d: sum(float(v) for k, v in d.items() if k.startswith(f"{n}/k/searchChi2"))
        print(f"{n:15s} {float(g(ref, 'wall')):6.2f}->{float(g(new, 'wall')):6.2f}s "
              f"{ks(ref):6.2f}->{ks(new):6.2f}s {rel:14.1e} {same('period'):>7s} {same('T0'):>9s} "
              f"{same('duration'):>9s} {same('depth'):>9s} {same('SDE'):>9s}")
        kr = {k.split('/k/')[1]: float(v) for k, v in ref.items() if k.startswith(f"{n}/k/")}
        kn = {k.split('/k/')[1]: float(v) for k, v in new.items() if k.startswith(f"{n}/k/")}
        print("    " + " ".join(f"{k}:{kr.get(k, 0):.3f}->{kn.get(k, 0):.3f}" for k in sorted(set(kr) | set(kn))))


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    res = run(only)
    if mode == "save":
        np.savez(path, **res)
    else:
        compare(dict(np.load(path)), res)
