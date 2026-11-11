"""End to end: searchChi2 with G lanes per window chosen per duration, vs the current kernel.

Patches the searchChi2 launch of the installed/source gputls (no change to the package) and
compares run time and results of: the current kernel (orig), one warp per window for L >= 256
(warp256), and G = smallest power of 2 >= S / 8 (S/8). Usage: python bench/window_lanes/end_to_end.py
"""
import pathlib, sys, time
import numpy as np, cupy as cp
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from gputls import core
from gputls.main import gtls


def make(days, cadence_min, seed=0):
    """White-noise light curve with a 1000 ppm box transit (P = 7.3 d)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, cadence_min / 1440.)
    y = 1 + rng.normal(0, 5e-4, t.size)
    ph = ((t - 2.1 + 7.3 / 2) % 7.3) - 7.3 / 2
    y[np.abs(ph) < 0.06] -= 1e-3
    return t, y, np.full_like(t, 5e-4)

SRC = r'''
__device__ __forceinline__ unsigned int orderedKey(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
template <int G>
__device__ void body(unsigned long long* best, const double* cumResidual, const float2* patchedWeighted,
    const double* base, const float* signal, const float* overshoot, const int* widths, const int* selected,
    const int* offsets, const int* strides, const bool* periodMask, int yBegin, int yEnd, int nDurations,
    int tSize, int patchedSize, int maxWidth, float transitDepthMin) {
    int z = blockIdx.y;
    int sub = threadIdx.x % G;
    int g = offsets[yBegin] + (blockIdx.x * blockDim.x + threadIdx.x) / G;
    unsigned long long key = 0xffffffffffffffffull;
    float ar = 0.f, a2 = 0.f, meanDepth = 0.f; bool active = false; int y = yBegin, start = 0, d = 0;
    if (g < offsets[yEnd]) {
        while (offsets[y + 1] <= g) y++;
        start = (g - offsets[y]) * strides[y];
        d = selected[y];
        int L = widths[d];
        if (periodMask[(long long)z * nDurations + d]) {
            const double* C = cumResidual + (long long)z * patchedSize;
            double windowSum = C[start + L - 1] - (start > 0 ? C[start - 1] : 0.0);
            meanDepth = (float)(windowSum / L);
            active = meanDepth > transitDepthMin;
            if (active) {
                const float2* x = patchedWeighted + (long long)z * patchedSize + start;
                const float* s = signal + (long long)d * maxWidth;
                for (int i = sub; i < L; i += G) { float si = s[i]; float2 v = x[i];
                    ar = fmaf(si, v.x, ar); a2 = fmaf(si * si, v.y, a2); }
            }
        }
    }
    #pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) {
        ar += __shfl_xor_sync(0xffffffff, ar, o); a2 += __shfl_xor_sync(0xffffffff, a2, o); }
    if (active && sub == 0) {
        float k = meanDepth * overshoot[d] * 2.0f;
        float chi2 = (float)(base[z] + (double)(k * (k * a2 - 2.0f * ar)));
        key = ((unsigned long long)orderedKey(chi2) << 32) | (unsigned int)(y * tSize + start);
    }
    for (int o = 16; o > 0; o >>= 1) { unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
        key = other < key ? other : key; }
    __shared__ unsigned long long warpMin[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpMin[warp] = key;
    __syncthreads();
    if (warp == 0) {
        key = lane < (int)(blockDim.x >> 5) ? warpMin[lane] : 0xffffffffffffffffull;
        for (int o = 16; o > 0; o >>= 1) { unsigned long long other = __shfl_down_sync(0xffffffff, key, o);
            key = other < key ? other : key; }
        if (lane == 0 && key != 0xffffffffffffffffull) atomicMin(best + z, key);
    }
}
#define DEF(G) extern "C" __global__ void searchChi2_G##G(unsigned long long* b, const double* c, const float2* w, \
  const double* base, const float* s, const float* ov, const int* wd, const int* sel, const int* off, const int* st, \
  const bool* m, int y0, int y1, int nD, int tS, int pS, int mW, float dmin) { \
  body<G>(b, c, w, base, s, ov, wd, sel, off, st, m, y0, y1, nD, tS, pS, mW, dmin); }
DEF(1) DEF(2) DEF(4) DEF(8) DEF(16) DEF(32)
'''
mod = cp.RawModule(code=SRC, options=("-std=c++14",))
K = {g: mod.get_function(f"searchChi2_G{g}") for g in [1, 2, 4, 8, 16, 32]}
RULE = {"mode": "orig"}

def choose(L, S):
    if RULE["mode"] == "warp256": return 32 if L >= 256 else 1
    if RULE["mode"] == "S/8": return next((g for g in K if g >= S / 8), 32)

class Mod:
    def __init__(s, m): s.m = m
    def get_function(s, name):
        f = s.m.get_function(name)
        if name != "searchChi2" or RULE["mode"] == "orig": return f
        def call(grid, block, args):
            (best, C, W, base, sig, ovs, widths, selected, offsets, strides, mask,
             nSel, nDur, tSize, pSize, maxW, dmin) = args
            sel = selected.get(); off = offsets.get(); st = strides.get(); L = DUR[sel]
            Gs = [choose(l, s_) for l, s_ in zip(L, st)]
            y0 = 0
            while y0 < len(sel):
                y1 = y0
                while y1 < len(sel) and Gs[y1] == Gs[y0]: y1 += 1
                G = Gs[y0]; n = int(off[y1] - off[y0])
                K[G](((n * G + 255) // 256, grid[1]), (256,), (best, C, W, base, sig, ovs, widths, selected,
                     offsets, strides, mask, np.int32(y0), np.int32(y1), nDur, tSize, pSize, maxW, dmin))
                y0 = y1
        return call
_load = core.load_module; core.load_module = lambda: Mod(_load())
_init = core.GPUSearch.__init__
def init(self, *a, **k):
    global DUR; _init(self, *a, **k); DUR = self.durations
core.GPUSearch.__init__ = init

for days, cad, name in [(27, 2, "TESS 2min 27d"), (360, 29.4, "Kepler 4q"), (1460, 29.4, "Kepler 17q"),
                        (27, 1/3, "TESS 20s 27d"), (54, 2, "TESS 2min 54d")]:
    t, y, dy = make(days, cad)
    res = {}
    for mode in ["orig", "warp256", "S/8"]:
        RULE["mode"] = mode
        gtls(t, y, dy, verbose=False).power(GPUDeviceIDs=[0], fast=True) if days < 100 else None
        ts = []
        for _ in range(2 if days < 1000 else 1):
            t0 = time.perf_counter(); r = gtls(t, y, dy, verbose=False).power(GPUDeviceIDs=[0]); ts.append(time.perf_counter() - t0)
        res[mode] = (min(ts), r)
    a = res["orig"][1]
    line = f"{name:15s} N={t.size:7d} " + " ".join(f"{m}={res[m][0]:6.2f}s" for m in res)
    for m in ["warp256", "S/8"]:
        b = res[m][1]
        rel = np.max(np.abs(np.asarray(a.chi2) - np.asarray(b.chi2)) / np.asarray(a.chi2))
        same = (a.period == b.period) and (a.T0 == b.T0 if hasattr(a, "T0") else True)
        line += f" | {m}: same period={same} SDE {b.SDE:.4f} vs {a.SDE:.4f} dchi2={rel:.0e}"
    print(line, flush=True)
