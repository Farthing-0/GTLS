"""Time per window of the searchChi2 inner loop for G lanes per window, as a function of L.

Usage: python bench/window_lanes/bench.py [--device 0] [--n 71511] [--periods 256] [--margin 0.125]
                                         [--p 0.5] [--out results.csv]
"""
import argparse, csv, pathlib, sys
import numpy as np, cupy as cp

HERE = pathlib.Path(__file__).parent
GS = [1, 2, 4, 8, 16, 32]
LS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]

# Unified L1 + shared memory per SM by compute capability (CUDA C Programming Guide,
# "Technical Specifications per Compute Capability"); shared memory carveout steps (KiB).
UNIFIED_L1_KIB = {"70": 128, "72": 128, "75": 96, "80": 192, "86": 128, "87": 192, "89": 128,
                  "90": 256, "100": 256, "120": 128}
CARVEOUTS_KIB = [0, 8, 16, 32, 64, 100, 132, 164, 196, 228]


def unified_l1(cc, attrs):
    if cc in UNIFIED_L1_KIB:
        return UNIFIED_L1_KIB[cc] * 1024
    return attrs["MaxSharedMemoryPerMultiprocessor"] + 28 * 1024  # max carveout + the L1 minimum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n", type=int, default=71511, help="points per light curve (tSize)")
    ap.add_argument("--periods", type=int, default=256, help="periods per batch (grid y)")
    ap.add_argument("--margin", type=float, default=0.125)
    ap.add_argument("--p", type=float, default=0.5, help="fraction of windows passing the depth cut")
    ap.add_argument("--fp32div", action="store_true", help="mean depth without the FP64 division")
    ap.add_argument("--smem", type=int, default=0,
                    help="dynamic shared memory per block (bytes), to emulate a smaller L1 / occupancy")
    ap.add_argument("--l1", type=int, default=None, help="unified L1+shared memory per SM (bytes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cp.cuda.Device(a.device).use()
    mod = cp.RawModule(code=(HERE / "lanes.cu").read_text(), options=("-std=c++14",))
    suffix = "_f32" if a.fp32div else ""
    kern = {g: mod.get_function(f"search_G{g}{suffix}") for g in GS}

    # Resident threads per SM (occupancy API) and L1 left beside the shared memory carveout
    from cupy.cuda import driver
    attrs = cp.cuda.Device(a.device).attributes
    l1_total = a.l1 or unified_l1(cp.cuda.Device(a.device).compute_capability, attrs)
    occ = {g: driver.occupancyMaxActiveBlocksPerMultiprocessor(kern[g].kernel.ptr, 256, a.smem) for g in GS}
    threads_sm = 256 * occ[1]
    smem_sm = occ[1] * (a.smem + 256 + 1024)  # + static warpMin + 1 KiB reserved per block
    carve = min(c for c in CARVEOUTS_KIB if c * 1024 >= smem_sm) * 1024 if smem_sm else 0
    c_l1 = l1_total - carve
    print(f"# threads/SM={threads_sm} unified={l1_total//1024} KiB carveout={carve//1024} KiB "
          f"L1~{c_l1//1024} KiB  G_cap = {threads_sm}*8*S/{c_l1} = S/{c_l1/(8*threads_sm):.1f}")

    rng = np.random.default_rng(0)
    Lmax = max(LS)
    P = a.n + Lmax
    r = rng.normal(0, 1e-3, (a.periods, P)).astype(np.float32)
    cum = cp.asarray(np.cumsum(r.astype(np.float64), axis=1))
    w = np.full_like(r, 1e6)
    W = cp.asarray(np.stack([w * r, w], axis=-1))
    best = cp.empty(a.periods, dtype=cp.uint64)
    skip = int(1 / a.margin) if a.margin > 0 else 0
    # depth cut giving pass fraction p for a window of L N(0, s) points: quantile of the mean
    from scipy.stats import norm
    rows = []
    name = cp.cuda.runtime.getDeviceProperties(a.device)["name"].decode()
    for L in LS:
        s = cp.asarray((0.5 * (1 - np.cos(2 * np.pi * (np.arange(L) + 0.5) / L))).astype(np.float32))
        stride = L // skip if (skip and L > skip) else 1
        nWin = -(-a.n // stride)
        dmin = np.float32(norm.ppf(1 - a.p) * 1e-3 / np.sqrt(L)) if 0 < a.p < 1 else np.float32(-1 if a.p >= 1 else 1)
        line = {"device": name, "n": a.n, "periods": a.periods, "margin": a.margin, "p": a.p, "L": L,
                "stride": stride}
        for g in GS:
            grid = ((nWin * g + 255) // 256, a.periods)
            args = (best, cum, W, np.float64(0), s, np.float32(1), np.int32(L), np.int32(stride),
                    np.int32(nWin), np.int32(P), dmin)
            kern[g](grid, (256,), args, shared_mem=a.smem)
            ev = [cp.cuda.Event() for _ in range(2)]
            reps = max(1, int(2e8 // (a.periods * nWin * max(L, 32))))
            times = []
            for _ in range(5):
                ev[0].record()
                for _ in range(reps):
                    best.fill(0xFFFFFFFFFFFFFFFF)
                    kern[g](grid, (256,), args, shared_mem=a.smem)
                ev[1].record(); ev[1].synchronize()
                times.append(cp.cuda.get_elapsed_time(*ev) / reps)
            line[f"G{g}"] = np.median(times) * 1e6 / (a.periods * nWin)  # ns per window
        gbest = min(GS, key=lambda g: line[f"G{g}"])
        gcap = threads_sm * 8 * stride / c_l1
        gpred = next((g for g in GS if g >= gcap), 32)
        line.update(best=gbest, Gcap=gcap, pred=gpred, loss=line[f"G{gpred}"] / line[f"G{gbest}"] - 1)
        print(f"L={L:5d} stride={stride:4d} " + " ".join(f"G{g}={line[f'G{g}']:8.3f}" for g in GS)
              + f"  best G={gbest:2d} Gcap={gcap:6.2f} pred={gpred:2d} loss={line['loss']:+.0%}", flush=True)
        rows.append(line)
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

if __name__ == "__main__":
    main()
