# GTLS vs TLS 1.x vs TLS 2.x comparison framework

Benchmarks the speed and the search results of three engines on the same light curves:

| Engine | What it runs |
| --- | --- |
| `gtls` | GTLS from this repository (`src/`), on the GPU |
| `tls1` | `transitleastsquares` 1.32, the last 1.x release |
| `tls2` | `transitleastsquares` 2.0.4 with its default `fused-pl` backend |
| `tls2-fused`, `tls2-numba` | TLS 2 with the exact `fused` backend / the legacy 1.x-style `numba` backend (opt-in) |

Both TLS releases import as `transitleastsquares`, so each is installed with
`pip install --no-deps --target` into its own directory under `tests/.tls_envs/`.
Every engine then runs in a fresh subprocess (`worker.py`) that puts only its own
directory on `sys.path`. The numpy, numba, tqdm and batman packages come from the
current environment. Override the versions with `--tls1-version`/`--tls2-version` or with
the `TLS1_VERSION`/`TLS2_VERSION` environment variables.

## Data: mock pipeline products prepared with lightkurve

`simulate.py` writes FITS files with the layout of the official products: Kepler
`llc.fits` (`SAP_QUALITY`, BKJD) and TESS-SPOC `lc.fits` (`QUALITY`, BTJD), including
`SAP_FLUX`/`PDCSAP_FLUX` columns with errors. The simulated light curves contain:

- the mission cadence, quarter/sector length, data-downlink gaps and multi-segment spacing
- reaction-wheel desaturations, manual-exclude and coarse-point cadences, flagged with the
  quality bits of lightkurve's default bitmask, which carry corrupted or NaN flux
- quasi-periodic stellar spot modulation, correlated granulation-like noise, residual
  post-gap ramps, white noise and unflagged cosmic rays
- a batman transit with quadratic limb darkening, supersampled over the exposure

Each product is read with `lightkurve.read(quality_bitmask="default")`. The segments are
then stitched, flattened with a Savitzky-Golay filter and clipped of upward outliers,
exactly as archive data would be. All engines receive the same `t`, `flux` and `flux_err`
arrays. The `kepler_q3_real` scenario instead injects a transit into the real Kepler Q3
light curve in `tests/data`; that star's own 4.125 d signal stays in the data as a
competing signal.

| Scenario | Data |
| --- | --- |
| `kepler_q3_real` | Real Kepler Q3, 30-min, injected 6.83 d Neptune |
| `kepler_lc_1q` | 1 simulated Kepler quarter, 30-min, 9.13 d small planet |
| `kepler_lc_4q` | 4 simulated Kepler quarters (~375 d), 41.7 d small planet |
| `tess_2min_1s` | 1 simulated TESS sector, 2-min, 3.27 d sub-Neptune |
| `tess_2min_2s` | 2 simulated TESS sectors, 2-min, 11.4 d planet |
| `tess_2min_null` | 1 simulated TESS sector without a planet (false-alarm level) |

Suites: `quick` (used by pytest), `standard` (default) and `full` (all scenarios).
Run `python tests/compare_tls.py --list` to show them.

## Usage

```bash
python tests/compare_tls.py                                  # standard suite
python tests/compare_tls.py --suite full --plot              # everything, plus periodogram plots
python tests/compare_tls.py --scenarios tess_2min_2s --engines gtls,tls2,tls2-fused
python tests/compare_tls.py --gpus 0,1 --threads 16 --repeats 3
pytest tests/test_tls_compare.py                             # correctness checks (quick suite)
```

All engines use their own automatic period grid with the same parameters (`R_star`,
`M_star` and the TLS defaults), which yields identical grids. Each engine otherwise keeps
its shipped defaults; for example, TLS uses `T0_fit_margin=0.01`, while GTLS uses 0.125
for its coarse search before refining the peaks. TLS runs with `use_threads` = all
cores, unless `--threads` is given.

Before timing, every engine runs once on a tiny light curve to fill the numba and CuPy
compilation caches. For each scenario, the first `power()` call in a fresh process is
reported as **cold**; it includes JIT cache loading, CUDA context creation and TLS 2
worker-pool start-up. The median of the `--repeats` later calls is reported as **warm**.
Speedups are relative to `--baseline` (default `tls1`).

Results are written to `tests/.tls_compare_results/<timestamp>/`:

- `report.md`: environment, datasets, speed, search results versus the injected truth,
  pairwise agreement (ΔP, ΔT0, ΔSDE, Δdepth, periodogram correlation) and a summary
- `report.json`: all of the above in machine-readable form
- `<scenario>/products/*.fits`: the simulated pipeline products
- `<scenario>/lightcurve.npz` and `<scenario>/<engine>_periodogram.npz`: the searched
  data and the periodograms
- `<scenario>_periodograms.png`: written with `--plot`

TLS reports `depth` as the in-transit flux level. The framework converts it to the
fractional depth (`1 - depth`) that GTLS reports.

## Files

| File | Purpose |
| --- | --- |
| `simulate.py` | Mock Kepler/TESS products and lightkurve preparation |
| `scenarios.py` | Scenario presets and suites |
| `engines.py` | Engine registry, isolated TLS installation, subprocess runner |
| `worker.py` | Runs one engine on one light curve in a fresh process |
| `compare.py` | Orchestration, recovery and agreement metrics, Markdown/JSON report, plots |
| `../compare_tls.py` | Command-line interface |
| `../test_tls_compare.py` | pytest checks |
