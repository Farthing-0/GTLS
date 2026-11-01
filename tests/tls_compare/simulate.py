"""Mock Kepler / TESS-SPOC light-curve products, read back and detrended with lightkurve.

The simulated light curves are written as FITS files with the layout of the official
pipeline products (Kepler ``*_llc.fits`` and TESS-SPOC ``*_lc.fits``): same extensions,
columns, time systems and quality bits. They are then loaded with ``lightkurve.read``
(applying its default quality bitmask), stitched, flattened and sigma-clipped exactly as
one would process archive data, so every engine searches the same realistic input.
"""
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import batman
import numpy as np
from astropy.io import fits

KEPLER_FITS = Path(__file__).resolve().parents[1] / "data" / "kplr011403044-2009350155506_llc.fits"

G = 6.674e-11
R_SUN = 6.957e8
M_SUN = 1.989e30
DAY = 86400.0


@dataclass(frozen=True)
class Mission:
    """Observing pattern of one pipeline product type."""

    name: str  # "Kepler" or "TESS"
    product: str
    cadence: float  # days
    segment_days: float  # length of one quarter / sector
    segment_gap: float  # days between consecutive segments
    downlinks: tuple  # (offset within segment, length) of data gaps, days
    dump_interval: float  # days between reaction-wheel desaturations (momentum dumps)
    t_start: float  # first time stamp, in BKJD / BTJD
    bjdref: int
    time_unit: str
    quality_column: str
    creator: str
    obsmode: str
    first_segment: int  # first quarter / sector number
    manual_exclude_bit: int  # "Manual exclude" quality bit (differs between Kepler and TESS)
    exposure_fraction: float = 0.9  # exposure time / cadence, for transit supersampling
    files_per_segment: int = 1  # e.g. 3 monthly files per quarter for Kepler short cadence
    file_tag: str = "lc"


KEPLER_LC = Mission(
    name="Kepler",
    product="Kepler long cadence (PDCSAP, llc.fits)",
    cadence=29.4243814 / 1440,
    segment_days=93.0,
    segment_gap=1.0,
    downlinks=((31.0, 1.0), (62.0, 1.0)),  # monthly data downlinks
    dump_interval=3.0,
    t_start=443.49,
    bjdref=2454833,
    time_unit="BJD - 2454833",
    quality_column="SAP_QUALITY",
    creator="FluxExporter2PipelineModule",
    obsmode="long cadence",
    first_segment=5,
    manual_exclude_bit=256,  # bit 128 is "cosmic ray in optimal aperture" for Kepler
)

TESS_2MIN = Mission(
    name="TESS",
    product="TESS-SPOC 2-min (PDCSAP, lc.fits)",
    cadence=2.0 / 1440,
    segment_days=27.4,
    segment_gap=1.1,
    downlinks=((13.1, 1.0),),  # perigee data downlink between the two orbits
    dump_interval=3.125,
    t_start=1683.35,
    bjdref=2457000,
    time_unit="BJD - 2457000, days",
    quality_column="QUALITY",
    creator="LightCurveExporterPipelineModule",
    obsmode="2-min cadence",
    first_segment=14,
    manual_exclude_bit=128,
)

# Kepler short cadence: one file per month, written alongside the long-cadence quarters
KEPLER_SC = replace(
    KEPLER_LC,
    product="Kepler short cadence (PDCSAP, slc.fits)",
    cadence=58.84876 / 86400,
    obsmode="short cadence",
    files_per_segment=3,
    file_tag="slc",
)

# The whole Kepler mission, Q1-Q17, with the approximate length of every quarter (days)
KEPLER_QUARTER_DAYS = (33.5, 88.9, 89.3, 93.5, 95.5, 91.7, 90.4, 67.3, 96.8, 93.4, 97.4, 82.1,
                       92.0, 96.9, 91.0, 87.5, 31.8)
KEPLER_MISSION_LC = replace(KEPLER_LC, t_start=131.51, segment_gap=2.0, first_segment=1)

TESS_20S = replace(
    TESS_2MIN,
    product="TESS-SPOC 20-s (PDCSAP, fast-lc.fits)",
    cadence=20.0 / 86400,
    obsmode="20-s cadence",
    file_tag="fast-lc",
)

# Quality bits with the same meaning for Kepler and TESS, in lightkurve's default bitmask
FLAG_COARSE_POINT = 4
FLAG_DESAT = 32


@dataclass(frozen=True)
class Star:
    radius: float = 1.0  # solar radii
    mass: float = 1.0  # solar masses
    teff: int = 5800
    u: tuple = (0.45, 0.20)  # quadratic limb darkening


@dataclass(frozen=True)
class Planet:
    period: float  # days
    epoch_offset: float  # first mid-transit time after the start of the data, days
    rp_rs: float
    b: float = 0.3


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    mission: Mission
    star: Star
    planet: Optional[Planet]
    seed: int
    n_segments: int = 1
    segment_days: tuple = ()  # length of every segment; default: mission.segment_days
    dense_mission: Optional[Mission] = None  # observing mode of the segments in dense_segments
    dense_segments: tuple = ()  # segment indices observed with dense_mission (e.g. short cadence)
    noise_ppm: float = 300.0  # white noise per cadence of `mission` (scaled for dense_mission)
    variability_ppm: float = 3000.0  # stellar rotational modulation amplitude
    rotation_period: float = 11.0  # days
    flatten_window_days: float = 1.0
    real_kepler: bool = False  # use the tests/data Kepler Q3 PDCSAP flux instead of simulated flux
    search: dict = field(default_factory=dict)  # extra keyword arguments shared by all engines


@dataclass
class Dataset:
    scenario: Scenario
    t: np.ndarray
    y: np.ndarray
    dy: np.ndarray
    truth: Optional[dict]
    files: list
    n_raw: int  # cadences in the pipeline products before quality masking and clipping
    lightkurve_class: str

    @property
    def span(self):
        return float(self.t.max() - self.t.min())

    def save(self, path):
        np.savez(path, t=self.t, y=self.y, dy=self.dy)
        return path


def semi_major_axis(period, star):
    """a / R_star from Kepler's third law."""
    a = (G * star.mass * M_SUN * (period * DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    return a / (star.radius * R_SUN)


def _batman_params(planet, star, t0):
    params = batman.TransitParams()
    params.t0 = t0
    params.per = planet.period
    params.rp = planet.rp_rs
    params.a = semi_major_axis(planet.period, star)
    params.inc = np.degrees(np.arccos(planet.b / params.a))
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(star.u)
    params.limb_dark = "quadratic"
    return params


def transit_model(t, planet, star, t0, cadence, exposure_fraction):
    params = _batman_params(planet, star, t0)
    model = batman.TransitModel(
        params, t, supersample_factor=7, exp_time=cadence * exposure_fraction)
    return model.light_curve(params)


def transit_truth(planet, star, t0):
    params = _batman_params(planet, star, t0)
    arg = np.sqrt((1 + planet.rp_rs) ** 2 - planet.b ** 2) / (
        params.a * np.sin(np.radians(params.inc)))
    duration = planet.period / np.pi * np.arcsin(arg)
    t_fine = t0 + np.linspace(-duration, duration, 4001)
    depth = 1 - batman.TransitModel(params, t_fine).light_curve(params).min()
    return {
        "period": planet.period,
        "T0": t0,
        "rp_rs": planet.rp_rs,
        "b": planet.b,
        "a_rs": float(params.a),
        "inc": float(params.inc),
        "duration": float(duration),
        "depth": float(depth),
    }


def _segment_starts(scenario):
    """Start time and length of every segment."""
    mission = scenario.mission
    if not scenario.segment_days:
        return [(mission.t_start + index * (mission.segment_days + mission.segment_gap),
                 mission.segment_days) for index in range(scenario.n_segments)]
    starts = mission.t_start + np.concatenate(
        ([0.0], np.cumsum(np.add(scenario.segment_days, mission.segment_gap))[:-1]))
    return list(zip(starts, scenario.segment_days))[:scenario.n_segments]


def _segment_cadences(mission, start, length, first_cadence):
    """Time stamps and cadence numbers of one quarter / sector, without the downlink gaps."""
    n = int(round(length / mission.cadence))
    cadenceno = np.arange(n) + first_cadence
    t = start + np.arange(n) * mission.cadence
    keep = np.ones(n, dtype=bool)
    for offset, gap in mission.downlinks:
        keep &= ~((t >= start + offset) & (t < start + offset + gap))
    return t[keep], cadenceno[keep] + 1000


def _red_noise(n, rng, sigma, corr_cadences):
    """AR(1) correlated (granulation-like) noise with standard deviation sigma."""
    phi = np.exp(-1.0 / max(corr_cadences, 1.0))
    white = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    out = np.empty(n)
    out[0] = rng.normal(0, sigma)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + white[i]
    return out


def stellar_variability(t, rng, amplitude, rotation_period):
    """Quasi-periodic spot modulation with slowly evolving amplitude."""
    phases = rng.uniform(0, 2 * np.pi, 4)
    evolve = 2.5 * rotation_period
    a1 = 1.0 + 0.35 * np.sin(2 * np.pi * t / evolve + phases[0])
    a2 = 0.45 + 0.20 * np.sin(2 * np.pi * t / (0.7 * evolve) + phases[1])
    phase = 2 * np.pi * t / rotation_period
    signal = a1 * np.sin(phase + phases[2]) + a2 * np.sin(2 * phase + phases[3])
    return amplitude * signal / 1.5


def _systematics(t, mission, segment_start, rng):
    """Residual PDC systematics: thermal / scattered-light ramps after each data gap."""
    out = np.zeros_like(t)
    edges = [segment_start] + [segment_start + o + l for o, l in mission.downlinks]
    for edge in edges:
        after = t >= edge
        amp = rng.uniform(2e-4, 6e-4) * rng.choice([-1, 1])
        out[after] += amp * np.exp(-(t[after] - edge) / rng.uniform(0.2, 0.6))
    return out


def _quality_flags(t, mission, segment_start, segment_days, rng):
    quality = np.zeros(t.size, dtype=np.int32)
    dumps = segment_start + np.arange(mission.dump_interval / 2, segment_days,
                                      mission.dump_interval)
    for dump in dumps:
        quality[np.abs(t - dump) < 1.5 * mission.cadence] |= FLAG_DESAT
    quality[rng.random(t.size) < 2e-3] |= mission.manual_exclude_bit
    quality[rng.random(t.size) < 1e-3] |= FLAG_COARSE_POINT
    return quality


def _write_product(path, mission, segment, target_id, star, t, pdcsap, pdcsap_err,
                   quality, cadenceno, rng):
    level = float(np.nanmedian(pdcsap))
    sap = pdcsap * (1 + 2e-3 * (t - t.mean()) / max(np.ptp(t), 1e-3)) * 0.92
    if mission.name == "Kepler":
        primary = {
            "TELESCOP": "Kepler", "INSTRUME": "Kepler Photometer", "MISSION": "Kepler",
            "OBJECT": f"KIC {target_id}", "KEPLERID": target_id, "QUARTER": segment,
            "CHANNEL": 42, "MODULE": 13, "OUTPUT": 2,
        }
    else:
        primary = {
            "TELESCOP": "TESS", "INSTRUME": "TESS Photometer", "MISSION": "TESS",
            "OBJECT": f"TIC {target_id}", "TICID": target_id, "SECTOR": segment,
            "CAMERA": 1, "CCD": 2,
        }
    primary.update({
        "ORIGIN": "NASA/Ames", "CREATOR": mission.creator, "OBSMODE": mission.obsmode,
        "TEFF": star.teff, "RADIUS": star.radius, "DATA_REL": 0, "SIMDATA": True,
    })
    hdu0 = fits.PrimaryHDU()
    for key, value in primary.items():
        hdu0.header[key] = value
    hdu0.header["COMMENT"] = "Simulated light curve generated by the GTLS/TLS comparison framework"

    n = t.size
    columns = [
        fits.Column("TIME", "D", unit=mission.time_unit, array=t),
        fits.Column("TIMECORR", "E", unit="d", array=np.zeros(n)),
        fits.Column("CADENCENO", "J", array=cadenceno),
        fits.Column("SAP_FLUX", "E", unit="e-/s", array=sap),
        fits.Column("SAP_FLUX_ERR", "E", unit="e-/s", array=pdcsap_err),
        fits.Column("SAP_BKG", "E", unit="e-/s", array=rng.normal(0.02 * level, 1e-3 * level, n)),
        fits.Column("SAP_BKG_ERR", "E", unit="e-/s", array=np.full(n, 1e-3 * level)),
        fits.Column("PDCSAP_FLUX", "E", unit="e-/s", array=pdcsap),
        fits.Column("PDCSAP_FLUX_ERR", "E", unit="e-/s", array=pdcsap_err),
        fits.Column(mission.quality_column, "J", array=quality),
        fits.Column("MOM_CENTR1", "D", unit="pixel", array=500 + rng.normal(0, 0.01, n)),
        fits.Column("MOM_CENTR2", "D", unit="pixel", array=800 + rng.normal(0, 0.01, n)),
        fits.Column("POS_CORR1", "E", unit="pixel", array=rng.normal(0, 0.01, n)),
        fits.Column("POS_CORR2", "E", unit="pixel", array=rng.normal(0, 0.01, n)),
    ]
    hdu1 = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")
    hdu1.header.update({
        "TIMEREF": "SOLARSYSTEM", "TASSIGN": "SPACECRAFT", "TIMESYS": "TDB",
        "BJDREFI": mission.bjdref, "BJDREFF": 0.0, "TIMEUNIT": "d",
        "TSTART": float(t.min()), "TSTOP": float(t.max()), "TIMEDEL": mission.cadence,
    })
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True, checksum=True)
    return path


def _simulate_products(scenario, workdir):
    mission, star, planet = scenario.mission, scenario.star, scenario.planet
    rng = np.random.default_rng(scenario.seed)
    target_id = int(rng.integers(10_000_000, 90_000_000))
    t0 = mission.t_start + planet.epoch_offset if planet is not None else None
    noise = scenario.noise_ppm * 1e-6
    level = float(rng.uniform(1e4, 5e4))  # e-/s
    files = []
    n_raw = 0
    for index, (start, length) in enumerate(_segment_starts(scenario)):
        segment = mission.first_segment + index
        mode = scenario.dense_mission if index in scenario.dense_segments else mission
        # Same photometric precision per unit time at every cadence
        white = noise * np.sqrt(mission.cadence / mode.cadence)
        first_cadence = int(round((start - mission.t_start) / mode.cadence))
        t, cadenceno = _segment_cadences(mode, start, length, first_cadence)
        n = t.size
        flux = 1.0 + stellar_variability(t, rng, scenario.variability_ppm * 1e-6,
                                         scenario.rotation_period)
        flux += _systematics(t, mode, start, rng)
        flux += _red_noise(n, rng, 0.25 * noise, corr_cadences=0.05 / mode.cadence)
        if planet is not None:
            flux *= transit_model(t, planet, star, t0, mode.cadence, mode.exposure_fraction)
        err = white * rng.uniform(0.95, 1.05, n)
        flux += rng.normal(0, 1, n) * err
        # Unflagged cosmic rays, to be removed by the upper sigma clip
        hits = rng.random(n) < 5e-4
        flux[hits] += rng.uniform(5, 20, hits.sum()) * white
        quality = _quality_flags(t, mode, start, length, rng)
        # Flagged cadences carry corrupted flux, so ignoring the quality mask hurts
        bad = quality != 0
        flux[bad] -= rng.uniform(5, 50, bad.sum()) * white
        flux[(quality & FLAG_COARSE_POINT) != 0] = np.nan
        prefix = "kplr" if mode.name == "Kepler" else "tess"
        # Equal-length files within the segment (e.g. the three months of a quarter)
        part = np.minimum(((t - start) / length * mode.files_per_segment).astype(int),
                          mode.files_per_segment - 1)
        for p in range(mode.files_per_segment):
            sel = part == p
            if not sel.any():
                continue
            suffix = "m%d" % (p + 1) if mode.files_per_segment > 1 else ""
            path = Path(workdir) / f"{prefix}{target_id:09d}-s{segment:02d}{suffix}_sim_{mode.file_tag}.fits"
            files.append(_write_product(path, mode, segment, target_id, star, t[sel],
                                        flux[sel] * level, err[sel] * level, quality[sel],
                                        cadenceno[sel], rng))
        n_raw += n
    truth = transit_truth(planet, star, t0) if planet is not None else None
    return files, truth, n_raw


def _inject_into_real_kepler(scenario, workdir):
    """Multiply the real Kepler Q3 flux by a transit model and save it as a new product."""
    planet, star = scenario.planet, scenario.star
    with fits.open(KEPLER_FITS) as hdul:
        hdul = fits.HDUList([h.copy() for h in hdul])
    data = hdul[1].data
    t = np.array(data["TIME"], dtype=float)
    finite = np.isfinite(t)
    t0 = float(np.nanmin(t)) + planet.epoch_offset
    model = np.ones_like(t)
    model[finite] = transit_model(t[finite], planet, star, t0, KEPLER_LC.cadence,
                                  KEPLER_LC.exposure_fraction)
    for column in ("PDCSAP_FLUX", "SAP_FLUX"):
        data[column] = data[column] * model
    hdul[0].header["SIMDATA"] = True
    hdul[0].header["COMMENT"] = "Real Kepler Q3 light curve with an injected transit"
    path = Path(workdir) / f"{KEPLER_FITS.stem}_injected.fits"
    hdul.writeto(path, overwrite=True, checksum=True)
    return [path], transit_truth(planet, star, t0), int(finite.sum())


def read_and_detrend(files, flatten_window_days, outlier_sigma=4.0):
    """Standard lightkurve preparation of archive light curves for a transit search."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lightkurve as lk

        lcs = [lk.read(str(f), quality_bitmask="default") for f in files]
        cadences = [float(np.median(np.diff(lc.time.value))) for lc in lcs]

        def flatten(lc, cadence):
            window = int(round(flatten_window_days / cadence)) | 1
            return lc.flatten(window_length=window, polyorder=2, break_tolerance=10)

        if max(cadences) < 1.5 * min(cadences):
            lc = lk.LightCurveCollection(lcs).stitch().remove_nans()
            flat = flatten(lc, float(np.median(np.diff(lc.time.value))))
        else:
            # The flatten window is counted in cadences, so mixed cadences (e.g. Kepler long
            # and short cadence) are flattened file by file
            flat = lk.LightCurveCollection(
                [flatten(lc.remove_nans(), cadence) for lc, cadence in zip(lcs, cadences)]
            ).stitch(corrector_func=lambda lc: lc)
        flat = flat.remove_outliers(sigma_lower=float("inf"), sigma_upper=outlier_sigma)
    t = np.asarray(flat.time.value, dtype=float)
    y = np.asarray(flat.flux.value, dtype=float)
    dy = np.asarray(flat.flux_err.value, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    return t[ok], y[ok], dy[ok], type(lcs[0]).__name__


def build_dataset(scenario, workdir):
    """Write the pipeline products of a scenario and return the lightkurve-prepared data."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if scenario.real_kepler:
        files, truth, n_raw = _inject_into_real_kepler(scenario, workdir)
    else:
        files, truth, n_raw = _simulate_products(scenario, workdir)
    t, y, dy, lc_class = read_and_detrend(files, scenario.flatten_window_days)
    return Dataset(scenario, t, y, dy, truth, [str(f) for f in files], n_raw, lc_class)
