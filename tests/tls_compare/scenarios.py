"""Benchmark scenarios and suites."""
from .simulate import (KEPLER_LC, KEPLER_MISSION_LC, KEPLER_QUARTER_DAYS, KEPLER_SC, TESS_2MIN,
                       TESS_20S, Planet, Scenario, Star)

SUN_KEPLER = Star(radius=1.0, mass=1.0, teff=5800, u=(0.45, 0.20))
SUN_TESS = Star(radius=1.0, mass=1.0, teff=5800, u=(0.35, 0.25))

SCENARIOS = {
    s.name: s
    for s in [
        Scenario(
            name="kepler_q3_real",
            description="Real Kepler Q3 PDCSAP light curve (KIC 11403044, tests/data) with an "
                        "injected Neptune; the star's own 4.125 d, ~1200 ppm transit-like signal "
                        "stays in the data as a competing signal",
            mission=KEPLER_LC,
            star=Star(radius=1.1, mass=1.0, teff=6174, u=(0.40, 0.25)),
            planet=Planet(period=6.83, epoch_offset=1.83, rp_rs=0.048, b=0.2),
            seed=0,
            real_kepler=True,
            flatten_window_days=1.0,
        ),
        Scenario(
            name="kepler_lc_1q",
            description="Simulated Kepler long-cadence quarter, spotted star, small planet",
            mission=KEPLER_LC,
            star=SUN_KEPLER,
            planet=Planet(period=9.13, epoch_offset=4.2, rp_rs=0.025, b=0.3),
            seed=11,
            noise_ppm=250,
            variability_ppm=3000,
            rotation_period=11.0,
            flatten_window_days=1.0,
        ),
        Scenario(
            name="kepler_lc_4q",
            description="Four simulated Kepler long-cadence quarters, long-period small planet",
            mission=KEPLER_LC,
            star=SUN_KEPLER,
            planet=Planet(period=41.7, epoch_offset=17.3, rp_rs=0.02, b=0.2),
            seed=12,
            n_segments=4,
            noise_ppm=150,
            variability_ppm=2000,
            rotation_period=17.0,
            flatten_window_days=1.5,
        ),
        Scenario(
            name="tess_2min_1s",
            description="Simulated TESS-SPOC 2-min sector, short-period sub-Neptune",
            mission=TESS_2MIN,
            star=SUN_TESS,
            planet=Planet(period=3.27, epoch_offset=1.1, rp_rs=0.032, b=0.3),
            seed=21,
            noise_ppm=1000,
            variability_ppm=4000,
            rotation_period=6.5,
            flatten_window_days=0.75,
        ),
        Scenario(
            name="tess_2min_2s",
            description="Two consecutive simulated TESS-SPOC 2-min sectors, warm planet",
            mission=TESS_2MIN,
            star=SUN_TESS,
            planet=Planet(period=11.4, epoch_offset=5.6, rp_rs=0.035, b=0.4),
            seed=22,
            n_segments=2,
            noise_ppm=800,
            variability_ppm=3000,
            rotation_period=9.0,
            flatten_window_days=1.0,
        ),
        Scenario(
            name="tess_2min_null",
            description="Simulated TESS-SPOC 2-min sector without a planet (false-alarm check)",
            mission=TESS_2MIN,
            star=SUN_TESS,
            planet=None,
            seed=23,
            noise_ppm=1000,
            variability_ppm=4000,
            rotation_period=6.5,
            flatten_window_days=0.75,
        ),
        # Long baselines and dense cadences
        Scenario(
            name="kepler_lc_17q",
            description="The whole simulated Kepler mission (Q1-Q17, ~1460 d) at long cadence, "
                        "long-period sub-Neptune",
            mission=KEPLER_MISSION_LC,
            star=SUN_KEPLER,
            planet=Planet(period=122.4, epoch_offset=61.7, rp_rs=0.02, b=0.3),
            seed=31,
            n_segments=17,
            segment_days=KEPLER_QUARTER_DAYS,
            noise_ppm=150,
            variability_ppm=2000,
            rotation_period=17.0,
            flatten_window_days=2.5,
        ),
        Scenario(
            name="kepler_sc_1q",
            description="One simulated Kepler quarter at short cadence (1 min, three monthly "
                        "files), short-period small planet",
            mission=KEPLER_SC,
            star=SUN_KEPLER,
            planet=Planet(period=2.73, epoch_offset=1.1, rp_rs=0.018, b=0.2),
            seed=32,
            noise_ppm=1400,  # per 1-min cadence, = 255 ppm per 30 min
            variability_ppm=3000,
            rotation_period=11.0,
            flatten_window_days=0.75,
        ),
        Scenario(
            name="kepler_mixed_17q",
            description="The whole simulated Kepler mission at long cadence, with short cadence "
                        "instead in Q10-Q12 (stitched LC + SC), long-period sub-Neptune",
            mission=KEPLER_MISSION_LC,
            star=SUN_KEPLER,
            planet=Planet(period=122.4, epoch_offset=61.7, rp_rs=0.02, b=0.3),
            seed=33,
            n_segments=17,
            segment_days=KEPLER_QUARTER_DAYS,
            dense_mission=KEPLER_SC,
            dense_segments=(9, 10, 11),
            noise_ppm=150,
            variability_ppm=2000,
            rotation_period=17.0,
            flatten_window_days=2.5,
        ),
        Scenario(
            name="tess_20s_1s",
            description="Simulated TESS-SPOC 20-s sector, short-period sub-Neptune",
            mission=TESS_20S,
            star=SUN_TESS,
            planet=Planet(period=3.27, epoch_offset=1.1, rp_rs=0.03, b=0.3),
            seed=34,
            noise_ppm=2450,  # per 20-s cadence, = 1000 ppm per 2 min
            variability_ppm=4000,
            rotation_period=6.5,
            flatten_window_days=0.75,
        ),
    ]
}

SUITES = {
    "quick": ["kepler_q3_real", "tess_2min_1s"],
    "standard": ["kepler_q3_real", "kepler_lc_1q", "tess_2min_1s", "tess_2min_null"],
    "full": ["kepler_q3_real", "kepler_lc_1q", "kepler_lc_4q", "tess_2min_1s", "tess_2min_2s",
             "tess_2min_null"],
    # Whole Kepler mission and dense cadences; TLS 1.x needs hours here, so run with --timeout
    "long": ["kepler_lc_17q", "kepler_sc_1q", "kepler_mixed_17q", "tess_20s_1s"],
    "all": list(SCENARIOS),
}


def search_kwargs(scenario):
    """Search parameters passed identically to every engine."""
    kwargs = {"R_star": scenario.star.radius, "M_star": scenario.star.mass}
    kwargs.update(scenario.search)
    return kwargs
