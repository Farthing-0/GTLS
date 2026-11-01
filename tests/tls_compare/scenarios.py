"""Benchmark scenarios and suites."""
from .simulate import KEPLER_LC, TESS_2MIN, Planet, Scenario, Star

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
    ]
}

SUITES = {
    "quick": ["kepler_q3_real", "tess_2min_1s"],
    "standard": ["kepler_q3_real", "kepler_lc_1q", "tess_2min_1s", "tess_2min_null"],
    "full": list(SCENARIOS),
}


def search_kwargs(scenario):
    """Search parameters passed identically to every engine."""
    kwargs = {"R_star": scenario.star.radius, "M_star": scenario.star.mass}
    kwargs.update(scenario.search)
    return kwargs
