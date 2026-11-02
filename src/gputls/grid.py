import warnings

import numpy
from numpy import pi, sqrt

from . import constants


def T14(
    R_s, M_s, P, upper_limit=constants.FRACTIONAL_TRANSIT_DURATION_MAX, small=False
):
    """Input:  Stellar radius and mass; planetary period
               Units: Solar radius and mass; days
       Output: Maximum planetary transit duration T_14max
               Unit: Fraction of period P"""

    P = P * constants.SECONDS_PER_DAY
    R_s = constants.R_sun * R_s
    M_s = constants.M_sun * M_s

    if small:  # small planet assumption
        T14max = R_s * ((4 * P) / (pi * constants.G * M_s)) ** (1 / 3)
    else:  # planet size 2 R_jup
        T14max = (R_s + 2 * constants.R_jup) * (
            (4 * P) / (pi * constants.G * M_s)
        ) ** (1 / 3)

    result = T14max / P
    if result > upper_limit:
        result = upper_limit
    return result


def _density(M, R):
    return M / (R * R * R)


# Stellar densities (solar units) of the default corners: the smallest star
# (R_STAR_MIN, M_STAR_MIN) and the largest one (R_STAR_MAX, M_STAR_MAX).
RHO_DEFAULT_MAX = _density(constants.M_STAR_MIN, constants.R_STAR_MIN)
RHO_DEFAULT_MIN = _density(constants.M_STAR_MAX, constants.R_STAR_MAX)


def duration_limit_masses(R_star_min, R_star_max, M_star_min, M_star_max):
    """Masses to pair with R_star_min (shortest) and R_star_max (longest duration), as TLS 2.

    T14 ~ R * M^(-1/3) ~ (P / rho)^(1/3). The interval extremes are (R_min, M_max) for the
    shortest and (R_max, M_min) for the longest duration. TLS <= 1.33 paired (R_min, M_min)
    and (R_max, M_max) instead, which is too narrow for narrow priors. For wide intervals the
    extreme corners are unphysical (R 0.13 with M 1.0 is 455 rho_sun), so the extreme density
    is clipped to the more extreme of the old corner and the default corner. Hence the default
    limits give the TLS 1.33 grid exactly, narrow priors get the full interval range, and no
    range is ever narrower than in TLS 1.33.

    Returns (m_short, m_long).
    """
    # shortest duration: highest density
    rho_cap = max(_density(M_star_min, R_star_min), RHO_DEFAULT_MAX)
    if _density(M_star_max, R_star_min) <= rho_cap:
        m_short = M_star_max
    elif _density(M_star_min, R_star_min) >= RHO_DEFAULT_MAX:
        m_short = M_star_min  # old corner (exact)
    else:
        m_short = RHO_DEFAULT_MAX * R_star_min**3
    # longest duration: lowest density
    rho_floor = min(_density(M_star_max, R_star_max), RHO_DEFAULT_MIN)
    if _density(M_star_min, R_star_max) >= rho_floor:
        m_long = M_star_min
    elif _density(M_star_max, R_star_max) <= RHO_DEFAULT_MIN:
        m_long = M_star_max  # old corner (exact)
    else:
        m_long = RHO_DEFAULT_MIN * R_star_max**3
    return m_short, m_long


def duration_grid(
    periods,
    log_step=constants.DURATION_GRID_STEP,
    R_star_min=constants.R_STAR_MIN,
    R_star_max=constants.R_STAR_MAX,
    M_star_min=constants.M_STAR_MIN,
    M_star_max=constants.M_STAR_MAX,
):
    """Logarithmic grid of trial durations (fractions of the period)."""
    m_short, m_long = duration_limit_masses(R_star_min, R_star_max, M_star_min, M_star_max)
    duration_max = T14(
        R_s=R_star_max, M_s=m_long, P=numpy.min(periods), small=False
    )  # large planet for long transit duration
    duration_min = T14(
        R_s=R_star_min, M_s=m_short, P=numpy.max(periods), small=True
    )  # small planet for short transit duration

    durations = [duration_min]
    current_depth = duration_min
    while current_depth * log_step < duration_max:
        current_depth = current_depth * log_step
        durations.append(current_depth)
    durations.append(duration_max)  # Append endpoint. Not perfectly spaced.
    return durations


def period_grid(
    R_star,
    M_star,
    time_span,
    period_min=0,
    period_max=float("inf"),
    oversampling_factor=constants.OVERSAMPLING_FACTOR,
    n_transits_min=constants.N_TRANSITS_MIN,
):
    """Returns array of optimal sampling periods for transit search in light curves
       Following Ofir (2014, A&A, 561, A138)"""

    if R_star < 0.01:
        text = (
            "Warning: R_star was set to 0.01 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 0.1

    if R_star > 10000:
        text = (
            "Warning: R_star was set to 10000 for period_grid (was unphysical: "
            + str(R_star)
            + ")"
        )
        warnings.warn(text)
        R_star = 10000

    if M_star < 0.01:
        text = (
            "Warning: M_star was set to 0.01 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 0.01

    if M_star > 1000:
        text = (
            "Warning: M_star was set to 1000 for period_grid (was unphysical: "
            + str(M_star)
            + ")"
        )
        warnings.warn(text)
        M_star = 1000

    R_star = R_star * constants.R_sun
    M_star = M_star * constants.M_sun
    time_span = time_span * constants.SECONDS_PER_DAY  # seconds

    # boundary conditions
    f_min = n_transits_min / time_span
    f_max = 1.0 / (2 * pi) * sqrt(constants.G * M_star / (3 * R_star) ** 3)

    # optimal frequency sampling, Equations (5), (6), (7)
    A = (
        (2 * pi) ** (2.0 / 3)
        / pi
        * R_star
        / (constants.G * M_star) ** (1.0 / 3)
        / (time_span * oversampling_factor)
    )
    C = f_min ** (1.0 / 3) - A / 3.0
    N_opt = (f_max ** (1.0 / 3) - f_min ** (1.0 / 3) + A / 3) * 3 / A

    X = numpy.arange(N_opt) + 1

    f_x = (A / 3 * X + C) ** 3
    P_x = 1 / f_x

    # Cut to given (optional) selection of periods
    periods = P_x / constants.SECONDS_PER_DAY
    selected_index = numpy.where(
        numpy.logical_and(periods > period_min, periods <= period_max)
    )

    number_of_periods = numpy.size(periods[selected_index])

    if number_of_periods > 10 ** 6:
        text = (
            "period_grid generates a very large grid ("
            + str(number_of_periods)
            + "). Recommend to check physical plausibility for stellar mass, radius, and time series duration."
        )
        warnings.warn(text)

    if number_of_periods < constants.MINIMUM_PERIOD_GRID_SIZE:
        if time_span < 5 * constants.SECONDS_PER_DAY:
            time_span = 5 * constants.SECONDS_PER_DAY
        warnings.warn(
            "period_grid defaults to R_star=1 and M_star=1 as given density yielded grid with too few values"
        )
        return period_grid(
            R_star=1, M_star=1, time_span=time_span / constants.SECONDS_PER_DAY
        )
    else:
        return periods[selected_index]  # periods in [days]
