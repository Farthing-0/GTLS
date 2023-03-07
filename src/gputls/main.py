from __future__ import division, print_function
import numpy

from . import constants as constants
from .grid import duration_grid, period_grid
from .transit import get_cache
from .validate import validate_inputs, validate_args
from . import core as core

class gtls(object):
    """Compute the transit least squares of limb-darkened transit models"""

    def __init__(self, t, y, dy=None, verbose=True):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        self.verbose = verbose

    def power(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""
        self, kwargs = validate_args(self, kwargs)

        if self.verbose:
            print(constants.TLS_VERSION)

        # Generate possible periods
        periods = period_grid(
            R_star=self.R_star,
            M_star=self.M_star,
            time_span=numpy.max(self.t) - numpy.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        # Generate possible durations
        durations = duration_grid(
            periods, shortest=1 / len(self.t), log_step=self.duration_grid_step
        )

        maxwidth_in_samples = int(numpy.max(durations) * numpy.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        lc_cache_overview, lc_arr = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
            verbose=self.verbose
        )
        if self.verbose:
            print(
                "Searching "
                + str(len(self.y))
                + " data points, "
                + str(len(periods))
                + " periods from "
                + str(round(min(periods), 3))
                + " to "
                + str(round(max(periods), 3))
                + " days"
            )

        periods = numpy.sort(periods)

        periods,period,transit_duration_in_days,Depth,bestT0,SDE,chi2 = core.search_multi_periods(
            periods=periods,
            t=self.t,
            y=self.y,
            dy=self.dy,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
            lc_arr=lc_arr,
            lc_cache_overview=lc_cache_overview,
            T0_fit_margin=self.T0_fit_margin,
            oversampling_factor = self.oversampling_factor,
            verbose=self.verbose,
        )
        return periods,period,transit_duration_in_days,Depth,bestT0,SDE,chi2