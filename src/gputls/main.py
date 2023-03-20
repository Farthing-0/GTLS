from __future__ import division, print_function
import numpy as np

from . import constants as constants
from .grid import duration_grid, period_grid
from .transit import get_cache
from .validate import validate_inputs, validate_args
from . import core as core
from .results import gtlsResult

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
            time_span=np.max(self.t) - np.min(self.t),
            period_min=self.period_min,
            period_max=self.period_max,
            oversampling_factor=self.oversampling_factor,
            n_transits_min=self.n_transits_min,
        )

        # Generate possible durations
        durations = duration_grid(
            periods, shortest=1 / len(self.t), log_step=self.duration_grid_step
        )

        maxwidth_in_samples = int(np.max(durations) * np.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
        self.maxwidth_in_samples = maxwidth_in_samples
        self.lc_cache_overview, self.lc_arr = get_cache(
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

        _, lc_arr_grazing = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.per,
            rp=self.rp,
            a=self.a,
            inc=self.grazing_inc,
            ecc=self.ecc,
            w=self.w,
            u=self.u,
            limb_dark=self.limb_dark,
            verbose=self.verbose
        )

        _, lc_arr_box = get_cache(
            durations=durations,
            maxwidth_in_samples=maxwidth_in_samples,
            per=self.box_per,
            rp=self.box_rp,
            a=self.box_a,
            inc=self.box_inc,
            ecc=self.ecc,
            w=self.w,
            u=self.box_u,
            limb_dark=self.box_limb_dark,
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

        periods = np.sort(periods)

        self.periods,self.period,self.rawDuration,self.duration,self.Depth,self.bestT0,SDE,chi2,self.transitTimes,power,snr,snrPink,snrFit,snrFitPink = core.search_multi_periods(
            periods=periods,
            t=self.t,
            y=self.y,
            dy=self.dy,
            transit_depth_min=self.transit_depth_min,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
            lc_arr=self.lc_arr,
            lc_arr_grazing=lc_arr_grazing,
            lc_arr_box=lc_arr_box,
            lc_cache_overview=self.lc_cache_overview,
            T0_fit_margin=self.T0_fit_margin,
            oversampling_factor = self.oversampling_factor,
            verbose=self.verbose,
        )
        # return periods,period,duration,Depth,bestT0,SDE,chi2
        return gtlsResult(self.periods,self.period,self.rawDuration,durations,self.duration,self.Depth,self.bestT0,SDE,chi2,self.transitTimes,power,snr,snrPink,snrFit,snrFitPink)
    
    def showFit(self):
        # from .stats import calculate_fill_factor,calculate_stretch
        # from .transit import fractional_transit
        # # Folded model / model curve
        # # Data phase 0.5 is not always at the midpoint (not at cadence: len(y)/2),
        # # so we need to roll the model to match the model so that its mid-transit
        # # is at phase=0.5
        # fill_factor = calculate_fill_factor(self.t)
        # fill_half = 1 - ((1 - fill_factor) * 0.5)
        # stretch = calculate_stretch(self.t, self.period, self.transitTimes)
        # # internal_samples = (
        # #     int(len(self.y) / len(self.transitTimes))
        # # ) * constants.OVERSAMPLE_MODEL_LIGHT_CURVE

        # # Folded model flux
        # self.model_folded_model = fractional_transit(
        #     duration=self.duration * self.maxwidth_in_samples * fill_half,
        #     maxwidth=self.maxwidth_in_samples / stretch,
        #     depth = 1 - self.Depth,
        #     samples=int(len(self.t / len(self.transitTimes))),
        #     per=self.per,
        #     rp=self.rp,
        #     a=self.a,
        #     inc=self.inc,
        #     ecc=self.ecc,
        #     w=self.w,
        #     u=self.u,
        #     limb_dark=self.limb_dark,
        # )
        # # return self.model_folded_model
        return self.lc_arr