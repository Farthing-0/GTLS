# The GTLS(GPU Transit Least Squares) algorithm is adapted from the TLS(Transit Least Squares) algorithm by Michael Hippke & René Heller (2019)
# The TLS is an open source software with MIT license. The copyright of the TLS algorithm is held by the authors.
# You can find the original paper here: https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..39H/abstract

# The GTLS algorithm is also an open source software with MIT license.

import numpy as np
import cupy as cp
from gtls.validate import validate_inputs,validateAndChooseDevice,validate_args
import gtls.core as core
import gtls.tls_constants as tls_constants
from gtls.grid import duration_grid, period_grid
from gtls.transit import get_cache

class gtls(object):
    """Compute the transit least squares of limb-darkened transit models using GPU"""

    def __init__(self,t,y,dy = None,verbose = False):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        self.verbose = verbose
    
    def power(self,**kwargs):
        self, kwargs = validate_args(self, kwargs)
        # self.show_progress_bar = True
        self.device = validateAndChooseDevice(self.deviceId)
        if self.device == None:
            return
        # if self.verbose:
        #     print(tls_constants.TLS_VERSION)

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
        # maxwidth_in_samples : durations width in the longest transit
        maxwidth_in_samples = int(np.max(durations) * np.size(self.y))
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

        period,duration,depth,T0,SDE = core.search_multi_periods(
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
            show_progress_bar=self.show_progress_bar,
            oversampling_factor=self.oversampling_factor
        )
        return period, duration, depth, T0, SDE