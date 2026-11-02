import numpy as np

from . import constants, core
from .grid import duration_grid, period_grid
from .results import gtlsResult
from .transit import get_cache
from .validate import validate_args, validate_inputs


class gtls(object):
    """GPU transit least squares search of a light curve.

    Parameters
    ----------
    t, y : array_like
        Time (days) and normalized flux.
    dy : array_like, optional
        Flux uncertainties; the flux standard deviation is used if omitted.
    verbose : bool
        Print status information and a progress bar.
    """

    def __init__(self, t, y, dy=None, verbose=True):
        self.t, self.y, self.dy = validate_inputs(t, y, dy)
        self.verbose = verbose

    def power(self, **kwargs):
        """Run the search and return a :class:`gtlsResult`.

        Accepts the TLS keyword arguments (``period_min``, ``period_max``, ``periods``,
        ``R_star``, ``M_star``, ``T0_fit_margin``, ...) plus ``GPUDeviceID``,
        ``GPUDeviceIDs`` (list of devices to split the search across), ``bar_location``
        and ``fast`` (return only ``(periods, power)``).
        """
        self, kwargs = validate_args(self, kwargs)

        if self.verbose:
            print(constants.TLS_VERSION)

        # Generate possible periods
        if len(self.periods) == 0:
            periods = period_grid(
                R_star=self.R_star,
                M_star=self.M_star,
                time_span=np.max(self.t) - np.min(self.t),
                period_min=self.period_min,
                period_max=self.period_max,
                oversampling_factor=self.oversampling_factor,
                n_transits_min=self.n_transits_min,
            )
        else:
            periods = self.periods

        # Generate possible durations
        durations = duration_grid(
            periods,
            log_step=self.duration_grid_step,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
        )

        maxwidth_in_samples = int(np.max(durations) * np.size(self.y))
        if maxwidth_in_samples % 2 != 0:
            maxwidth_in_samples = maxwidth_in_samples + 1
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

        result = core.search_multi_periods(
            periods=periods,
            t=self.t,
            y=self.y,
            dy=self.dy,
            transit_depth_min=self.transit_depth_min,
            lc_arr=self.lc_arr,
            lc_cache_overview=self.lc_cache_overview,
            T0_fit_margin=self.T0_fit_margin,
            oversampling_factor=self.oversampling_factor,
            verbose=self.verbose,
            GPUDeviceIDs=self.GPUDeviceIDs,
            fast=self.fast,
            bar_location=self.bar_location,
            R_star_min=self.R_star_min,
            R_star_max=self.R_star_max,
            M_star_min=self.M_star_min,
            M_star_max=self.M_star_max,
        )
        if self.fast:
            return result

        self.periods, self.period, self.rawDuration, durationPoints, self.duration, self.Depth, self.bestT0, \
            SDE, chi2, self.transitTimes, power, snr, snrPink, snrFit, snrFitPink, raw_power, raw_chi2, \
            possiblePeriodsIndices, possiblePeriods = result
        self.rawDurations = durations
        return gtlsResult(self.periods, self.period, self.rawDuration, durationPoints, durations, self.duration,
                          self.Depth, self.bestT0, SDE, chi2, self.transitTimes, power, snr, snrPink, snrFit,
                          snrFitPink, raw_power, raw_chi2, possiblePeriodsIndices, possiblePeriods)

    # Useful tools

    def showFit(self):
        def centerFold(time, period, T0):
            """Normal phase folding"""
            T0 = T0 + period/2
            return (time - T0) / period - np.floor((time - T0) / period)

        phases = centerFold(self.t, self.period, self.bestT0)
        phasesIndex = np.argsort(phases)
        phasesSorted = phases[phasesIndex]
        fluxesSorted = self.y[phasesIndex]

        durationStart = (self.bestT0 - self.duration/2)
        durationStartPhase = centerFold(durationStart, self.period, self.bestT0)

        lcArr = self.lc_arr
        
        assumeCurve = lcArr[np.where(self.rawDurations == self.rawDuration)[0][0]]
        assumeCurve = 1 - ((1- np.array(assumeCurve)) * 2 * (self.Depth))
        fitCurve = []
        IntransitCount = 0
        for point in phasesSorted:
            if point > durationStartPhase and IntransitCount < len(assumeCurve):
                fitCurve.append(assumeCurve[IntransitCount])
                IntransitCount += 1
            else:
                fitCurve.append(1)

        return fitCurve, phasesSorted, fluxesSorted 
    
    def periodogram(self, **kwargs):
        """Compute the periodogram for a set of user-defined parameters"""
        self, kwargs = validate_args(self, kwargs)

        if self.verbose:
            print(constants.TLS_VERSION)

        # Generate possible periods
        if self.periods == []:
            periods = period_grid(
                R_star=self.R_star,
                M_star=self.M_star,
                time_span=np.max(self.t) - np.min(self.t),
                period_min=self.period_min,
                period_max=self.period_max,
                oversampling_factor=self.oversampling_factor,
                n_transits_min=self.n_transits_min,
            )
        else:
            periods = self.periods        
        return periods