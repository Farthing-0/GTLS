class gtlsResult(object):
    """The results of a GTLS search"""

    def __init__(self, periods, period, duration, depth, T0, SDE, chi2, power):
        self.periods = periods
        self.period = period
        self.duration = duration
        self.depth = depth
        self.T0 = T0
        self.SDE = SDE
        self.chi2 = chi2
        self.power = power