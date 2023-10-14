class gtlsResult(object):
    """The results of a GTLS search"""

    def __init__(self, periods,period,rawDuration,durationPoints,rawDurations,duration,depth,T0,SDE,chi2,transitTimes,power,snr,snrPink,snrFit,snrFitPink):#,lossSDE,KLossMean,KLossStd):
        self.periods = periods
        self.period = period
        self.rawDuration = rawDuration
        self.durationPoints = durationPoints
        self.rawDurations = rawDurations
        self.duration = duration
        self.depth = depth
        self.T0 = T0
        self.SDE = SDE
        self.chi2 = chi2
        self.transitTimes = transitTimes
        self.power = power
        self.snr = snr
        self.snrPink = snrPink
        self.snrFit = snrFit
        self.snrFitPink = snrFitPink
