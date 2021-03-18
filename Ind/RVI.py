import backtrader as bt

class RVI(bt.Indicator):
    lines = ('rvi','upDaySTD','downDaySTD')
    params = (('period', 14),
    ('movav', bt.ind.MovAv.Smoothed),
    ('upperband', 70.0),
    ('lowerband', 30.0),
    ('safehigh', 100.0),
    ('safelow', 50.0),
    )

    plotinfo = dict(subplot=True)
    plotlines = dict(
        upDaySTD=dict(_plotskip=True),
        downDaySTD=dict(_plotskip=True),
    )

    def _plotinit(self):
        self.plotinfo.plotyhlines = [self.p.upperband, self.p.lowerband]

    def __init__(self):
        super(RVI, self).__init__()
        self.lines.upDaySTD=bt.If(self.data-self.data(-1)<0,bt.ind.StdDev(self.data, period=self.p.period),0)
        self.lines.downDaySTD=bt.If(self.data-self.data(-1)>0,bt.ind.StdDev(self.data, period=self.p.period),0)
        maup = self.p.movav(self.lines.upDaySTD, period=self.p.period)
        madown = self.p.movav(self.lines.downDaySTD, period=self.p.period)
        highrs = self._rscalc(self.p.safehigh)
        lowrs = self._rscalc(self.p.safelow)
        rv = bt.ind.DivZeroByZero(maup, madown, highrs, lowrs)
        self.l.rvi = 100.0 - 100.0 / (1.0 + rv)

    def _rscalc(self, rsi):
        try:
            rs = (-100.0 / (rsi - 100.0)) - 1.0
        except ZeroDivisionError:
            return float('inf')
        return rs