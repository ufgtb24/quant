import backtrader as bt

class MFI(bt.Indicator):
    lines = ('mfi',)
    params = (('period', 14),
    )

    plotinfo = dict(subplot=True)
    plotlines = dict(
        # mfi=dict(_plotskip=True),
        mfi=dict(ls='--'),
    )


    def __init__(self):
        super(MFI, self).__init__()
        typical_price=(self.data.high+self.data.low+self.data.close)/3
        rmf=typical_price*self.data.volume
        
        positive_rmf=bt.If(self.data.close>=self.data.close(-1),rmf,0)
        negative_rmf=bt.If(self.data.close<self.data.close(-1),rmf,0)
        mf_ratio=bt.ind.SumN(positive_rmf, period=self.params.period)/bt.ind.SumN(negative_rmf, period=self.params.period)
        self.l.mfi=100-100/(1+mf_ratio)
        
