import backtrader as bt


class Keltler(bt.Indicator):
    lines = ('ema','upper','lowwer',)
    params = (
        ('ema_period', 20),
        ('atr_period', 17),
        ('up_rate',1),
        ('low_rate',1)
    )

    plotinfo = dict(subplot=False)
    plotlines = dict(
        ema=dict(color='black'),
        upper=dict(ls='--',color='red'),
        lowwer=dict(ls='--',color='blue'),
    )


    def __init__(self):
        super(Keltler, self).__init__()
        self.l.ema=bt.talib.EMA(self.data.close,timeperiod=self.p.ema_period)
        atr=bt.talib.ATR(self.data.high,self.data.low,self.data.close,timeperiod=self.p.atr_period)
        self.l.upper=self.l.ema+self.p.up_rate*atr
        self.l.lowwer=self.l.ema-self.p.low_rate*atr


class Keltler_strategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        self.keltler = Keltler()
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    
    def next(self):
        if not self.position:
            if self.data.close < self.keltler.lowwer:
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.buy()
        
        elif self.data.close > self.keltler.upper:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.close()