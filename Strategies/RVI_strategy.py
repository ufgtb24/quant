import math

import backtrader as bt
import backtrader.indicator as btind

from Ind.RVI import RVI


class RVI_stg(bt.Strategy):
    params = (
        ('rvi_low', 50),
        ('sma_period', 20)
    )
    lines=()
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    
    def __init__(self):
        self.rvi = RVI(movav=bt.ind.EMA)
        ma = bt.ind.SMA(period=self.params.sma_period)
        
        self.crossover = bt.ind.CrossOver(self.data.close, ma)

        
        
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
                self.sell(exectype=bt.Order.StopTrail, trailpercent=0.1)
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
        size_buy=math.floor(self.broker.get_cash()/self.data.close[0])
        if not self.position:
            if self.rvi.rvi[0] < self.p.rvi_low:
                if self.rvi.rvi[0]>self.rvi.rvi[-1] and self.rvi.rvi[-1]>self.rvi.rvi[-2]:
                    self.log('BUY CREATE, %.2f' % self.data.close[0])
                    self.buy()
        
        elif self.crossover == -1:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.close()