# Import the backtrader platform
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
# import quantstats

from Data_ops.data_A import aquire_CN
from Strategies.RVI_strategy import RVI_stg
from tutorial import MaCrossStrategy


class TestStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

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
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + self.params.exitbars):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(RVI_stg)
    stock='ANNX'
    # stock='002600.SZ'
    
    #### load CSV
    # data = bt.feeds.YahooFinanceCSVData(
    #     dataname='datas/IVR.csv',
    #     fromdate=dt.datetime(2020, 4, 1),
    #     # todate=dt.datetime(2021, 2, 15),
    #     todate=dt.datetime.now(),
    #     reverse=False)
    
    
    ######### US    # 不能开 VPN   https://algotrading101.com/learn/yahoo-finance-api-guide/

    from yahoo_fin.stock_info import get_data
    df = get_data(stock, start_date="3/04/2019", end_date="3/04/2021", index_as_date=True, interval="1d")
    data = bt.feeds.PandasData(dataname=df)

    
    
    ######### A
    # df=aquire_CN(stock='002600.SZ',start_date='20180101',end_date='20181011')
    # data = bt.feeds.PandasData(dataname=df)
    ##############
    
    

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(12000.0)

    # Add a FixedSize sizer according to the stake
    # cerebro.addsizer(bt.sizers.FixedSize,stake=10)
    cerebro.addsizer(bt.sizers.PercentSizer,percents=99)

    # Analyzer
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(btanalyzers.Returns, _name='myreturns')
    # cerebro.addanalyzer(btanalyzers.Returns, _name='PyFolio')
    
    
    
    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    backs=cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    ratio_list=[[x.analyzers.myreturns.get_analysis()['rtot']*100,  # 总收益比例
                 x.analyzers.myreturns.get_analysis()['rnorm100'],  # 年化收益比例
                 x.analyzers.mydrawdown.get_analysis()['max']['drawdown'], # 最大回撤比例
                 x.analyzers.mysharpe.get_analysis()['sharperatio'],  # 夏普比率
                 ]for x in backs]
    ratio_df=pd.DataFrame(ratio_list,columns=['Total_return','APR','DrawDown','Shapre_ratio'])
    print(ratio_df)
    # Print out the final result
    # pyfolio_stats=backs[0].analyzers.getbyname('PyFolio')
    # returns,positions,transactions,gross_lev=pyfolio_stats.get_pf.items()
    # returns.index=returns.index.tz_convert(None)
    # quantstats.reports.html(returns,output='results/%s result.html'%(stock),title=stock+'analysis')
    
    
    #Plot
    cerebro.plot(style='candle')
