# Import the backtrader platform
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
# import quantstats

from Data_ops.data_A import aquire_CN
from Strategies.Ketler_stg import Keltler_strategy
from Strategies.MFI_stg import MFI_strategy
from Strategies.RVI_strategy import RVI_stg
from tutorial import MaCrossStrategy



# Create a cerebro entity
cerebro = bt.Cerebro()

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

# Add a strategy
cerebro.addstrategy(Keltler_strategy)

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
ratio_list=[[x.analyzers.myreturns.get_analysis()['rtot']*100,  # 总收益百分比
             x.analyzers.myreturns.get_analysis()['rnorm100'],  # 年化收益百分比
             x.analyzers.mydrawdown.get_analysis()['max']['drawdown'], # 最大回撤百分比
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
