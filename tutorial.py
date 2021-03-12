import backtrader as bt
from Strategies.MACross import MaCrossStrategy




# df=get_data()
# print(df.shape[0])


        
if __name__=='__main__':
    # df = get_data()
    
    from yahoo_fin.stock_info import get_data
    # 不能开 VPN   https://algotrading101.com/learn/yahoo-finance-api-guide/
    df=get_data("NIO", start_date="12/04/2019", end_date="3/04/2021",
                index_as_date=True, interval="1d")
    
    data=bt.feeds.PandasData(dataname=df)
    cerebro=bt.Cerebro()
    # Add a strategy
    cerebro.addstrategy(MaCrossStrategy)

    #Add data
    cerebro.adddata(data)

    #Set cash
    cerebro.broker.setcash(30000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer,percents=99)

    #Run
    print('start portfolio value {}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('end portfolio value {}'.format(cerebro.broker.getvalue()))

    #Plot
    cerebro.plot(style='candle')
