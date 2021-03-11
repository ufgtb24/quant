import pandas as pd
import requests
import datetime as dt
import backtrader as bt
import json
import matplotlib.pyplot as plt

def get_binance_bars(symbol,interval,startTime,endTime):
    url='https://api.binance.com/api/v3/klines'
    startTime=str(int(startTime.timestamp()*1000))
    endTime=str(int(endTime.timestamp()*1000))
    limit='1000' # 间隔 1000 毫秒 再拿数据
    
    req_params={'symbol':symbol,'interval':interval,'startTime':startTime,'endTime':endTime,'limit':limit}

    proxies = {
        "http": "http://127.0.0.1:35568",
        "https": "http://127.0.0.1:35568",
    }
    df=pd.DataFrame(json.loads(requests.get(url,params=req_params, proxies=proxies).text))
    if(len(df.index)==0):
        return None
    
    df=df.iloc[:,:6]
    df.columns=['datetime','open','high','low','close','volume']
    df.open=df.open.astype('float')
    df.high=df.high.astype('float')
    df.low=df.low.astype('float')
    df.close=df.close.astype('float')
    df.volume=df.volume.astype('float')
    
    # 将dataframe的时间戳改成 datetime的格式
    df.index=[dt.datetime.fromtimestamp(x/1000.0) for x in df.datetime]
    return df
    
def get_data():
    df_list=[]
    last_datetime=dt.datetime(2021,1,1)
    while True:
        # 实际上获取不到这么长时间的数据，因为单次时长上限是 1000 行，需要拼接
        new_df=get_binance_bars('BTCUSDT','1d',last_datetime,dt.datetime.now())
        if new_df is None:
            break
        df_list.append(new_df)
        last_datetime=max(new_df.index)+dt.timedelta(0,1) # 多加1秒，就能读到下个交易日
    df=pd.concat(df_list)
    return df

# df=get_data()
# print(df.shape[0])

class MaCrossStrategy(bt.Strategy):
    params=(
        ('fast_length',3),
        ('slow_length',10)
    )
    
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    
    def __init__(self):
        ma_fast=bt.ind.SMA(period=self.params.fast_length)
        ma_slow=bt.ind.SMA(period=self.params.slow_length)
        self.crossover=bt.ind.CrossOver(ma_fast,ma_slow)
        
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
            if self.crossover>0:
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.buy()
                
        elif self.crossover<0:
            self.log('SELL CREATE, %.2f' % self.data.close[0])
            self.close()
        
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
