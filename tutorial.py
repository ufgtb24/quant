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
    
    # 改成 datetime的格式
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
        last_datetime=max(new_df.index)+dt.timedelta(0,1)
    df=pd.concat(df_list)
    return df

df=get_data()
# df.head()

class MaCrossStrategy(bt.Strategy):
    params=(
        ('fast_length',3),
        ('slow_length',10)
    )
    def __init__(self):
        ma_fast=bt.ind.SMA(period=self.params.fast_length)
        ma_slow=bt.ind.SMA(period=self.params.slow_length)
        self.crossover=bt.ind.CrossOver(ma_fast,ma_slow)
        
    def next(self):
        if not self.position:
            if self.crossover>0:
                self.buy()
        elif self.crossover<0:
            self.close()
        
if __name__=='__main__':
    df=get_data()
    data=bt.feeds.PandasData(dataname=df)
    cerebro=bt.Cerebro()
    # Add a strategy
    cerebro.addstrategy(MaCrossStrategy)
    
    #Add data
    cerebro.adddata(data)
    
    #Set cash
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer,percents=99)
    
    #Run
    print('start portfolio value {}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('end portfolio value {}'.format(cerebro.broker.getvalue()))
    
    #Plot
    cerebro.plot(style='candle')
