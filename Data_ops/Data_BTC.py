import pandas as pd
import json
import requests
import datetime as dt


def get_binance_bars(symbol, interval, startTime, endTime):
    url = 'https://api.binance.com/api/v3/klines'
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'  # 间隔 1000 毫秒 再拿数据
    
    req_params = {'symbol': symbol, 'interval': interval, 'startTime': startTime, 'endTime': endTime, 'limit': limit}
    
    proxies = {
        "http": "http://127.0.0.1:35568",
        "https": "http://127.0.0.1:35568",
    }
    df = pd.DataFrame(json.loads(requests.get(url, params=req_params, proxies=proxies).text))
    if (len(df.index) == 0):
        return None
    
    df = df.iloc[:, :6]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df.open = df.open.astype('float')
    df.high = df.high.astype('float')
    df.low = df.low.astype('float')
    df.close = df.close.astype('float')
    df.volume = df.volume.astype('float')
    
    # 将dataframe的时间戳改成 datetime的格式
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.datetime]
    return df


def get_data():
    df_list = []
    start_datetime = dt.datetime(2021, 1, 1)
    end_datetime=dt.datetime.now()
    
    while True:
        # 实际上获取不到这么长时间的数据，因为单次时长上限是 1000 行，需要拼接
        new_df = get_binance_bars('BTCUSDT', '1d', start_datetime, end_datetime)
        if new_df is None:
            break
        df_list.append(new_df)
        start_datetime = max(new_df.index) + dt.timedelta(0, 1)  # 多加1秒，就能读到下个交易日
    df = pd.concat(df_list)
    return df

if __name__=='__main__':
    df=get_data()
    print(df)
