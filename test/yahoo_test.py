# from yahoo_fin.stock_info import get_data
# amazon_weekly= get_data("amzn", start_date="12/04/2009", end_date="12/04/2019", index_as_date = True, interval="1wk")
# print(amazon_weekly.head())

import tushare as ts
import pandas as pd
token = '3b3f2d29a13f4e9be116b72d1e9fcb6abf1acffa659b5f5d01bba5b4'
ts.set_token(token)
df = ts.pro_bar(ts_code='000001.SZ', adj='qfq', start_date='20180101', end_date=None)
dates = pd.to_datetime(df['trade_date'])
df = df[['open', 'high', 'low', 'close', 'vol']]
df.columns = ['open', 'high', 'low', 'close', 'volume']
df.index = dates
df.sort_index(ascending=True, inplace=True)
print(df)
