import tushare as ts
import pandas as pd
token = '3b3f2d29a13f4e9be116b72d1e9fcb6abf1acffa659b5f5d01bba5b4'
ts.set_token(token)



def aquire_CN(stock,start_date,end_date):
    df = ts.pro_bar(ts_code=stock, adj='qfq', start_date=start_date, end_date=end_date)
    dates = pd.to_datetime(df['trade_date'])
    df = df[['open', 'high', 'low', 'close', 'vol']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = dates
    df.sort_index(ascending=True, inplace=True)
    return df




if __name__=='__main__':
    df=aquire_CN(stock='002600.SZ',start_date='20180101',end_date=None)
    print(df)