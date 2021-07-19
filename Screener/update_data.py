from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import time


# https://zhuanlan.zhihu.com/p/35360694 plot
def process_data(tickers):
    for stock in tickers:
        try:
            df = pd.read_csv('./latest_data/'+f'{stock}.csv', index_col=0)
            sma = [20, 50, 150]
            MA=[]
            for x in sma:
                df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)
                MA.append(df["SMA_" + str(x)][-1])
                
            RV=df['Volume'][-10:].max()/df['Volume'][-50:].mean()
            currentClose = df["Adj Close"][-1]
            

            condition_1=currentClose>MA[0]
            condition_5=currentClose<MA[1]
            condition_2=RV>2
            condition_3=df['Adj Close'][-10:].pct_change().max()<0.4
            condition_4=df['Adj Close'][-10:].pct_change().min()>-0.4

            # if ( condition_1 and condition_2 and condition_3 and condition_4 and condition_5):
            #     print(stock)
            
            if condition_5:
                print(stock)
                import matplotlib.pyplot as plt
                plt.plot(df["Adj Close"][-50:])
                plt.plot(df["SMA_20"][-50:])
                plt.plot(df["SMA_50"][-50:])
                plt.title("Price Change")
                
                plt.grid(True)
                plt.show()
                
        except Exception as e:
            print(e)
            print(f"Could not gather data on {stock}")
            
if __name__=='__main__':
    yf.pdr_override()
    
    # Variables
    tickers = si.tickers_sp500()
    tickers = [item.replace(".", "-") for item in tickers]  # Yahoo Finance uses dashes instead of dots
    index_name = '^GSPC'  # S&P 500
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    end_date = datetime.date.today()
    # for ticker in tickers[:10]:
    #     # Download historical data as CSV for each stock (makes the process faster)
    #     df = pdr.get_data_yahoo(ticker, start_date, end_date)
    #     df.to_csv('./latest_data/'+f'{ticker}.csv')
    #     time.sleep(0.1)

    process_data(tickers[:10])
    

    
            

    
    