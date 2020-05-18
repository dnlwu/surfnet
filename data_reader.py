import pandas_datareader as web
import pandas as pd

# REQUIRES: valid ticker (string), startDate and endDate yyyy-mm-dd (string)
# MODIFIES: none
# EFFECTS: gives stock data in pandas dataframe
def download(ticker,startDate,endDate):
    df = web.DataReader(ticker,'av-daily-adjusted',startDate,endDate,api_key='7V30G7UZ756MYU34')
    return df

def save_df(df,filename):
    df.to_csv(filename)

def read_csv(filename,col_index=False):
    return pd.read_csv(filename,index_col=col_index)

# ticker = 'AAPL'
# start = '2002-01-01'
# end = '2019-01-10'
# df = download(ticker,start,end)
# print(df.head())