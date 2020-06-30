import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#Start and end dat for aquiring data
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)
#Aquiring the apple stock
df_apple = web.DataReader('AAPL', 'yahoo', start, end)
#Savind the apple data to csv
df_apple.to_csv('apple.csv')

df = pd.read_csv('apple.csv', parse_dates=True, index_col=0)

#resamplng to get the OpenHighLowClose for each 10 days, and the sum of the volume for 10 days
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

#Setting the dates for the graph
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

""" print(df[['Open', 'Close']].head())
df['Adj Close'].plot()
df['Close'].plot()
df.plot()
plt.show() """

#Creating a 100day rolling average
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
#ax1.plot(df.index, df['Adj Close'])
#ax1.plot(df.index, df['100ma'])
#ax2.plot(df.index, df['Volume'])
plt.show()