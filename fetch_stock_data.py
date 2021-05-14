import datasource_manager
import pandas as pd
from IPython.display import display
import mpl_finance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

if __name__ == '__main__':
    # datasourceManager = datasource_manager.DataSourceManager()

    # datasourceManager.make_connection_sino("D121824611", "koop5725")

    # datasourceManager.dump_ticks_to_file_sino("2330", "2020-03-04")
    # datasourceManager.dump_kbars_to_file_sino("2330", "2020-03-04", "2020-03-24")

    rawDataframe = pd.read_csv("2330_kbars_2020-03-04_2020-03-24.csv")

    # rawDataframe.set_index('ts')
    rawDataframe.index=pd.to_datetime(rawDataframe.ts, format='%Y-%m-%d %H:%M:%S')
    entry_open = rawDataframe.groupby(rawDataframe.index.date).head(1).index.strftime('%Y-%m-%d %H:%M:%S')
    entry_close = rawDataframe.groupby(rawDataframe.index.date).tail(1).index.strftime('%Y-%m-%d %H:%M:%S')
    entry_high = rawDataframe.loc[rawDataframe.groupby(rawDataframe.index.date)["High"].idxmax()].index.strftime('%Y-%m-%d %H:%M:%S')
    entry_low = rawDataframe.loc[rawDataframe.groupby(rawDataframe.index.date)["Low"].idxmin()].index.strftime('%Y-%m-%d %H:%M:%S')

    # Find the value directly
    # entry_high_value = rawDataframe.groupby(rawDataframe.index.date)["High"].max()

    kbars_daily_open = rawDataframe[rawDataframe.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_open)]
    kbars_daily_open.index = pd.to_datetime(kbars_daily_open.ts, format='%Y-%m-%d').dt.date
    kbars_daily_close = rawDataframe[rawDataframe.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_close)]
    kbars_daily_close.index = pd.to_datetime(kbars_daily_close.ts, format='%Y-%m-%d').dt.date
    kbars_daily_high = rawDataframe[rawDataframe.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_high)]
    kbars_daily_high.index = pd.to_datetime(kbars_daily_high.ts, format='%Y-%m-%d').dt.date
    kbars_daily_low = rawDataframe[rawDataframe.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_low)]
    kbars_daily_low.index = pd.to_datetime(kbars_daily_low.ts, format='%Y-%m-%d').dt.date

    # display(kbars_daily_open)
    # display(kbars_daily_close)
    # display(kbars_daily_high)
    # display(kbars_daily_low)

    kbars_daily = pd.concat([kbars_daily_open[("Open")], kbars_daily_open["High"], kbars_daily_low["Low"], kbars_daily_open["Close"]], axis=1) 

    kbars_daily.index.name = "Date"
    kbars_daily['Date'] = pd.to_datetime(kbars_daily.index, format='%Y-%m-%d')
    kbars_daily = kbars_daily.reindex(columns=['Date','Open','High','Low','Close'])
    kbars_daily['Date'] = kbars_daily['Date'].apply(mpl_dates.date2num)

    display(kbars_daily)


    # Show the candle stick
    # https://tn00343140a.pixnet.net/blog/post/278811668-python%E8%A9%A6%E8%91%97%E7%94%A8matplotlib%E7%95%AB%E5%87%BAk%E7%B7%9A%E5%9C%96

    fig, ax = plt.subplots()    
    
    mpf.candlestick_ohlc(ax, kbars_daily.values, width=0.6, colorup='red', colordown='green')
    # ax.set_xticks(range(0, len(kbars_daily.index), 2))
    # ax.set_xticklabels(kbars_daily["Date"].index[::2])
    # mpf.candlestick2_ochl(ax, kbars_daily['Open'], kbars_daily['Close'], kbars_daily['High'],
    #                   kbars_daily['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)

    date_format = mpl_dates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.show()
    