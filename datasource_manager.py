import shioaji as sj
from shioaji.data import Kbars

from importlib import import_module

import pandas as pd
from IPython.display import display

import mpl_finance as mpf
import matplotlib.dates as mpl_dates

import matplotlib.pyplot as plt

import time


class DataSourceManager:
    "Access stock data"

    def __init__(self):
        pass

    def make_connection_sino(self, pid, passwd):

        self.api = sj.Shioaji()

        self.person_id = pid
        self.passwd = passwd

        self.api.login(
            person_id=self.person_id, 
            passwd=self.passwd, 
            contracts_cb=lambda security_type: print(f"{repr(security_type)} fetch done.")
        )

    def dump_ticks_to_file_sino(self, stock_num_str, date_str):
        ticks = self.api.ticks(self.api.Contracts.Stocks[stock_num_str], date_str)
        df_tick = pd.DataFrame({**ticks})
        df_tick.ts = pd.to_datetime(df_tick.ts)
        df_tick.head()

        df_tick.to_csv('{}_ticks_{}.csv'.format(stock_num_str, date_str), index = False)

    def dump_kbars_to_file_sino(self, stock_num, date_start_str, date_end_str):
        kbars = self.api.kbars(self.api.Contracts.Stocks[stock_num], start=date_start_str, end=date_end_str)
        df_kbars = pd.DataFrame({**kbars})
        df_kbars.ts = pd.to_datetime(df_kbars.ts)

        df_kbars.to_csv('{}_kbars_{}_{}.csv'.format(stock_num, date_start_str, date_end_str), index = False)

        display(df_kbars)

    def load_kbars_csv(self, filepath):
        return pd.read_csv(filepath)

    def kbars_1minute2daily(self, df_min_kbars):
        # df_min_kbars.set_index('ts')
        df_min_kbars.index=pd.to_datetime(df_min_kbars.ts, format='%Y-%m-%d %H:%M:%S')
        entry_open = df_min_kbars.groupby(df_min_kbars.index.date).head(1).index.strftime('%Y-%m-%d %H:%M:%S')
        entry_close = df_min_kbars.groupby(df_min_kbars.index.date).tail(1).index.strftime('%Y-%m-%d %H:%M:%S')
        entry_high = df_min_kbars.loc[df_min_kbars.groupby(df_min_kbars.index.date)["High"].idxmax()].index.strftime('%Y-%m-%d %H:%M:%S')
        entry_low = df_min_kbars.loc[df_min_kbars.groupby(df_min_kbars.index.date)["Low"].idxmin()].index.strftime('%Y-%m-%d %H:%M:%S')

        # Find the value directly
        # entry_high_value = df_min_kbars.groupby(df_min_kbars.index.date)["High"].max()

        kbars_daily_open = df_min_kbars[df_min_kbars.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_open)]
        kbars_daily_open.index = pd.to_datetime(kbars_daily_open.ts, format='%Y-%m-%d').dt.date
        kbars_daily_close = df_min_kbars[df_min_kbars.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_close)]
        kbars_daily_close.index = pd.to_datetime(kbars_daily_close.ts, format='%Y-%m-%d').dt.date
        kbars_daily_high = df_min_kbars[df_min_kbars.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_high)]
        kbars_daily_high.index = pd.to_datetime(kbars_daily_high.ts, format='%Y-%m-%d').dt.date
        kbars_daily_low = df_min_kbars[df_min_kbars.index.strftime('%Y-%m-%d %H:%M:%S').isin(entry_low)]
        kbars_daily_low.index = pd.to_datetime(kbars_daily_low.ts, format='%Y-%m-%d').dt.date

        display(kbars_daily_open)
        display(kbars_daily_close)
        display(kbars_daily_high)
        display(kbars_daily_low)

        kbars_daily = pd.concat([kbars_daily_open[("Open")], kbars_daily_high["High"], kbars_daily_low["Low"], kbars_daily_close["Close"]], axis=1) 

        kbars_daily.index.name = "Date"
        kbars_daily['Date'] = pd.to_datetime(kbars_daily.index, format='%Y-%m-%d')
        kbars_daily = kbars_daily.reindex(columns=['Date','Open','High','Low','Close'])
        

        return kbars_daily

    def display_daily_candlestick(self, df_kbars_daily):
        # Show the candle stick
        # https://tn00343140a.pixnet.net/blog/post/278811668-python%E8%A9%A6%E8%91%97%E7%94%A8matplotlib%E7%95%AB%E5%87%BAk%E7%B7%9A%E5%9C%96

        fig, ax = plt.subplots()    
        
        df_kbars_daily['Date'] = df_kbars_daily['Date'].apply(mpl_dates.date2num)

        mpf.candlestick_ohlc(ax, df_kbars_daily.values, width=0.6, colorup='red', colordown='green')
        # ax.set_xticks(range(0, len(df_kbars_daily.index), 2))
        # ax.set_xticklabels(df_kbars_daily["Date"].index[::2])
        # mpf.candlestick2_ochl(ax, df_kbars_daily['Open'], df_kbars_daily['Close'], df_kbars_daily['High'],
        #                   df_kbars_daily['Low'], width=0.6, colorup='r', colordown='g', alpha=0.75)

        date_format = mpl_dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()
        plt.show()

        df_kbars_daily['Date'] = df_kbars_daily['Date'].apply(mpl_dates.num2date)


    """
    File format:
        Gmt time	            Open	High	Low	Close	Volume
        04.01.2021 00:00:00.000	530	540	528	536	38770328
    """
    def kbars2csv_dailygmttime_ohlcv(self, df_kbars, filepath):
        

        # df = pd.DataFrame(columns=['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume'])

        output_df = df_kbars[['Date', 'Open', 'High', 'Low', 'Close']].copy()
        output_df['Volume'] = 0.0
        output_df.columns = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']

        # output_df['Gmt time'] = (output_df['Gmt time'].dt.strftime('%d.%m.%Y'))+pd.DateOffset(hours=0)

        # output_df['Gmt time'] +=  pd.to_timedelta(0, unit='s')

        output_df['Gmt time'] = output_df['Gmt time'].dt.strftime('%d.%m.%Y %H:%M:%S.000')

        output_df.to_csv(filepath, index=False)

        

        