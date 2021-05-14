import shioaji as sj
from shioaji.data import Kbars

from importlib import import_module

import pandas as pd
from IPython.display import display


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
        