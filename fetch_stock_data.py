import datasource_manager
import pandas as pd
from IPython.display import display

def fetching_and_dump(self):
    pass

def preprocessing(self):
    pass

def make_data_tw2330(self):
    pass

if __name__ == '__main__':
    datasourceManager = datasource_manager.DataSourceManager()

    # datasourceManager.make_connection_sino("D121824611", "koop5725")

    # 台積
    # datasourceManager.dump_ticks_to_file_sino("2330", "2020-03-04")
    # datasourceManager.dump_kbars_to_file_sino("2330", "2020-03-04", "2020-03-24")
    # rawDataframe = datasourceManager.load_kbars_csv("2330_kbars_2020-03-04_2020-03-24.csv")
    # df_kbars_daily = datasourceManager.kbars_1minute2daily(rawDataframe)
    # datasourceManager.kbars2csv_dailygmttime_ohlcv(df_kbars_daily, "2330_kbar_daily_gmt_ohlc.csv")
    # datasourceManager.display_daily_candlestick(df_kbars_daily)
    # display(df_kbars_daily)


    # 聯電
    # datasourceManager.dump_ticks_to_file_sino("2330", "2020-03-04")
    # datasourceManager.dump_kbars_to_file_sino("1229", "2021-01-04", "2021-05-14")
    rawDataframe = datasourceManager.load_kbars_csv("1229_kbars_2021-01-04_2021-05-14.csv")
    df_kbars_daily = datasourceManager.kbars_1minute2daily(rawDataframe)
    datasourceManager.kbars2csv_dailygmttime_ohlcv(df_kbars_daily, "./data/1229_kbar_daily_gmt_ohlc.csv")
    datasourceManager.display_daily_candlestick(df_kbars_daily)
    display(df_kbars_daily)



    
    