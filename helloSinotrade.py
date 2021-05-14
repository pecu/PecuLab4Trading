import shioaji as sj
from shioaji.data import Kbars

import pandas as pd
from IPython.display import display


api = sj.Shioaji()

api.login(
    person_id="D121824611", 
    passwd="koop5725", 
    contracts_cb=lambda security_type: print(f"{repr(security_type)} fetch done.")
)

ticks = api.ticks(api.Contracts.Stocks["2330"], "2020-03-04")
df_tick = pd.DataFrame({**ticks})
df_tick.ts = pd.to_datetime(df_tick.ts)
df_tick.head()

df_tick.to_csv('2330_ticks.csv', index = False)


kbars = api.kbars(api.Contracts.Stocks["2330"], start="2020-06-01", end="2020-07-01")
df_kbars = pd.DataFrame({**kbars})
df_kbars.ts = pd.to_datetime(df_kbars.ts)

df_kbars.to_csv('2330_kbars.csv', index = False)

display(df_kbars)
