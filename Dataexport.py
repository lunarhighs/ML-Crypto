import pandas as pd
import numpy as no
import matplotlib as plot
import sqlite3

conn = sqlite3.connect('poloniex_0.1.db')
c = conn.cursor()
#c.execute("CREATE TABLE test2(id INTEGER, name, TEXT)")
#conn.commit()

df = pd.read_sql_query("SELECT * FROM candles_USDT_BTC", conn)
for y in df.columns:
    print y,df[y].dtype
