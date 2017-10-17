

import krakenex
import csv
import random
import time
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import svm
import talib
from io import StringIO
import sqlite3


k = krakenex.API()
k.load_key('kraken.key')
underlying = 'XETHZEUR'

a = open('ohlc.csv', 'wb')
a.truncate()
b = csv.writer(a)
queryname = 'OHLC'
interval = 30


out = k.query_public(queryname, {'pair': underlying,'interval' : interval})

#out2 = k.query_private('TradesHistory')
#print(out[underlying])
if 'result' in  out.keys():
    res = out['result']
    lasttime = res['last']
    rawdat = res[underlying]
    #print rawdat[0]
    b.writerows(rawdat)
    #df = pd.DataFrame(rawdat, columns = ['start','open', 'high', 'low', 'close', 'vwp', 'volume', 'count'])
    df = pd.read_csv('ohlc.csv', header = None, names = ['start','open', 'high', 'low', 'close', 'vwp', 'volume', 'count'])
    #print df.head()
    #df.apply(pd.to_numeric, errors='ignore')
    df['start'] = pd.to_datetime(df['start'],unit='s')
    df.set_index('start' , inplace = True)
    # for y in df.columns:
    #     print y,df[y].dtype
    #newdf = df.tail(60)
    newdf = df
    price = np.array(newdf['vwp'])
    h = np.array(newdf['high'])
    l = np.array(newdf['low'])
    c = np.array(newdf['close'])
    #sma  = talib.SMA(price,50)
    m = talib.MACD(price, 12,26,9)

    mom = talib.MOM(price,12 )
    rsi = talib.RSI(price)
    krsi, drsi = talib.STOCHRSI(price)
    roc = talib.ROC(price)
    mean5d = newdf['vwp'].rolling(window=5,center=False).mean()
    mean10d = newdf['vwp'].rolling(window=10,center=False).mean()
    newdf['willR'] = talib.WILLR(h,l,c)
    newdf['krsi'] = krsi
    newdf['drsi'] = drsi
    newdf['disp5d'] = newdf['vwp'] - mean5d
    newdf['disp10d'] = newdf['vwp'] - mean10d
    newdf['cci'] = talib.CCI(h,l,c)
    newdf['roc'] = roc
    #m[1][40:60]
    # newdf['msig'] = np.where((m[1]>m[2]),1,0)
    newdf['msig'] = m[1]-m[2]
    #newdf['mom'] = np.where(mom>0,1,0)
    # newdf['rsi'] = np.where((rsi>50)& (rsi> np.roll(rsi,1)),1,0)
    newdf['rsi'] = rsi
    newdf.dropna(axis = 0, how = 'any', inplace = True)
    newdf.drop(newdf.index[0], inplace = True)
    factors = ['msig','rsi','cci','willR','krsi','drsi','disp5d','disp10d']
    X = newdf[factors]

    X1 = X.tail(2)
    currinput = X1.head(1)
    print currinput
    scaler = joblib.load('/Users/rahul/Coding/ML Crypto/scaler.save')
    currinput = scaler.transform(currinput)
    model = joblib.load('/Users/rahul/Coding/ML Crypto/finalized_model.sav')
    ypredict = model.predict(currinput)
    yproba = model.predict_proba(currinput)
    #b.write(lasttime)
    print "predicted side : " , ypredict
    print "prob: " , yproba[0][1]
else:
    print out
a.close()
print lasttime
