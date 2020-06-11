from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
font_name = 'NotoSansKR-Light.otf'
font_size = 12
fontprop = fm.FontProperties(fname=font_name, size=font_size)


names = [
    'name',
    'price',
    'year',
    'regist_year',
    'mileage',
    'transmission',
    'fuel',
    'plate_num',
    'options'
]
df = pd.read_csv('carmanager.csv', sep=',', names=names)

def gen_name_year(row):
    name = row.get('name', '')
    year = row.get('year', '')
    return f'{name} {year}'

def gen_plate_price(row):
    plate_num = row.get('plate_num', '')
    price = row.get('price', '')
    return f'{plate_num} {price}'

df = df.drop_duplicates()
df = df[df['price'] < 90_000_000]
df['name_cleaned'] = df.apply(lambda x: x['name'].strip(), axis=1)
df = df.replace('None', np.nan)
df = df.replace('', np.nan)
df = df.dropna()
most_freq = df['name'].mode().to_numpy()[0]
df_most = df[df.name_cleaned.eq(most_freq)]
df_most.sort_values(by=['year'])

years = list(set(df_most['year'].tolist()))
years.sort()

prices = list(set(df_most['price'].tolist()))
prices.sort()
prices = range(prices[0], prices[-1]+1, 2_000_000)

plt.figure(figsize = (7,7), tight_layout=True)
plt.scatter(df_most['year'], df_most['price'])
m, b = np.polyfit(df_most['year'], df_most['price'], 1)
plt.plot(df_most['year'], m*df_most['year'] + b, 'g-')

def fmt_price(price: int) -> str:
    return f'{round(price / 10_1000)}만원'

plt.xticks(years, years, rotation=45, fontproperties=fontprop)
plt.yticks(prices, [fmt_price(p) for p in prices], rotation=45, fontproperties=fontprop)

plt.xlabel('년식', fontsize=font_size, fontproperties=fontprop)
plt.ylabel('판매가', fontsize=font_size, fontproperties=fontprop)
plt.title(f'{most_freq}, Y = {int(round(m))}X + {int(round(b))}', fontproperties=fontprop)

plt.show()
