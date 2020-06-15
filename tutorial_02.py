import pandas as pd
import numpy as np
import datetime as dt
import os

names = [
    'timestamp',
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

def to_hour(row):
    ts = row.get('timestamp', 0)
    return dt.datetime.utcfromtimestamp(ts).hour

def to_year(row):
    ts = row.get('timestamp', 0)
    return dt.datetime.utcfromtimestamp(ts).year

def to_month(row):
    ts = row.get('timestamp', 0)
    return dt.datetime.utcfromtimestamp(ts).month

def to_day(row):
    ts = row.get('timestamp', 0)
    return dt.datetime.utcfromtimestamp(ts).day

def gen_nm_yr(row) -> str:
    nm = row.get('name', 'xxx')
    yr = row.get('year', 'xxx')
    return f'{nm} {yr}'

def adjust_price(row) -> int:
    return int(row.get('price', 0) / 10_000)

df['name_year'] = df.apply(lambda x: gen_nm_yr(x), axis=1)
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)
df['xhour'] = df.apply(lambda x: to_hour(x), axis=1)
df['xyear'] = df.apply(lambda x: to_year(x), axis=1)
df['xmonth'] = df.apply(lambda x: to_month(x), axis=1)
df['xday'] = df.apply(lambda x: to_day(x), axis=1)

df = df[~df['name_year'].str.contains('xxx')]
df = df[df['price'] != 0]
df = df.drop_duplicates(subset=['price', 'name_year', 'plate_num', 'xyear', 'xmonth', 'xday'])
df = df.drop([
    'transmission',
    'fuel',
    'options',
    'timestamp',
    'name',
    'year',
    'regist_year',
    'plate_num',
], axis=1)

#price  mileage                        name_year  xyear  xmonth  xday

# attributes to consider
# price
# mileage
for ny in df.name_year.unique():
    dfx = df[df.name_year == ny]
    dfx['No'] = np.arange(len(dfx))
    dfx = dfx[['No', 'xyear', 'xmonth', 'xday', 'xhour', 'price', 'mileage']]
    dfx.to_csv(os.path.join('data', f'{ny}.csv'), index=False)

