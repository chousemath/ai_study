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

def gen_nm_yr(row) -> str:
    nm = row.get('name', 'xxx').strip()
    yr = str(row.get('year', 'xxx')).strip()
    return f'{nm or "xxx"} {yr or "xxx"}'

def adjust_price(row) -> int:
    return int(row.get('price', 0) / 10_000)

df['name_year'] = df.apply(lambda x: gen_nm_yr(x), axis=1)
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)

df = df[~df['name_year'].str.contains('xxx')]
df = df[df['price'] != 0]
df = df.drop_duplicates(subset=['price', 'name_year', 'plate_num'])
df = df.drop([
    'transmission',
    'fuel',
    'options',
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
    dfx = dfx[['timestamp', 'price', 'mileage']]
    # if len(dfx.index) >= 50:
    dfx.to_csv(os.path.join('data', f'{ny}.csv'), index=False)

