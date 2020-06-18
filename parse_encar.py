import pandas as pd
import numpy as np
import datetime as dt
import os
import unicodedata as ud
import urllib.parse


def norm(input: str) -> str:
    return ud.normalize('NFC', urllib.parse.unquote(input))

names = [
    'x',
    'timestamp',
    'name',
    'plate_num',
    'price',
    'mileage',
    'color',
    'year',
    'regist_year',
    'fuel',
    'transmission',
    'options',
    'accident'
]
df1 = pd.read_csv('encar_domestic.csv', sep=',', names=names)
df2 = pd.read_csv('encar_foreign.csv', sep=',', names=names)
df = pd.concat([df1, df2], sort=False)

def gen_nm_yr(row) -> str:
    nm = row.get('name', 'xxx').strip()
    yr = str(row.get('year', 'xxx')).strip()
    return norm(f'{nm or "xxx"} {yr or "xxx"}')

def adjust_price(row) -> int:
    return int(row.get('price', 0))

df['name_year'] = df.apply(lambda x: gen_nm_yr(x), axis=1)
df = df[~df['price'].isnull()]
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)
df['plate_num'] = df.apply(lambda x: norm(x.get('plate_num')), axis=1)

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
    'accident',
    'x',
], axis=1)
df = df[['timestamp', 'name_year', 'plate_num', 'price', 'mileage']]

# intentionally shrink the file
df = df.head(50_000)
df.to_csv(os.path.join('encar', f'encar.csv'), index=False)
df.to_excel(os.path.join('encar', 'encar.xlsx'))
