import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import unicodedata as ud
import urllib.parse


def norm(input: str) -> str:
    return ud.normalize('NFC', urllib.parse.unquote(input))

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
    'color',
    'no_accident',
    'yes_warranty',
    'options'
]
df = pd.read_csv('carmanager.csv', sep=',', names=names)

def gen_nm_yr(row) -> str:
    nm = row.get('name', 'xxx').strip()
    yr = str(row.get('year', 'xxx')).strip()
    return norm(f'{nm or "xxx"} {yr or "xxx"}')

def adjust_price(row) -> int:
    return int(row.get('price', 0) / 10_000)

df['name_year'] = df.apply(lambda x: gen_nm_yr(x), axis=1)
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)
df['plate_num'] = df.apply(lambda x: norm(x.get('plate_num')), axis=1)
df['date'] = df.apply(lambda x: dt.utcfromtimestamp(x.get('timestamp')).strftime('%Y-%m-%d'), axis=1)

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
    'timestamp',
], axis=1)
df = df[['date', 'name_year', 'plate_num', 'price', 'mileage', 'color', 'no_accident', 'yes_warranty']]
df = df.sort_values(by=['date'])
df.to_csv(os.path.join('phoenix', f'phoenix.csv'), index=False)
df.to_excel(os.path.join('phoenix', 'phoenix.xlsx'))
