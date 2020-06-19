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
    return int(row.get('price', 0))

def format_price(row) -> str:
    num = row.get('price')
    return f"{num:,}"

def format_mileage(row) -> str:
    num = row.get('mileage')
    return f"{num:,}KM"

def format_regist_year(row) -> str:
    num = row.get('regist_year', '0000.00.00')
    return str(num).replace('-', '.').strip()


df['name_year'] = df.apply(lambda x: gen_nm_yr(x), axis=1)
df['regist_year'] = df.apply(lambda x: format_regist_year(x), axis=1)
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)
df['plate_num'] = df.apply(lambda x: norm(x.get('plate_num')), axis=1)
df['date'] = df.apply(lambda x: dt.utcfromtimestamp(x.get('timestamp')).strftime('%Y.%m.%d'), axis=1)

df = df[~df['name_year'].str.contains('xxx')]
df = df[df['price'] != 0]
df['price'] = df.apply(lambda x: format_price(x), axis=1)
df['mileage'] = df.apply(lambda x: format_mileage(x), axis=1)
#df = df.drop_duplicates(subset=['price', 'name_year', 'plate_num'])
df = df[['date', 'name', 'price', 'year', 'regist_year', 'mileage', 'transmission', 'fuel', 'plate_num', 'color', 'no_accident', 'yes_warranty']]
df = df.sort_values(by=['date'])
headers = ['날짜', '모델명', '차량가격', '형식년도', '최초등록일', '마일리지', '변속', '연료', '자동차 번호', '색상', '무사고', '워런티']
df.columns = [norm(x) for x in headers]

num_rows = len(df.index)

count = 0
for x in range(10_000, num_rows, 10_000):
    df1 = df.head(x).tail(10_000)
    ext = f'_{count * 10_000}-{min((count + 1) * 10_000, num_rows)}'
    df1.to_csv(os.path.join('phoenix', f'phoenix{ext}.csv'), index=False)
    df1.to_excel(os.path.join('phoenix', f'phoenix{ext}.xlsx'))
    count += 1
