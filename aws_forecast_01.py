import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import unicodedata as ud
import urllib.parse
import json
import io


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

options = set()
for opt in (x for x in df['options'] if isinstance(x, str)):
    for _opt in opt.split('|'):
        options.add(_opt)


def gen_nm_yr_color(row) -> str:
    nm = row.get('name', 'xxx').strip()
    yr = str(row.get('year', 'xxx')).strip()
    color = str(row.get('color', 'xxx')).strip()
    return norm(f'{nm or "xxx"} {yr or "xxx"} {color or "xxx"}')

def adjust_price(row) -> int:
    return int(row.get('price', 0))

# truncate off the scraping date to the day
df['date'] = df.apply(lambda x: dt.utcfromtimestamp(x.get('timestamp')).strftime('%Y-%m-%d 00:00:00'), axis=1)
df['name_year_color'] = df.apply(lambda x: gen_nm_yr_color(x), axis=1)
df['price'] = df.apply(lambda x: adjust_price(x), axis=1)
df = df[~df['name_year_color'].str.contains('xxx')]
df = df[df['price'] != 0]
for option in options:
    print(f'option: {option}')
    df[option] = df.apply(lambda x: 1 if isinstance(x.get('options'), str) and option in x['options'] else 0, axis=1)


options = list(options)
df = df.drop_duplicates()
df = df[['date', 'price', 'name', 'year', 'mileage', 'color', 'no_accident', 'yes_warranty'] + options]
df = df.dropna()
df = df.sort_values(by=['date'])

df.to_csv('carmanager_forecast.csv', index=False, header=False)
headers = list(df.columns.values)
attributes = []
for h in headers:
    attributes.append({
        "AttributeName": h,
        "AttributeType": str(df[h].dtype),
    })

with io.open('carmanager_forecast.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps({'Attributes': attributes}, ensure_ascii=False, indent=4))


