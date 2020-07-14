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
    'item_id',
    'target_value', # price
    'modelyear',
    'regist_year',
    'mileage',
    'transmission',
    'fuel',
    'plate_num',
    'color',
    'noaccident',
    'yeswarranty',
    'options'
]
df = pd.read_csv('carmanager.csv', sep=',', names=names)

options = set()
for opt in (x for x in df['options'] if isinstance(x, str)):
    for _opt in opt.split('|'):
        options.add(_opt)


def gen_nm_yr_color(row) -> str:
    nm = row.get('item_id', 'xxx').strip()
    yr = str(row.get('modelyear', 'xxx')).strip()
    color = str(row.get('color', 'xxx')).strip()
    return norm(f'{nm or "xxx"} {yr or "xxx"} {color or "xxx"}')

def adjust_price(row) -> int:
    return int(row.get('target_value', 0))

# truncate off the scraping date to the day
df['timestamp'] = df.apply(lambda x: dt.utcfromtimestamp(x.get('timestamp')).strftime('%Y-%m-%d 00:00:00'), axis=1)
df['name_year_color'] = df.apply(lambda x: gen_nm_yr_color(x), axis=1)
df['target_value'] = df.apply(lambda x: adjust_price(x), axis=1)
df = df[~df['name_year_color'].str.contains('xxx')]
df = df[df['target_value'] != 0]

opt_count = 0
option_alias = {}
selected_options = [
    #"파노라마썬루프",
    "스마트키",
    "네비게이션",
    #"썬루프",
    "후방카메라",
    "썬팅",
    #"GPS",
    "블랙박스",
]
#for option in options:
for option in selected_options:
    print(f'option: {option}')
    df[option] = df.apply(lambda x: '1' if isinstance(x.get('options'), str) and option in x['options'] else '0', axis=1)
    option_alias[option] = f'opt{opt_count}'
    opt_count += 1


options = list(options)
df = df.drop_duplicates()

df = df[['timestamp', 'target_value', 'item_id', 'modelyear', 'color', 'mileage', 'noaccident'] + selected_options]
df['modelyear'] = df.apply(lambda x: str(x.get('modelyear')), axis=1)
df['mileage'] = df.apply(lambda x: str(x.get('mileage')), axis=1)
df['noaccident'] = df.apply(lambda x: str(x.get('noaccident')), axis=1)
df = df.dropna()
df = df.sort_values(by=['timestamp'])

df.to_csv('carmanager_forecast.csv', index=False, header=False)
headers = list(df.columns.values)
attributes = []
for h in headers:
    t = str(df[h].dtype)
    if h == 'timestamp':
        t = 'timestamp'
    elif t == 'object':
        t = 'string'
    elif t == 'int64':
        t = 'integer'

    attributes.append({
        "AttributeName": option_alias[h] if h in option_alias else h,
        "AttributeType": t,
    })

with io.open('carmanager_forecast.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps({'Attributes': attributes}, ensure_ascii=False, indent=4))

with io.open('option_alias.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(option_alias, ensure_ascii=False, indent=4))

