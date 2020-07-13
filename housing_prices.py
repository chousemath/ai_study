import os
import unicodedata as ud
import urllib.parse
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

font_name = 'NotoSansKR-Light.otf'
font_size = 12
fontprop = fm.FontProperties(fname=font_name, size=font_size)

def norm(input: str) -> str:
    return ud.normalize('NFC', urllib.parse.unquote(input))

def gen_nm_yr(row) -> str:
    nm = row.get('name', 'xxx').strip()
    yr = str(row.get('year', 'xxx')).strip()
    return norm(f'{nm or "xxx"} {yr or "xxx"}').replace('/', '|')

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

df = pd.read_csv('_carmanager.csv', sep=',', names=names, low_memory=False)

df['name_year'] = df.apply(
    lambda x: gen_nm_yr(x),
    axis=1
)

df = df[~df['name_year'].str.contains('xxx')]

df = df[['timestamp', 'name_year', 'price', 'mileage', 'transmission', 'fuel', 'color', 'no_accident']]

df = df.dropna()

df['transmission'] = df['transmission'].astype('category')
df['fuel'] = df['fuel'].astype('category')
df['no_accident'] = df['no_accident'].astype('category')
df['color'] = df.apply(
    lambda x: norm(x.get('color')),
    axis=1
)
df['price'] = df.apply(
    lambda x: x.get('price', 0) // 10_000,
    axis=1
)

#transmissions = {}
#count = 1
#for transmission in df.transmission.unique():
#    transmissions[transmission] = count
#    count += 1
#
#fuels = {}
#count = 1
#for fuel in df.fuel.unique():
#    fuels[fuel] = count
#    count += 1
#

#df['transmission'] = df.apply(
#    lambda x: transmissions[x.get('transmission')],
#    axis=1
#)
#df['fuel'] = df.apply(
#    lambda x: fuels[x.get('fuel')],
#    axis=1
#)

colors = {}
color_values = {}
count = 1

color_counts = []
color_keys = []

for color in df.color.unique():
    color = color if color.strip() else 'n/a'
    colors[color] = count
    color_values[count] = color

    color_counts.append(count)
    color_keys.append(color)

    count += 1

print(colors)
print(color_values)

df['color'] = df.apply(
    lambda x: colors[x.get('color')],
    axis=1
)
df['color_category'] = df['color'].astype('category')

for name_year in df.name_year.unique():
    try:
        df_name_year = df[df['name_year'] == name_year]

        # if the dataset is too small, I don't think
        # the result will be very meaningful
        if len(df_name_year.index) < 10:
            continue

        df1 = df_name_year[['timestamp', 'price', 'mileage', 'transmission', 'fuel', 'color_category', 'no_accident']]
        df1.to_csv(
            os.path.join('models', f'{name_year}.csv'),
            index=False
        )

        dfx = pd.get_dummies(df1, columns=['mileage', 'transmission', 'fuel', 'color_category', 'no_accident'], drop_first=True)
        y = df1['price']
        dfx = dfx.drop(['timestamp', 'price'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(dfx, y, train_size=0.8, random_state=42)

        reg = LinearRegression()
        reg.fit(x_train,y_train)
        coeff = reg.coef_
        rmse = metrics.mean_squared_error(y_test, reg.predict(x_test))
        rsq = reg.score(x_test,y_test)
        if rsq > 0.9:
            print(f'------{name_year}-----')
            print(coeff)
            print('rmse: ', rmse)
            print('rsq:', rsq)

            plot = sns.regplot(x='mileage', y='price', data=df_name_year)
            plot.set_title(name_year, fontproperties=fontprop)
            plot.set_ylabel('차량가격 (만원)', fontproperties=fontprop)
            plot.set_xlabel('마일리지 (KM)', fontproperties=fontprop)
            #plot.axis(xmin=df_name_year['mileage'].min() - 1, xmax=df_name_year['mileage'].max() + 1)
            fig = plot.get_figure()
            fig.savefig(os.path.join('plots_mileage_vs_price', f'{name_year}.png'))
            fig.clf()

            plot = sns.stripplot(x='color', y='price', data=df_name_year[df_name_year['color'].isin(color_counts)])
            plot.set_title(name_year, fontproperties=fontprop)
            plot.set_ylabel('차량가격 (만원)', fontproperties=fontprop)
            plot.set_xlabel('색상', fontproperties=fontprop)
            plot.set_xticks(color_counts)
            plot.set_xticklabels(color_keys, fontproperties=fontprop, rotation=90)
            fig = plot.get_figure()
            fig.savefig(os.path.join('plots_color_vs_price', f'{name_year}.png'))
            fig.clf()
    except Exception as e:
        print(str(e))


