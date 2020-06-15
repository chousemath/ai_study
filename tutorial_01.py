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

def gen_date(row):
    ts = row.get('timestamp', 0)
    return dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')

def gen_name_year(row):
    name = row.get('name', '')
    year = row.get('year', '')
    return f'{name} {year}'

def gen_plate_price(row):
    plate_num = row.get('plate_num', '')
    price = row.get('price', '')
    return f'{plate_num} {price}'

def fmt_price(price: int) -> str:
    return f'{round(price / 10_1000)}만원'


#df = df[df['price'] < 90_000_000]
df['name_year'] = df.apply(lambda x: gen_name_year(x), axis=1)
df['date'] = df.apply(lambda x: gen_date(x), axis=1)
print(f'max: {df["timestamp"].max()}, min: {df["timestamp"].min()}')
df = df.drop_duplicates(subset=['price', 'name', 'plate_num', 'date'])
df = df.replace('None', np.nan)
df = df.replace('', np.nan)
df = df.dropna()

most_freq = df['name_year'].mode().to_numpy()[0]

df_most = df[df.name_year.eq(most_freq)]
#df_most.sort_values(by=['year'])
num_rows = len(df_most.index)

#years = list(set(df_most['year'].tolist()))
#years.sort()
#
#prices = list(set(df_most['price'].tolist()))
#prices.sort()
#prices = range(prices[0], prices[-1]+1, 2_000_000)
#
#plt.figure(figsize = (7,7), tight_layout=True)
#plt.scatter(df_most['year'], df_most['price'])
#m, b = np.polyfit(df_most['year'], df_most['price'], 1)
#plt.plot(df_most['year'], m*df_most['year'] + b, 'g-')

#plt.xticks(years, years, rotation=45, fontproperties=fontprop)
#plt.yticks(prices, [fmt_price(p) for p in prices], rotation=45, fontproperties=fontprop)
#
#plt.xlabel('년식', fontsize=font_size, fontproperties=fontprop)
#plt.ylabel('판매가', fontsize=font_size, fontproperties=fontprop)
#plt.title(f'{most_freq}, Y = {int(round(m))}X + {int(round(b))}', fontproperties=fontprop)

# plt.show()

# Create x, where x the 'scores' column's values as floats
x = df[['price']].values.astype(float)
# Create a minimum and maximum processor object
min_max_scaler = MinMaxScaler()
# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)
# Run the normalizer on the dataframe
df_norm = pd.DataFrame(x_scaled)
df_norm.columns = ['price_norm']
df_most['price_norm'] = df_norm['price_norm']

mid = num_rows // 2
#train_data = df_most.iloc[:mid, :].to_numpy()
#test_data = df_most.iloc[mid:, :].to_numpy()
train_data = df_most.iloc[:mid, :]['price_norm'].to_numpy()
test_data = df_most.iloc[mid:, :]['price_norm'].to_numpy()

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
#EMA = 0.0
#gamma = 0.1
#for ti in range((num_rows-1)):
#    print(f'index: {ti}')
#    EMA = gamma*train_data[ti] + (1-gamma)*EMA
#    train_data[ti] = EMA


ema_data = {
    'error': 1,
    'predictions': [],
}
for window_size in range(2, 200):
    for decay in np.arange(0.1, 0.8, 0.01):
        N = train_data.size
        run_avg_predictions = []
        run_avg_x = []
        mse_errors = []
        running_mean = 0.0
        run_avg_predictions.append(running_mean)
        
        for pred_idx in range(1,N):
            running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
            run_avg_predictions.append(running_mean)
            mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
            run_avg_x.append(pred_idx)
        
        mse_err = 0.5*np.mean(mse_errors)
        if mse_err < ema_data['error']:
            ema_data['error'] = mse_err
            ema_data['predictions'] = run_avg_predictions
            ema_data['window'] = window_size
            ema_data['decay'] = round(decay, 5)

plt.figure(figsize = (7, 7))
plt.plot(range(df_most.shape[0]), df_most['price_norm'], color='b', label='True')
plt.plot(range(0, N), ema_data['predictions'], color='orange', label='Prediction')
plt.xlabel('Date', fontproperties=fontprop)
plt.ylabel(f'{most_freq} Normalized Price', fontproperties=fontprop)
plt.title('EMA AVG MSE Err: %.5f'%(ema_data['error']) + f', window: {ema_data["window"]}, decay: {ema_data["decay"]}', fontproperties=fontprop)
plt.legend(fontsize=14)
plt.show()
