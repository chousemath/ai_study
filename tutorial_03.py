from PIL import Image
import os
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
font_name = 'NotoSansKR-Light.otf'
font_size = 12
fontprop = fm.FontProperties(fname=font_name, size=font_size)

# convert series to supervised learning
# A time series is a sequence of numbers that are ordered by a time index.
# supervised learning problem is comprised of
# input patterns (X) and output patterns (y),
# such that an algorithm can learn how to
# predict the output patterns from the input
# patterns.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
	# input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        for i in range(0, n_out):
	    # forecast sequence (t, t+1, ... t+n)
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
	# drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

for fname in [x for x in os.listdir('data') if '.csv' in x]:
    # load dataset
    dataset = read_csv(os.path.join('data', fname), header=0, index_col=0)
    num_rows = len(dataset.index)
    #dataset.iloc[:, 0] = datetime.fromtimestamp(dataset.iloc[:, 0])
    #dataset = dataset[['date', 'price', 'mileage']]
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify the number of lag hours
    n_hours = 5
    n_features = 2
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_hours, 1)
    
    # split into train and test sets
    values = reframed.values
    n_train_hours = num_rows // 2
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=2, validation_data=(test_X, test_y), verbose=0, shuffle=True)
    
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    rmse = ('RMSE: %.3f' % rmse)
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    car_name = fname.replace('.csv', '')
    title = f'{car_name}: {rmse}'
    print(title)
    plt.title(title, fontproperties=fontprop)
    plt.legend()
    #plt.show()
    im_name = os.path.join('data_images', f'{car_name}.png')
    plt.savefig(im_name)
    Image.open(im_name).save(im_name, 'PNG')
    plt.clf()
