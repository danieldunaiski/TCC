# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:31:04 2019

@author: danie
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from sklearn.preprocessing import MinMaxScaler
import math
from keras import optimizers
import seaborn as sns



df = pd.read_excel('C:/Users/danie/Desktop/tcc/final/df_call.xlsx')

df = df[['Ibov_fut',  'Spot', 'moneyness', 'Vol']]

df['Ibov_fut'] = df['Ibov_fut']/df['Spot']



np.random.seed(42)

train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size,:], df.iloc[train_size:len(df),:]


scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# REDE VANILLA

# convert series to supervised learning
def series_to_supervised(data, cols_orig, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [(cols_orig[j]+'(t-%d)' % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [(cols_orig[j]+'(t)') for j in range(n_vars)]
		else:
			names += [(cols_orig[j]+'(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# ensure all data is float
    
np.random.seed(42)
n_steps = 10

train_sup = series_to_supervised(train_scaled, df.columns, n_in = n_steps)
test_sup = series_to_supervised(test_scaled, df.columns, n_in = n_steps)

x_train = train_sup.iloc[:,:-len(train.columns)]
y_train = train_sup.iloc[:,-1]

x_test = test_sup.iloc[:,:-len(train.columns)]
y_test = test_sup.iloc[:,-1]

from keras.regularizers import l1,l2



np.random.seed(42)
reg_alpha = 0.0003

np.random.seed(42)
model = Sequential()
model.add(Dense(60, input_dim=len(x_train.columns), activation='relu',kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
model.add(Dense(40, activation='relu',kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
model.add(Dense(20, activation='relu',kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
model.add(Dense(1,kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adadelta')
# fit the keras model on the dataset
history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test,y_test))

#plt.plot(history.history['loss'][20:], label='train')
#plt.plot(history.history['val_loss'][20:], label='test')
#plt.legend()
#plt.show()

ynew = model.predict(x_test)

fill_pred = pd.DataFrame(np.zeros((len(ynew),len(train.columns))))
fill_pred.iloc[:,-1] = ynew

y_pred = scaler.inverse_transform(fill_pred)

mean_squared_error(test[n_steps:]['Vol'],y_pred[:,-1])


#### LSTM

#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/



def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# y = f(x,y)
# choose a number of time steps
np.random.seed(42)
n_steps = 10
# convert into input/output

train_X, train_y = split_sequences(train_scaled, n_steps)
test_X, test_y = split_sequences(test_scaled, n_steps)

n_features = train_X.shape[2]
# define model

np.random.seed(42)
model = Sequential()
model.add(GRU(40, input_shape=(n_steps, n_features)))
model.add(Dense(n_features))


model.compile(optimizer='adam', loss='mse')

# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history


#plt.plot(history.history['loss'], label='train')
#plt.legend()
#plt.show()

y_pred = model.predict(test_X)

y_pred = scaler.inverse_transform(y_pred)

mean_squared_error(test[n_steps:]['Vol'],y_pred[:,-1])




# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[n_steps:len(trainPredict)+n_steps, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(n_steps*2):len(df), :] = testPredict
# plot baseline and predictions
plt.plot(df)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Ajeitar shifts do plot












# y = f(x)


def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

np.random.seed(42)
n_steps = 10
# convert into input/output
train_X, train_y = split_sequences(train_scaled, n_steps)
test_X, test_y = split_sequences(test_scaled, n_steps)

train_X = train_X[:-1]
train_y = train_y[1:]
test_X = test_X[:-1]
test_y = test_y[1:]

# the dataset knows the number of features, e.g. 2
n_features = train_X.shape[2]
# define model
n_steps = 10
np.random.seed(42)
model = Sequential()
model.add(GRU(10, input_shape=(n_steps, n_features),return_sequences=True))
model.add(GRU(5, input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), verbose=1)


# demonstrate prediction
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.legend()
#plt.show()



ynew = model.predict(test_X)

fill_pred = pd.DataFrame(np.zeros((len(ynew),len(test.columns))))
fill_pred.iloc[:,-1] = ynew



y_pred = scaler.inverse_transform(fill_pred)

mean_squared_error(test[-len(y_pred[:,-1]):]['Vol'],y_pred[:,-1])
