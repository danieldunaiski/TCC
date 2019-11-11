# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:05:52 2019

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


df = pd.read_excel('C:/Users/danie/Desktop/tcc/final/df_call.xlsx')
df = df[['Vol']]


# REDE VANILLA

np.random.seed(42)

train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size,:], df.iloc[train_size:len(df),:]


scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or np array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

n_lags = 10

train_sup = series_to_supervised(train_scaled, n_in = n_lags)
test_sup = series_to_supervised(test_scaled, n_in = n_lags)

x_train = train_sup.iloc[:,:-1]
y_train = train_sup.iloc[:,-1]

x_test = test_sup.iloc[:,:-1]
y_test = test_sup.iloc[:,-1]

from keras.regularizers import l1,l2



np.random.seed(42)
reg_alpha = 0.00001


model = Sequential()
model.add(Dense(20, activation='relu',input_dim=len(x_train.columns), kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
model.add(Dense(10, activation='relu',kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
model.add(Dense(1, kernel_regularizer = l1(reg_alpha), activity_regularizer = l1(reg_alpha)))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adadelta')
# fit the keras model on the dataset
history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test,y_test))


#plt.plot(history.history['loss'][10:], label='train')
#plt.plot(history.history['val_loss'][10:], label='test')
#plt.legend()
#plt.show()



ynew = model.predict(x_test)
y_pred = scaler.inverse_transform(ynew)

mean_squared_error(test[n_lags:],y_pred)

## LSTM


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 

# choose a number of time steps
n_steps = 10
np.random.seed(42)
from keras.regularizers import L1L2

# split into samples
# summarize the data
trainX, trainY = split_sequence(train_scaled, n_steps)
testX, testY = split_sequence(test_scaled, n_steps)

np.random.seed(42)
reg = 0.01
# reshape into X=t and Y=t+1

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(GRU(50, input_shape=(1, n_steps)))#, return_sequences=True))
#model.add(GRU(10))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adadelta')
history = model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=1, validation_data=(testX,testY))

#plt.plot(history.history['loss'][10:], label='train')
#plt.plot(history.history['val_loss'][10:], label='test')
#plt.legend()
#plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)

testPredict = scaler.inverse_transform(testPredict)

# calculate root mean squared error

mean_squared_error(test['Vol'][n_steps:], testPredict[:,0])



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








