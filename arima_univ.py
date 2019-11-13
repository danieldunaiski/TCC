# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:06:21 2019

@author: danie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# ARIMA (2,1,2)

#Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
	model = ARIMA(Actual, order=(P, D, Q))
	model_fit = model.fit(disp=0)
	prediction = model_fit.forecast()[0][0]
	return prediction
    
#Get exchange rates
ActualData = pd.read_excel('C:/Users/danie/Desktop/tcc/final/df_call.xlsx')
ActualData = ActualData['Vol']


training_mod = sm.tsa.SARIMAX(ActualData[:478], order=(2,1,2))
training_res = training_mod.fit()

results = []
for i in range(478, len(ActualData)):
    model_test = sm.tsa.SARIMAX(ActualData[:i], order=(2,1,2))
    res = model_test.filter(training_res.params)
    results.append(res.forecast().iloc[0])

for i in range(1,11):
    print(i)
    sm.tsa.SARIMAX(ActualData[:478], order=(i,1,8)).fit().aic

#Size of exchange rates
NumberOfElements = len(ActualData)

#Use 70% of data as training, rest 30% to Test model
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = ActualData[0:TrainingSize]
TestData = ActualData[TrainingSize:NumberOfElements]

#new arrays to store actual and predictions
Actual = [x for x in TrainingData]
Predictions = list()


#in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData)):
	ActualValue =  TestData.iloc[timepoint]
	#forcast value
	Prediction = StartARIMAForecasting(Actual, 2,1,2)    
	print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
	#add it in the list
	Predictions.append(Prediction)
	Actual.append(ActualValue)

#Print MSE to see how good the model is
mean_squared_error(TestData, Predictions)



print('Test Mean Squared Error (smaller the better fit): %.3f' % Error)
# plot
plt.plot(TestData.reset_index(drop=True))
plt.plot(Predictions, color='red')
plt.show()


TrainingSize = int(NumberOfElements * 0.7)
TrainingData = ActualData[0:TrainingSize]
x = TrainingData.diff().dropna()






from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_acf(x);
plot_pacf(x);
