# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:14:38 2019

@author: danie
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


df = pd.read_excel('C:/Users/danie/Desktop/tcc/final/df_call.xlsx')

df = df[['Ibov_fut',  'Spot', 'moneyness', 'Vol']]

df['Ibov_fut'] = df['Ibov_fut']/df['Spot']


train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df[0:train_size], df[train_size:len(df)]

train2 = train.copy()
pred = pd.DataFrame()

model = VAR(train2).fit(maxlags=15, ic='aic')
lag_order = model.k_ar

for i in range(len(test)):

    yhat =  model.forecast(train2.values[-lag_order:], 1)
    train2 = train2.append(test.iloc[i,:])
    pred = pred.append(pd.DataFrame(yhat))
    print(i,lag_order)
    
vol_pred = pred.iloc[:,-1]
vol_test = test.iloc[:,-1]
mean_squared_error(vol_test, vol_pred)




model = VAR(train).fit(maxlags=15, ic='aic')
lag_order = model.k_ar
yhat =  model.forecast(train.values[-lag_order:], 1)
yhat

datas = df['Data']

plt.figure(figsize=(12,3))
plt.plot(datas[0:train_size],train['Vol'],label='Treino')
plt.plot(datas[train_size:len(df)],test['Vol'],label='Teste')
plt.legend(loc='upper left')
plt.show()






