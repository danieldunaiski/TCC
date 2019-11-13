# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:56:55 2019

@author: danie
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy
from math import sqrt

df = pd.read_excel('C:/Users/danie/Desktop/tcc/resultados.xlsx')

df2 = df.iloc[10:,:]


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2



y_true = df2.iloc[:,0]

y_tests = df2.iloc[:,1:]

for test in y_tests.columns:
    y_t = y_tests[test]
    print(test)
    print('RMSE', round(sqrt(mean_squared_error(y_true,y_t)), 3))
    print('R squared', round(rsquared(y_true,y_t), 3))
    print('MAE', round(mean_absolute_error(y_true,y_t), 3))
    print('MAPE', round(mean_absolute_percentage_error(y_true,y_t), 3))
    print('\n')
    
