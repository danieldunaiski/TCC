# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:28:52 2019

@author: danie
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

dir_vol = 'C:/Users/danie/Desktop/tcc/csv_git/vol/'
dir_delta = 'C:/Users/danie/Desktop/tcc/csv_git/delta/'
dir_taxas = 'C:/Users/danie/Desktop/tcc/csv_git/taxa/'

##### carrega o historico do spot ibov e CDI

ibov_hist = pd.read_csv('C:/Users/danie/Desktop/tcc/csv_git/Ibov_historico.csv', sep=';')
ibov_hist['Data'] = pd.to_datetime(ibov_hist['Data'], format='%Y%m%d')

cdi_hist = pd.read_csv('C:/Users/danie/Desktop/tcc/csv_git/cdi_historico.csv', sep=';')
cdi_hist['Data'] = pd.to_datetime(cdi_hist['Data'], format='%d/%m/%Y')

###########################
######### carrega a superficie

df = pd.read_csv(dir_vol + '2019_vol_equity.csv',sep=',')

df['Data'] = pd.to_datetime(df['Data'], format='%Y%m%d')
df['Vencimento'] = pd.to_datetime(df['Vencimento'], format='%Y%m%d')

df_call = df[df['Tipo de opcao'] == 'C']
df_put = df[df['Tipo de opcao'] == 'V']

df_call = df_call[['Data', 'Vencimento', 'DU', 'Delta', 'Vol']]
df_put = df_put[['Data', 'Vencimento', 'DU', 'Delta', 'Vol']]


df_call = pd.pivot_table(df_call, values = 'Vol', index=['Data', 'Vencimento', 'DU'], columns = 'Delta').reset_index()
df_put = pd.pivot_table(df_put, values = 'Vol', index=['Data', 'Vencimento', 'DU'], columns = 'Delta').reset_index()

df_call.columns = ['Data', 'Vencimento', 'DU', 'Delta_1','Delta_10','Delta_25','Delta_37','Delta_50','Delta_63','Delta_75','Delta_90','Delta_99']
df_put.columns = ['Data', 'Vencimento', 'DU', 'Delta_1','Delta_10','Delta_25','Delta_37','Delta_50','Delta_63','Delta_75','Delta_90','Delta_99']

###################### Carrega curvas 

curva_di = pd.read_csv(dir_taxas + '2019_di.csv')
curva_ibov = pd.read_csv(dir_taxas + '2019_ibov.csv')

curva_di = curva_di[['Data do Pregao', 'DU', 'Taxa']]
curva_ibov = curva_ibov[['Data do Pregao', 'DU', 'Taxa']]

curva_di['Data do Pregao'] = pd.to_datetime(curva_di['Data do Pregao'], format='%Y%m%d')
curva_ibov['Data do Pregao'] = pd.to_datetime(curva_ibov['Data do Pregao'], format='%Y%m%d')

def interpola_di(df_curva, data, du):
    df_day = df_curva[df_curva['Data do Pregao'] == data]
    day_post = df_day[df_day['DU'].ge(du)].iloc[0,1]
    taxa_post = df_day[df_day['DU'].ge(du)].iloc[0,2]
    day_ant = df_day[df_day['DU'].le(du)].iloc[-1,1]
    taxa_ant = df_day[df_day['DU'].le(du)].iloc[-1,2]
    
    if day_post == day_ant:
        return taxa_post
    
    discount_post = (1 + taxa_post/100)**(day_post/252)
    discount_ant = (1 + taxa_ant/100)**(day_ant/252)
    
    taxa_interp = (discount_post/discount_ant)**((du - day_ant)/(day_post - day_ant))
    taxa_interp = (taxa_interp*discount_ant)**(252/du) -1
    taxa_interp = taxa_interp*100
    
    return taxa_interp

def interpola_ibov(df_curva, data, du):
    df_day = df_curva[df_curva['Data do Pregao'] == data]
    day_post = df_day[df_day['DU'].ge(du)].iloc[0,1]
    taxa_post = df_day[df_day['DU'].ge(du)].iloc[0,2]
    day_ant = df_day[df_day['DU'].le(du)].iloc[-1,1]
    taxa_ant = df_day[df_day['DU'].le(du)].iloc[-1,2]
    print(data, du)
    
    if day_post == day_ant:
        return taxa_post

    taxa_interp = (taxa_post/taxa_ant)**((du - day_ant)/(day_post - day_ant))
    taxa_interp = taxa_interp*taxa_ant
    
    return taxa_interp





df_call['Ibov_fut'] = df_call.apply(lambda x: interpola_ibov(curva_ibov, x['Data'], x['DU']), axis=1)
df_call['DI_fut'] = df_call.apply(lambda x: interpola_di(curva_di, x['Data'], x['DU']), axis=1)
df_call = pd.merge(df_call, cdi_hist,on='Data')
df_call = pd.merge(df_call, ibov_hist,on='Data')


df_put['Ibov_fut'] = df_put.apply(lambda x: interpola_ibov(curva_ibov, x['Data'], x['DU']), axis=1)
df_put['DI_fut'] = df_put.apply(lambda x: interpola_di(curva_di, x['Data'], x['DU']), axis=1)
df_put = pd.merge(df_put, cdi_hist,on='Data')
df_put = pd.merge(df_put, ibov_hist,on='Data')

df_call.to_csv('C:/Users/danie/Desktop/tcc/csv_git/call_2019.csv')
df_put.to_csv('C:/Users/danie/Desktop/tcc/csv_git/put_2019.csv')

