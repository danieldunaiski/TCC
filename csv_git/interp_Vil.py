# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:34:50 2019

@author: danie
"""

import pandas as pd

df = pd.read_excel('C:/Users/danie/Desktop/tcc/csv_git/calls.xlsx')

df_simples = df[['Data', 'DU', 'Delta_10',
       'Delta_25', 'Delta_37', 'Delta_50', 'Delta_63', 'Delta_75', 'Delta_90']]

def calc_interp_vol (t,day,df,delta):
    temp = df[df['Data'] == day]
    lista_dias = list(temp['DU'])
    
    if t > max(lista_dias):
        t_nex = lista_dias[-1]
        t_bef = lista_dias[-2]
    elif t < min(lista_dias):
        t_bef = lista_dias[0]
        t_nex = lista_dias[1]
    else:
        t_bef = max(x for x in lista_dias if x <= t)
        t_nex = min(x for x in lista_dias if x >= t)
        
    dict_delta_pos = {10:2, 25:3, 37:4, 50:5, 63:6, 75:7, 90:8}
    
    vol_nex = temp[temp['DU'] == t_nex].iloc[0, dict_delta_pos[delta]]
    vol_bef = temp[temp['DU'] == t_bef].iloc[0, dict_delta_pos[delta]]
    
    if t == t_nex:
        return vol_nex
    if t == t_bef:
        return vol_bef
    vol = (t-t_bef)/(t_nex-t_bef)*t_nex/t*vol_nex**2 + (t_nex-t)/(t_nex-t_bef)*t_bef/t*vol_bef**2
    return vol**.5

new_days = [21,42,63,84,105,126,159,252,315]

from itertools import product

new_df = pd.DataFrame(list(product(df_simples['Data'].unique(), new_days)), columns=['Data','DU'])
new_df['Vol_10'] = 0
new_df['Vol_25'] = 0
new_df['Vol_37'] = 0
new_df['Vol_50'] = 0
new_df['Vol_63'] = 0
new_df['Vol_75'] = 0
new_df['Vol_90'] = 0

new_df['Vol_10'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 10), axis=1)
new_df['Vol_25'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 25), axis=1)
new_df['Vol_37'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 37), axis=1)
new_df['Vol_50'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 50), axis=1)
new_df['Vol_63'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 63), axis=1)
new_df['Vol_75'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 75), axis=1)
new_df['Vol_90'] = new_df.apply(lambda x: calc_interp_vol(x['DU'], x['Data'], df_simples, 90), axis=1)





###################### Carrega curvas 

curva_di = pd.read_excel('C:/Users/danie/Desktop/tcc/csv_git/di.xlsx')
curva_ibov = pd.read_excel('C:/Users/danie/Desktop/tcc/csv_git/ibov_fut.xlsx')

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


teste = new_df.copy()


teste['Ibov_fut'] = teste.apply(lambda x: interpola_ibov(curva_ibov, x['Data'], x['DU']), axis=1)
teste['DI_fut'] = teste.apply(lambda x: interpola_di(curva_di, x['Data'], x['DU']), axis=1)


ibov_hist = pd.read_csv('C:/Users/danie/Desktop/tcc/csv_git/Ibov_historico.csv', sep=';')
ibov_hist['Data'] = pd.to_datetime(ibov_hist['Data'], format='%Y%m%d')

cdi_hist = pd.read_csv('C:/Users/danie/Desktop/tcc/csv_git/cdi_historico.csv', sep=';')
cdi_hist['Data'] = pd.to_datetime(cdi_hist['Data'], format='%d/%m/%Y')

teste = pd.merge(teste, cdi_hist,on='Data')
teste = pd.merge(teste, ibov_hist,on='Data')

teste = teste[['Data', 'DU', 'Vol_10', 'Vol_25', 'Vol_37', 'Vol_50', 'Vol_63',
       'Vol_75', 'Vol_90', 'Ibov_fut', 'DI_fut', 'CDI', 'Last']]

teste.to_excel('C:/Users/danie/Desktop/tcc/csv_git/calls_tenor_cte.xlsx', engine = 'openpyxl')


m = pd.melt(teste, id_vars =['Data', 'DU', 'Ibov_fut', 'DI_fut', 'CDI', 'Last'], 
            value_vars=['Vol_10', 'Vol_25', 'Vol_37', 'Vol_50', 'Vol_63','Vol_75', 'Vol_90'], 
            var_name='Delta',value_name='Vol')


m.to_excel('C:/Users/danie/Desktop/tcc/csv_git/calls_tenor_cte_melt.xlsx', engine = 'openpyxl')



