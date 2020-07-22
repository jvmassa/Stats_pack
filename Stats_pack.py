# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:00:18 2020

@author: massa
"""

import numpy as np
import pandas as pd
import scipy.stats as scs
import I_Database as idb
import plotly.graph_objects as go
import plotly.express as px

def Log_Ret(data):
    """
    Calcula o log retorno de uma série temporal de preços e retorna um dataframe
    com os resultados
    """
    log_ret = np.log(data/data.shift(1))
    log_ret = log_ret.dropna()
    return log_ret

def Simp_Ret(data):
    """
    Calcula o retorno simples de uma série temporal de preços e retorna um data
    frame com os resultados
    """
    ret = data.pct_change()
    ret = ret.dropna()
    return ret

def Log_Vol(data):
    """
    Calcula a volatilidade de uma série de log-retornos
    """
    ret = Log_Ret(data)
    vol = ret.std(ddof = 1)
    return vol

def Simp_Vol(data):
    """
    Calcula a volatilidade de uma série de retornos simples
    """    
    ret = Simp_Ret(data)
    vol = ret.std(ddof = 1)
    return vol

def Ann_ret(data, periods = 252, ret_type = 'simp'):
    """
    Calcula o retorno anualizado de um determinado ativo com base em sua série 
    temporal de preços
    """
    if ret_type.lower() == 'simp':
        ret = Simp_Ret(data)
        compound = (1+ret).prod()
        n_periods = data.shape[0]
        ann_ret = compound**(periods/n_periods) - 1
        return ann_ret
    elif ret_type.lower() == 'log':
        ret = Log_Ret(data)
        compound = (1+ret).cumprod()
        n_periods = data.shape[0]
        ann_ret = compound**(periods/n_periods) - 1
        return ann_ret

def Ann_vol(data, periods = 252, ret_type = 'simp'):
    """
    Calcula o retorno anualizado de um determinado ativo com base em sua série 
    temporal de preços
    """
    if ret_type.lower() == 'simp':
        vol = Simp_Vol(data)
        ann_vol = vol * np.sqrt(periods)
        return ann_vol
    elif ret_type.lower() == 'log':
        vol = Log_Vol(data)
        ann_vol = vol * np.sqrt(periods)
        return ann_vol

def Skewness(data):
    """
    Calcula o skewness de uma determinada série temporal
    """
    if isinstance(data, pd.DataFrame):
        return data.aggregate(Skewness)
    elif isinstance(data, pd.Series):
        return scs.stats.skew(data)

def Kurtosis(data):
    """
    Calcula a curtose de uma determinada série temporal
    """
    if isinstance(data, pd.DataFrame):
        return data.aggregate(Kurtosis)
    elif isinstance(data, pd.Series):
        return scs.stats.kurtosis(data)
    
def Z_Score(data, ret_type = 'simp'):
    """
    Calcula o z-score de uma série temporal e retorna um dataframe
    """
    mu = data.mean()
    sigma = data.std(ddof=1)
    z_score = (data-mu)/sigma
    return z_score

def Drawdown(data, ret_type = 'simp'):
    """
    Calcula o drawdown de uma série temporal e retorna um dataframe
    """
    if ret_type.lower() == 'simp':
        ret = Simp_Ret(data)
        compound = (1+ret).cumprod()
        peaks = compound.cummax()
        drawdown = (compound-peaks)/peaks
        return drawdown
    elif ret_type.lower() == 'log':
        ret = Log_Ret(data)
        compound = (1+ret).cumprod()
        peaks = compound.cummax()
        drawdown = (compound-peaks)/peaks
        return drawdown

def VaR_Historico(data,ret_type = 'simp' ,alfa = 5):
    """
    Calcula o V@R Histórico de uma determinada série de retornos para um certo
    nível de significância 
    """
    if ret_type.lower() == 'simp':
        rets = Simp_Ret(data)
    elif ret_type.lower() == 'log':
        rets = Log_Ret(data)
        
    if isinstance(rets, pd.DataFrame):
        return rets.aggregate(VaR_Historico, alfa = alfa)
    elif isinstance(rets, pd.Series):
        return -np.percentile(rets, alfa)

def Ex_Shortfall(data,ret_type = 'simp' , alfa = 5):
    """
    Calcula o Expected Shortfall de uma determinada série de retornos para um 
    certo nível de significância
    """
    if ret_type.lower() == 'simp':       
        rets = Simp_Ret(data)
    elif ret_type.lower() == 'log':
        rets = Log_Ret(data)
    var_aux = np.percentile(rets, alfa)
    aux = rets < var_aux
    C_VaR = -rets[aux].mean()
    return C_VaR

def Rolling_Vol(data, window, periods = 252):
    vol = data.rolling(window).std(ddof =0)*np.sqrt(periods)
    vol = vol.dropna()
    return vol

def Sharpe_Ratio(data, rfree):

        ann_ret = Ann_ret(data)
        ann_vol = Ann_vol(data)
        exc = ann_ret - rfree
        sharpe = exc/ann_vol
        return sharpe
    
def Wealth_Index(data):
    """
    Calcula o wealth index para uma ou mais séries temporais de preços de 
    ativos. 
    """
    rets = Simp_Ret(data)
    wi = (1+rets).cumprod()
    return wi   
    
def Summary(data, rfree):
    ann_ret = Ann_ret(data)
    ann_vol = Ann_vol(data)
    sharpe = Sharpe_Ratio(data,rfree)
    dd = Drawdown(data).min()
    var = data.aggregate(VaR_Historico)
    exs = Ex_Shortfall(data)
    return pd.DataFrame({
            'Ret.Anualizado': ann_ret,
            'Vol.Anualizada': ann_vol,
            'Sharpe': sharpe,
            'V@R': var,
            'Ex.Shortfall':exs,
            'Drawdown': dd})

def Rolling_VaR(data, window = 21):
    """
    Gera a série temporal do V@R histórico de um determinado ativo
    """    
       
    N = data.shape[0]
    windows = [[start, start + window] for start in range(0,N-window)]
    nans = np.repeat(np.nan, window).tolist()
    var = [-VaR_Historico(data[w[0]:w[1]]) for w in windows]
    var = pd.Series(nans + var, index = data.index)
    return var

def Rolling_Shortfall(data, window = 21):
    """
    Gera a série temporal do Expected Shortfall de um determinado ativo
    """   
    N = data.shape[0]
    windows = [[start, start + window] for start in range(0,N-window)]
    nans = np.repeat(np.nan, window).tolist()
    exs = [-Ex_Shortfall(data[w[0]:w[1]]) for w in windows]
    exs = pd.Series(nans + exs, index = data.index)
    return exs
     
   
    
    