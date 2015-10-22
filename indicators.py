# Some basic functions for trading strategies
# Alec Myres
# MATH G4073 - Columbia University

# Load useful python libraries
import numpy as np
import pandas as pd

# Load csv as a DataFrame object
df = pd.read_csv('/Users/alecmyres/Documents/G4073_Qnt_Mthds/data.csv')

# Simple moving average
def get_sma(df_col, w = 20):
  return pd.rolling_mean(df_col, window = w)

# Exponentially weighted moving average
def get_ema(df_col, alpha = 0.95):
  s = (1.0 + alpha)/(1.0 - alpha)
  return pd.ewma(df_col, span = s)

# Bollinger bands
def get_bollinger(df_col, w = 20, s = 2):
  bandwidth = pd.rolling_std(df_col, window = w)
  upper = get_sma(df_col, w)+bandwidth*s
  lower = get_sma(df_col, w)-bandwidth*s
  return upper, lower

# Momentum (differences)
def get_mom(df_col, per = 10):
  roll = (np.roll(df_col, 0)-np.roll(df_col, per))[per:]
  fill = [np.nan]*per
  return pd.Series(fill+list(roll))

# Momentum EMA
def get_mom_ema(df_col, per = 10, alpha = 0.95):
  mom = get_mom(df_col, per)
  ema_mom = get_ema(mom, alpha)
  return mom/ema_mom

# The above functions take an array as an argument, with other optional parameters
# Each function returns an array, which can be used in other functions

# Add columns to DataFrame
df['sma20']  = get_sma(df['Price'])
df['ema']    = get_ema(df['Price'])
df['mom10']  = get_mom(df['Price'])
df['momEMA'] = get_mom_ema(df['Price'])
df['bollUpper'], df['bollLower'] = get_bollinger(df['Price'])
