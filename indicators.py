
# Technical trading strategies
# Alec Myres and Yeyun Chen
# MATH G4073 - Columbia University

# Load useful python libraries
import numpy as np
import pandas as pd
import datetime, dateutil, os, sys
import boosting

# Load csv as a DataFrame object
filepath = '/Users/alecmyres/Documents/G4073_Qnt_Mthds/data_example_MU.csv'
df = pd.read_csv(filepath)
df = df[['date','split adjusted px']]
df.rename(columns = {'split adjusted px':'Price'}, inplace = True)
df['Date'] = map(lambda x: dateutil.parser.parse(x).strftime('%Y-%m-%d'), df['date'])
df = df[['Date','Price']].sort('Date').reset_index(drop = True)

# min/max cumulative sum for an array
def min_max_cum_sum(array):
  cs = np.cumsum(array)
  return np.min(cs), np.max(cs)

# Add y columns
rw = 5 # rolling window
threshhold = 0.07
df['logreturn'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
# df['rollsum'] = pd.rolling_sum(df['logreturn'], rolling_window)
df['min_rollsum'] = map(lambda x: min_max_cum_sum(df['logreturn'][x:x+rw])[0], df.index)
df['max_rollsum'] = map(lambda x: min_max_cum_sum(df['logreturn'][x:x+rw])[1], df.index)
df['y_buy'] = np.where(df['max_rollsum'] > threshhold, 1, 0)
df['y_sell'] = np.where(df['min_rollsum'] < -threshhold, -1, 0)
# df['y_buy'] = df['y_buy'].shift(-5)
# df['y_sell'] = df['y_sell'].shift(-5)
# del df['rollsum']

# --------------------
# INDICATORS
# --------------------

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
  upper = get_sma(df_col, w) + bandwidth*s
  lower = get_sma(df_col, w) - bandwidth*s
  return upper, lower

# Momentum (differences)
def get_mom(df_col, per = 12):
  roll = (np.roll(df_col, 0) - np.roll(df_col, per))[per:]
  fill = [np.nan]*per
  return pd.Series(fill + list(roll))

# Momentum EMA
def get_mom_ema(df_col, per = 12, alpha = 0.75):
  mom = get_mom(df_col, per)
  ema_mom = get_ema(mom, alpha)
  return mom/ema_mom

# Acceleration (difference of price change)
def get_accel(df_col, per = 10):
    mom_t = get_mom(df_col, per)[per:]
    roll = (np.roll(mom_t, 0) - np.roll(mom_t, 1))[1:]
    fill = [np.nan]*(per + 1)
    return pd.Series(fill + list(roll))

# rate of change
def get_roc(df_col, per = 10):
  roll = (get_mom(df_col, per)/np.roll(df_col, per))[per:]
  fill = [np.nan]*per
  return pd.Series(fill + list(roll))

# moving average convergence divergence
# difference between two moving averages of slow(s) and fast(f) periods
def get_macd(df_col, s = 26, f = 12):
    return(pd.ewma(df_col, span = s) - pd.ewma(df_col, span = f))

# MACD signal line: moving average of MACD of past n periods
def get_macds(df_col, s = 26, f = 12, n = 9):
    return(pd.ewma(get_macd(df_col, s, f), span = n))

# Relative strength index
def get_rsi(df_col, per = 14):
    p_up = [np.nan]*df_col.size
    p_dn = [np.nan]*df_col.size
    for i in range(1, df_col.size):
        if (df_col[i] > df_col[i-1]):
            p_up[i] = df_col[i]
        elif (df_col[i] < df_col[i-1]):
            p_dn[i] = df_col[i]

    ma_up = [np.nan]*df_col.size
    ma_dn = [np.nan]*df_col.size
    rsi   = [np.nan]*df_col.size
    for j in range(per, df_col.size):
        ma_up[j] = np.nanmean(p_up[j-per:j])
        ma_dn[j] = np.nanmean(p_dn[j-per:j])
        rsi[j] = 100 - 100/(1 + ma_up[j]/ma_dn[j])
    return pd.Series(rsi)

# The above functions take an array as an argument, with other optional parameters
# Each function returns an array, which can be used in other functions

# Add columns to DataFrame
df['sma20']  = get_sma(df['Price'])
df['ema']    = get_ema(df['Price'])
df['mom12']  = get_mom(df['Price'])
df['momEMA12'] = get_mom_ema(df['Price'])
df['bollUpper'], df['bollLower'] = get_bollinger(df['Price'])
df['accel'] = get_accel(df['Price'])
df['roc'] = get_roc(df['Price'])
df['macd'] = get_macd(df['Price'])
df['macds'] = get_macds(df['Price'])
df['rsi'] = get_rsi(df['Price'])



# --------------------
# SIGNALS
# --------------------

# Derive buy/sell signals from technical indicators
#  1: buy
# -1: sell
#  0: no action

# Strategy 1: Bollinger Bands
def GetStrategyBollinger(prices, n = 20, s = 2):
    action = [0]*prices.size
    bollUpper, bollLower = get_bollinger(prices, n, s)
    for i in range(1, prices.size):
        if (prices[i] > bollUpper[i]):
            action[i] = -1
        elif (prices[i] < bollLower[i]):
            action[i] = 1     
    return action 

# Strategy 2: Momentum
def GetStrategyMomentum(prices, n = 12):
    action = [0]*prices.size
    mom  = get_mom(prices, n)
    momEMA = get_mom_ema(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(mom[i-1]) and ~np.isnan(momEMA[i-1]):
            if (mom[i] > momEMA[i]):
                action[i] = 1
            elif (mom[i] < momEMA[i]):
                action[i] = -1
    return action         

# Strategy 3: Rate of Change
def GetStrategyROC(prices, n = 10):
    action = [0]*prices.size
    ROC  = get_roc(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(ROC[i-1]):
            if (ROC[i-1] <= 0) and (ROC[i] > 0):
                action[i] = 1
            elif (ROC[i-1] >= 0) and (ROC[i] < 0):
                action[i] = -1
    return action         

# Strategy 4: Acceleration
def GetStrategyAcceleration(prices, n = 12):
    action = [0]*prices.size
    accel = get_accel(prices, n)
    for i in range(1, prices.size):
        if ~np.isnan(accel[i-1]):
            if (accel[i-1] + 1 <= 0) and (accel[i] > 0):
                action[i] = 1
            elif (accel[i-1] + 1 >= 0) and (accel[i] < 0):
                action[i] = -1
    return action

# Strategy 5: Moving Average Convergence Difference
def GetStrategyMACD(prices, s = 26, f = 12, n = 9):
    macd = get_macd(prices, s, f)
    macds = get_macds(prices, s, f, n)
    action = [0]*prices.size
    for i in range(1, prices.size):
      if (macd[i-1] <= macds[i]) and (macd[i] > macds[i]):
        action[i] = 1
      elif (macd[i-1] >= macds[i]) and (macd[i] < macds[i]):
        action[i] = -1
    return action

# Strategy 6: Relative Strength Index
def GetStrategyRSI(prices, per = 14):
    rsi = get_rsi(prices, 14)
    action = [0]*prices.size
    for i in range(per, prices.size):
      if (rsi[i-1] >= 30) and (rsi[i] < 30):
        action[i] = 1
      elif (rsi[i-1] <= 70) and (rsi[i] > 70):
        action[i] = -1
    return action



# Add indicator columns to data frame
df['action1'] = GetStrategyBollinger(df['Price'])
df['action2'] = GetStrategyMomentum(df['Price'])
df['action3'] = GetStrategyROC(df['Price'])
df['action4'] = GetStrategyAcceleration(df['Price'])
df['action5'] = GetStrategyMACD(df['Price'])
df['action6'] = GetStrategyRSI(df['Price'])

# Split into buy/sell components
df['action1buy'] = np.where(df['action1'] == -1, 0, df['action1'])
df['action1sell'] = np.where(df['action1'] == 1, 0, df['action1'])
df['action2buy'] = np.where(df['action2'] == -1, 0, df['action2'])
df['action2sell'] = np.where(df['action2'] == 1, 0, df['action2'])
df['action3buy'] = np.where(df['action3'] == -1, 0, df['action3'])
df['action3sell'] = np.where(df['action3'] == 1, 0, df['action3'])
df['action4buy'] = np.where(df['action4'] == -1, 0, df['action4'])
df['action4sell'] = np.where(df['action4'] == 1, 0, df['action4'])
df['action5buy'] = np.where(df['action5'] == -1, 0, df['action5'])
df['action5sell'] = np.where(df['action5'] == 1, 0, df['action5'])
df['action6buy'] = np.where(df['action6'] == -1, 0, df['action6'])
df['action6sell'] = np.where(df['action6'] == 1, 0, df['action6'])

del df['action1']
del df['action2']
del df['action3']
del df['action4']
del df['action5']
del df['action6']


alphas, strategies = boosting.update_weights(df[(df.index > 4800) & (df.index < 5000)], "buy")

df_copy = df
for i in range(len(alphas)):
  df_copy[strategies[i] + 'wgt'] = np.where(df[strategies[i]] == 0, -1, 1)*alphas[i]

columns = map(lambda x: x+"wgt", strategies)
df_copy['final'] = df_copy[columns].sum(axis = 1)
df_copy[(df.index >= 5000) & (df.index < 5100)]
