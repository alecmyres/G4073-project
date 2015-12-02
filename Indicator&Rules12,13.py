# Stochastic oscillator
def get_stoch(high, low, close, n = 12, per = 3):
    Fast_K = [np.nan]*high.size
    Fast_D = [np.nan]*high.size
    Slow_K = [np.nan]*high.size
    Slow_D = [np.nan]*high.size
    for i in range(n, high.size+1):
        P_high = high[i-n: i]
        P_low = low[i-n: i]
        Fast_K[i-1] = (close[i-1]-min(P_low))/(max(P_high)-min(P_low))
    Fast_K = pd.Series(Fast_K)
    Fast_D = pd.Series(pd.rolling_mean(Fast_K, window = per))
    Slow_K = pd.Series(pd.rolling_mean(Fast_K, window = per))
    Slow_D = pd.Series(pd.rolling_mean(Slow_K, window = per))
    return Fast_K, Fast_D, Slow_K, Slow_D
    
df['Fast_K'], df['Fast_D'], df['Slow_K'], df['Slow_D'] = get_stoch(df['high '], df['low '], df['close'])

# Rule 12
def GetStrategyFast(high, low, close):
    Fast_K, Fast_D, Slow_K, Slow_D = get_stoch(high, low, close)
    action = [0]*close.size
    for i in range(0, close.size):
      if (np.isnan(Fast_K[i]) == False) and (np.isnan(Fast_D[i]) == False):
          if Fast_K[i] > Fast_D[i]:
              action[i] = 1
          elif Fast_K[i] < Fast_D[i]:
              action[i] = -1
    return action

def GetStrategySlow(high, low, close):
    Fast_K, Fast_D, Slow_K, Slow_D = get_stoch(high, low, close)
    action = [0]*close.size
    for i in range(0, close.size):
      if (np.isnan(Slow_K[i]) == False) and (np.isnan(Slow_D[i]) == False):
          if Slow_K[i] > Slow_D[i]:
              action[i] = 1
          elif Slow_K[i] < Slow_D[i]:
              action[i] = -1
    return action

action1 = GetStrategyFast(df['high '], df['low '], df['close'])
action2 = GetStrategyFast(df['high '], df['low '], df['close'])
