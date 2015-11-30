# Analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University
                                                                   
# Load useful python libraries
import numpy as np
import pandas as pd
import datetime, dateutil, os, sys


# Find pnl
# argument: dataframe with columns "Price", "Side", and "Size"
def netPNL(df):
    amtBot  = 0
    amtSold = 0
    pos  = 0
    fees = 0
    pnl  = []
    # Find pnl day by day
    for i in df.index:
        if df['Side'][i] == "BUY":
            amtBot = amtBot + df['Price'][i]*df['Size'][i]
            pos += df['Size'][i]
            fees += df['Size'][i]*(-0.002) # Exchange fees
        elif df['Side'][i] == "SELL":
            amtSold = amtSold + df['Price'][i]*df['Size'][i]
            pos = pos - df['Size'][i]
            fees += df['Size'][i]*(-0.002) # Exchange fees
            fees += df['Size'][i]*df['Price'][i]*(-20.0/1000000.0) # SEC fees
        pnl.append(amtSold - amtBot + pos*df['Price'][i] + fees)
    # return pnl column
    return pnl


# Max drawdown calculation
# argument: dataframe with "pnl" column
def maxDrawdown(df):
    return pd.expanding_max(df['pnl']) - df['pnl']

