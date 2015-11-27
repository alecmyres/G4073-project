# Analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University
                                                                   
# Load useful python libraries
import numpy as np
import pandas as pd
import datetime, dateutil, os, sys


# Find pnl
# argument: dataframe with columns "Date", "Price", "Side", and "Size"
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
            pos = pos + df['Size'][i]
        elif df['Side'][i] == "SELL":
            amtSold = amtSold + df['Price'][i]*df['Size'][i]
            pos = pos - df['Size'][i]
        pnl.append(amtSold - amtBot + pos*df['Price'][i] + fees)
    # return pnl column
    return pnl


# Max drawdown calculation
# Dataframe with "pnl" column
def maxDrawdown(df):
    return pd.expanding_max(df['pnl']) - df['pnl']
