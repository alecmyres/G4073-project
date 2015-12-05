# Maintain portfolio, analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University
                                                                   
# Load useful python libraries
import numpy as np
import pandas as pd

# Cash and portfolio values
cash_value = 1000000.0

# Fee includes exchange, broker, and SEC components
def calcFees(size, side, price):
    fees  = (size*-0.002) +\
            (size*price*-0.0002) +\
            (size*price*(-20.0/1000000.0) if side == "SELL" else 0.0)
    return fees


# Stock class
class Stock:
    # initialize variables
    def __init__(self, ticker, industry):
        self.ticker = ticker
        self.industry = industry
        self.position = 0
        self.amtBought = 0.0
        self.amtSold = 0.0
        self.fees = 0.0
        self.pnl = 0.0
        self.last_price = 0.0
        self.max_mkt_val = 0.0

    # size: number of shares (integer)
    # side: "BUY" or "SELL" (string)
    # price: execution price (float)
    def addTrade(self, size, side, price):
        dir_size = size if (side == "BUY") else -size
        # Zero out max mkt value if new position is taken on
        if ((self.size == 0) & (size != 0)) or (np.sign(self.size) != np.sign(self.size + dir_size)):
            self.max_mkt_val = 0.0
        self.position += dir_size
        self.amtBought += (size*price if (side == "BUY")  else 0.0)
        self.amtSold += (size*price if (side == "SELL")  else 0.0)
        self.fees += calcFees(size, side, price)

    # Update pnl for stock with latest price
    def updatePNL(self, last_price):
        self.last_price = last_price
        self.pnl = self.amtSold - self.amtBought + self.fees + self.position*last_price
        self.max_mkt_val = max(self.max_mkt_val, self.position*last_price)


# Portfolio class
class Portfolio:
    def __init__(self, name, cash):
        self.name = name
        self.stocks = dict()
        self.industries = dict()
        self.cash_value = cash
        self.port_value = 0.0

    def addTrade(self, ticker, industry, size, side, price):
        # Check if stock exists in portfolio, add if needed
        if ticker not in self.stocks.keys():
            self.stocks[ticker] = Stock(ticker, industry)
        # Add trade, update cash value
        self.stocks[ticker].addTrade(size, side, price)
        self.cash_value += size*price*(1 if (side == "SELL") else -1) +\
                           calcFees(size, side, price)

    def updatePortValue(self):
        # Reset portfolio and sector values
        self.port_value = 0.0
        for i in self.industries.keys():
            self.industries[i] = 0.0
        for ticker in self.stocks.keys():
            pv = (self.stocks[ticker].position)*(self.stocks[ticker].last_price)
            self.port_value += pv
            industry = self.stocks[ticker].industry
            if industry not in self.industries.keys():
                self.industries[industry] = 0
            self.industries[industry] += abs(pv)

# Risk Parameters
max_stock_percent = 0.025
max_sector_percent = 0.20
max_adv_percent = 0.05

# ------------
# TEST
# ------------

test = pd.DataFrame()
size = 50
test['date'] = range(size)
test['price'] = 25 + np.cumsum(np.random.normal(size = size))
test['action'] = np.random.choice([-1,0,1], size = size, p = [.2, .6, .2])

portfolio = Portfolio("tester", 100000.0)
for i in test.index:
    size = 100 if (test['action'][i] != 0) else 0
    side = "BUY" if (test['action'][i] == 1) else "SELL"
    portfolio.addTrade("ABC", "tech", size, side, test['price'][i])

# ------------
# END TEST
# ------------



# Check trade signal for valid portfolio inclusion
# Set trade size based on risk/portfolio settings
def checkTrade(stock, side, price):
    global portfolio
    # Check available portfolio space

    
    # Set trade size
    # find average daily volume for the stock
    
    # Set price to be worse than closing price
    slip_rate = 5    # bps
    adjust = round(max(0.01, price*slip_rate/10000.0), 2)
    price  = price + (adjust if (side == "BUY") else -adjust)
    
    # Add trade to portfolio

    return





# Portfolio allocation
# IDEAS:
# Limits for single stock positions (% of portfolio)
# Limits for sector positions (% range of portfolio)
# Limits for single day buy/sell amounts
# Stop loss logic
# Manage/minimize cash component of portfolio 

# TO DO:
# Historic fee corrections
# Interest (short/long/cash)?
# 
