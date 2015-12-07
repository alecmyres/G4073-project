# Maintain portfolio, analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University

# Load useful python libraries
import numpy as np
import pandas as pd



# ---------------------------------------
# STOCK/PORTFOLIO CLASSES
# ---------------------------------------

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
        self.curr_amtBought = 0.0
        self.curr_amtSold = 0.0
        self.curr_pnl = 0.0

    # size: number of shares (integer)
    # side: "BUY" or "SELL" (string)
    # price: execution price (float)
    def addTrade(self, size, side, price):
        dir_size = size if (side == "BUY") else -size
        # Reset max mkt value if new position is taken on
        s1 = np.sign(self.position)
        s2 = np.sign(self.position + dir_size)
        if ((self.position == 0) & (size != 0)):
            self.curr_amtBought = (size*price if (side == "BUY") else 0.0)
            self.curr_amtSold = (size*price if (side == "SELL") else 0.0)
            self.max_mkt_val = dir_size*price
        elif ((s1 != s2) & (s1 + s2 == 0)):
            resid = self.position + dir_size
            self.curr_amtBought = (abs(resid)*price if (side == "BUY") else 0.0)
            self.curr_amtSold = (abs(resid)*price if (side == "SELL") else 0.0)
            self.max_mkt_val = resid*price
        else:
            self.curr_amtBought += (size*price if (side == "BUY") else 0.0)
            self.curr_amtSold += (size*price if (side == "SELL") else 0.0)
        self.position += dir_size
        self.amtBought += (size*price if (side == "BUY") else 0.0)
        self.amtSold += (size*price if (side == "SELL") else 0.0)
        self.fees += calcFees(size, side, price)
        self.last_price = price

    # Update pnl for stock with latest price
    def updatePNL(self, last_price):
        self.last_price = last_price
        self.pnl = self.amtSold - self.amtBought + self.fees + self.position*last_price
        self.curr_pnl = self.curr_amtSold - self.curr_amtBought + self.position*last_price
        self.max_mkt_val = max(self.max_mkt_val, self.position*last_price)


# Portfolio class
class Portfolio:
    def __init__(self, name, cash):
        self.name = name
        self.stocks = dict() # stores Stock objects
        self.industries = dict() # stores all industry allocations 
        self.start_cash = cash # starting cash
        self.cash_value = cash # current cash value
        self.port_value = 0.0 # sum of abs value of positions 

    def addTrade(self, ticker, industry, size, side, price):
        # Check if stock exists in portfolio, add if needed
        if ticker not in self.stocks.keys():
            self.stocks[ticker] = Stock(ticker, industry)
        # Add trade, update cash value
        self.stocks[ticker].addTrade(size, side, price)

    def updatePortValue(self):
        # Reset portfolio and sector values
        self.port_value = 0.0
        self.cash_value = self.start_cash
        for i in self.industries.keys():
            self.industries[i] = 0.0
        for ticker in self.stocks.keys():
            pv = (self.stocks[ticker].position)*(self.stocks[ticker].last_price)
            self.port_value += abs(pv)
            industry = self.stocks[ticker].industry
            if industry not in self.industries.keys():
                self.industries[industry] = 0.0
            self.industries[industry] += abs(pv)
            self.cash_value += self.stocks[ticker].pnl



# ---------------------------------------
# RISK MANAGEMENT
# ---------------------------------------

# Risk Parameters
max_stock_percent = 0.025
max_sector_percent = 0.20
max_adv_percent = 0.02
margin_add_percent = 0.50
margin_call_percent = 0.30 
stop_loss_percent = 0.10

# Check trade signal for valid portfolio inclusion
# Set trade size based on risk/portfolio settings
def checkTrade(ticker, industry, price, action):
    global portfolio
    global max_stock_percent
    global max_sector_percent
    global max_adv_percent
    global margin_add_percent
    # Set side text based on action value
    if (action == 0) or (action == 1):
        side = "BUY"
    elif action == -1:
        side = "SELL"
    else:
        print "ERROR w/ action!"
        return
    # Check if ticker is in portfolio, get position
    if ticker not in portfolio.stocks.keys():
        portfolio.addTrade(ticker, industry, 0, "BUY", 0.0)
    position = portfolio.stocks[ticker].position
    port_value = portfolio.port_value
    # Check available portfolio space
    port_alloc = (1.0/margin_add_percent)*portfolio.cash_value - port_value
    av_port_space = max(port_alloc, 0.0)
    # Check available sector space
    ind_alloc = portfolio.industries[industry]/port_value
    av_sector_space = max((max_sector_percent - ind_alloc), 0.0)*port_value
    # Check available stock space
    stock_alloc = portfolio.stocks[ticker].position*price/port_value
    av_stock_space = max((max_stock_percent - stock_alloc), 0.0)*port_value
    available_space = min(av_port_space, av_sector_space, av_stock_space)
    # find average daily volume for the stock
    adv = 1000000 # shares per day
    # Set trade size
    adv_alloc = (adv*price)*max_adv_percent
    if ((position != 0) & (sign(position) + sign(action) == 0)): # reduce position
        available_space = adv_alloc
    else:
        available_space = min(adv_alloc, available_space)
    trade_size = int(round(available_space/price, -2))
    # Set price to be worse than closing price
    slip_rate = 5 # in bps
    adjust = round(max(0.01, price*slip_rate/10000.0), 2)
    price  = price + (adjust if (side == "BUY") else -adjust)
    # Add trade to portfolio
    if action == 0:
        trade_size = 0
    portfolio.addTrade(ticker, industry, trade_size, side, price)
    return

# Check for margin call, sell assets if needed
def checkMarginCall():
    global portfolio
    global margin_call_percent
    excess_margin = (1.0/margin_call_percent)*portfolio.cash_value - portfolio.port_value
    if excess_margin < 0:
        # Sell assets to get excess margin positive again 
        liq_amt = abs(excess_margin)
        # TO DO
    return
    
# Check for stop loss
def checkStopLoss():
    global portfolio
    global stop_loss_percent
    for ticker in portfolio.stocks.keys():
        pos = portfolio.stocks[ticker].position
        curr_pnl = portfolio.stocks[ticker].curr_pnl
        max_val = portfolio.stocks[ticker].max_mkt_val
        loss = curr_pnl/abs(max_val) if (pos != 0) else 0
        if loss < stop_loss_percent:
            close_dir = "BUY" if (pos < 0) else "SELL"
            industry = portfolio.stocks[ticker].industry
            close_px = portfolio.stocks[ticker].last_price
            portfolio.addTrade(ticker, industry, abs(pos), close_dir, close_px)

# ---------------------------------------
# RUN PORTFOLIO, TRADES
# ---------------------------------------       

# Portfolio start
start_cash = 1000000.0
portfolio = Portfolio("main", start_cash)

# Load LIBOR rates
libor = pd.read_csv('libor.csv')
def getLiborRate(year, month):
    global libor
    lquery = libor.query('year == @year & month == @month')['VALUE']
    if len(lquery.index) == 1:
        return lquery.item()/100.0
    else:
        print "ERROR w/ Libor"
        return np.nan


# Day by day portfolio
# buy/sell
# pnl updates
# portfolio updates
# margin call
# stop loss


# TO DO:
# Historic fee corrections
# Interest (short/long/cash)?
# Handle many same signal stretches
# remove industry passing in arguments
