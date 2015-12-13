# Maintain portfolio, analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University

# Load useful python libraries
import numpy as np
import pandas as pd
import datetime, dateutil, os, sys


# ---------------------------------------
# STOCK/PORTFOLIO CLASSES
# ---------------------------------------

# Fee includes exchange, broker, and SEC components
def calcFees(size, side, price):
    fees  = (size*-0.001) +\
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
        self.curr_mkt_entry = 0.0
        self.last_trade_dt = '1900-01-01'

    # size: number of shares (integer)
    # side: "BUY" or "SELL" (string)
    # price: execution price (float)
    def addTrade(self, size, side, price):
        dir_size = size if (side == "BUY") else -size
        # Reset max mkt value if new position is taken on
        if self.position == 0:
            self.curr_amtBought = 0.0
            self.curr_amtSold = 0.0
        if self.position == 0 and size != 0:
            self.max_mkt_val = size*price
        self.position += dir_size
        self.curr_amtBought += (size*price if (side == "BUY") else 0.0)
        self.curr_amtSold += (size*price if (side == "SELL") else 0.0)
        self.curr_mkt_entry = abs(self.curr_amtSold - self.curr_amtBought)
        self.amtBought += (size*price if (side == "BUY") else 0.0)
        self.amtSold += (size*price if (side == "SELL") else 0.0)
        addfees = calcFees(size, side, price)
        if addfees > 0:
            print "FEES", self.ticker, size, side, price
        else:
            self.fees += calcFees(size, side, price)
        self.last_price = price

    # Update pnl for stock with latest price
    def updatePNL(self, last_price):
        self.last_price = last_price
        self.pnl = self.amtSold - self.amtBought + self.fees + self.position*last_price
        self.curr_pnl = self.curr_amtSold - self.curr_amtBought + self.position*last_price
        if self.position == 0:
            self.max_mkt_val = 0.0
        self.max_mkt_val = max(self.max_mkt_val, self.curr_mkt_entry + self.curr_pnl)


# Portfolio class
class Portfolio:
    def __init__(self, name, cash):
        self.name = name
        self.stocks = dict() # stores Stock objects
        self.industries = dict() # stores all industry allocations 
        self.start_cash = cash # starting cash
        self.cash_value = cash # current cash value
        self.port_value = 0.0 # sum of abs value of positions 
        self.fees = 0.0 

    def addTrade(self, ticker, industry, size, side, price, date):
        # Check if stock exists in portfolio, add if needed
        if ticker not in self.stocks.keys():
            self.stocks[ticker] = Stock(ticker, industry)
        # Add trade, update cash value
        self.stocks[ticker].addTrade(size, side, price)
        if size != 0:
            self.stocks[ticker].last_trade_dt = date

    def updatePortValue(self):
        # Reset portfolio and sector values
        self.port_value = 0.0
        self.cash_value = self.start_cash
        self.fees = 0.0
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
            self.fees += self.stocks[ticker].fees


# ---------------------------------------
# RISK MANAGEMENT
# ---------------------------------------

# Check trade signal for valid portfolio inclusion
# Set trade size based on risk/portfolio settings
def checkTrade(ticker, industry, price, action, date, PRC, VOL):
    global portfolio
    global max_stock_percent
    global max_sector_percent
    global max_adv_percent
    global margin_add_percent
    global slip_rate_bps
    global days_between_trades
    # Add to portfolio if needed
    if ticker not in portfolio.stocks.keys():
        portfolio.addTrade(ticker, industry, 0, "BUY", 0.0, date)
    # Set action to 0 if last trade too recent
    last_trade = dateutil.parser.parse(portfolio.stocks[ticker].last_trade_dt)
    if (dateutil.parser.parse(date) - last_trade).days < days_between_trades:
        action = 0
    # Set side text based on action value
    if (action == 0) or (action == 1):
        side = "BUY"
    elif action == -1:
        side = "SELL"
    else:
        print "ERROR w/ action!"
        return
    position = portfolio.stocks[ticker].position
    port_value = portfolio.port_value
    start_cash = portfolio.start_cash
    # Check available portfolio space
    port_alloc = (1.0/margin_add_percent)*portfolio.cash_value - port_value
    av_port_space = max(port_alloc, 0.0)
    # Check available sector space
    if industry not in portfolio.industries.keys():
        portfolio.industries[industry] = 0.0
    ind_alloc = portfolio.industries[industry]/max(port_value, 0.25*start_cash)
    av_sector_space = max((max_sector_percent - ind_alloc), 0.0)*max(port_value, 0.25*start_cash)
    # Check available stock space
    stock_alloc = portfolio.stocks[ticker].position*price/max(port_value, 0.25*start_cash)
    av_stock_space = max((max_stock_percent - stock_alloc), 0.0)*max(port_value, 0.25*start_cash)
    available_space = min(av_port_space, av_sector_space, av_stock_space, 0.1*VOL*PRC, 20000.0)
    # find average daily volume for the stock
    #adv = 1000000 # shares per day
    # Set trade size
    #adv_alloc = (adv*price)*max_adv_percent
    #if ((position != 0) & (np.sign(position) + np.sign(action) == 0)): # reduce position
    #    available_space = adv_alloc
    #else:
    #    available_space = min(adv_alloc, available_space)
    #trade_size = int(round(available_space/price, -2))
    # Keep positions within limits
    trade_size = 100
    if (position >= 500) and (action == 1):
        trade_size = 0
    elif (position <= -500) and (action == -1):
        trade_size = 0
    else:
        trade_size = max(100, round(available_space/price, -2))
    # Don't flip possitions from pos/neg, just zero out
    if ((side == "BUY") & (position < 0) & (position + trade_size > 0)):
        trade_size = abs(position)
    if ((side == "SELL") & (position > 0) & (position - trade_size < 0)):
        trade_size = abs(position)
    # Set price to be worse than closing price
    adjust = round(max(0.01, price*slip_rate_bps/10000.0), 2)
    price  = price + (adjust if (side == "BUY") else -adjust)
    # Add trade to portfolio
    if (PRC < 6) or (PRC > 500):
        trade_size = 0
    if action == 0:
        trade_size = 0
    portfolio.addTrade(ticker, industry, trade_size, side, price, date)
    return

# Check for margin call, sell assets if needed
def checkMarginCall():
    global portfolio
    global margin_call_percent
    excess_margin = (1.0/margin_call_percent)*portfolio.cash_value - portfolio.port_value
    if excess_margin < 0:
        # Sell assets to get excess margin positive again 
        liq_amt = abs(excess_margin)
        print "MARGIN CALL SELL"
        # Randomly sell some assets
        while liq_amt > 0:
            ticker = np.random.choice(portfolio.stocks.keys())
            industry = portfolio.stocks[ticker].industry
            position = portfolio.stocks[ticker].position
            date = portfolio.stocks[ticker].last_trade_dt
            if position >= 0:
                side = "SELL"
                size = position
            else:
                side = "BUY"
                size = abs(position)
            price = portfolio.stocks[ticker].last_price
            portfolio.addTrade(ticker, industry, size, side, price, date)
            liq_amt -= size*price
    return
    
# Check for stop loss
def checkStopLoss():
    global portfolio
    global stop_loss_percent
    for ticker in portfolio.stocks.keys():
        pos = portfolio.stocks[ticker].position
        curr_pnl = portfolio.stocks[ticker].curr_pnl
        entry_val = portfolio.stocks[ticker].curr_mkt_entry
        max_val = portfolio.stocks[ticker].max_mkt_val
        loss = (max_val - (entry_val + curr_pnl))/max_val if (pos != 0) else 0
        if loss > stop_loss_percent:
            close_dir = "BUY" if (pos < 0) else "SELL"
            industry = portfolio.stocks[ticker].industry
            close_px = portfolio.stocks[ticker].last_price
            date = portfolio.stocks[ticker].last_trade_dt
            portfolio.addTrade(ticker, industry, abs(pos), close_dir, close_px, date)

# Load LIBOR rates
#libor = pd.read_csv('libor.csv')
#def getLiborRate(year, month):
#    global libor
#    lquery = libor.query('year == @year & month == @month')['VALUE']
#    if len(lquery.index) == 1:
#        return lquery.item()/100.0
#    else:
#        print "ERROR w/ Libor"
#        return np.nan



# ---------------------------------------
# PARAMETERS
# ---------------------------------------

# Risk Parameters 
max_stock_percent = 0.02
max_sector_percent = 0.15 # Set to reasonable level if industry lookup works okay
max_adv_percent = 0.02
margin_add_percent = 0.50
margin_call_percent = 0.30
stop_loss_percent = 0.10
slip_rate_bps = 5
days_between_trades = 5

# Filepath to directory to input files (nothing else)
# files as specified in email
files_dir = "/Users/alecmyres/Documents/G4073_Qnt_Mthds/data/"


# Portfolio
start_cash = 100000000.0




# ---------------------------------------
# RUN PORTFOLIO, TRADES
# ---------------------------------------       

naics_codes = pd.read_csv('/Users/alecmyres/Documents/G4073_Qnt_Mthds/NAICS_codes.csv')
industry_lookup = pd.read_csv('/Users/alecmyres/Documents/G4073_Qnt_Mthds/stock_industry_lookup.csv')

# lookup industry
def industryLookup(ticker):
    global naics_codes
    global industry_lookup
    # lookup
    ticker = int(ticker)
    try:
        hnaic = industry_lookup.query('PERMNO == @ticker')['HNAICS'].item()
    except ValueError:
        hnaic = '00000'
    industry = str(hnaic)[0:2]
    return industry


# Work through file for a year
def runYearFile(file):
    global files_dir
    global portfolio
    global dates
    global pnl
    global fees
    global port_value
    last_date = '1900-01-01'
    # Load file, fix dates, randomize trade order, sort by date
    # This will give no preference to any stock in order entry/limits
    df = pd.read_csv(files_dir + file)
    df['Date'] = map(lambda x: dateutil.parser.parse(x).strftime('%Y-%m-%d'), df['Date'])
    df = df.reindex(np.random.permutation(df.index)).sort('Date').reset_index(drop = True)
    df['year'] = map(lambda x: int(x[0:4]), df['Date'])
    year = max(set(df['year']))
    df = df.query('year == @year')
    all_syms = set(df.Stock.unique())
    stock_list = all_syms - set(df.query('PRC < 0.0 or adjustedPRC < 0').Stock.unique())
    df = df.query('Stock in @stock_list').reset_index(drop = True)
    # Work through signals day by day
    for i in df.index:
        ticker = df['Stock'][i]
        date = df['Date'][i]
        price = df['adjustedPRC'][i]
        signal = df['Signal'][i]
        PRC = df['PRC'][i]
        VOL = df['VOL'][i]
        industry = industryLookup(ticker)
        # check trade and update stock pnl
        checkTrade(ticker, industry, price, signal, date, PRC, VOL)
        portfolio.stocks[ticker].updatePNL(price)
        # resettle calculate/settle portfolio each day
        if date != last_date:
            print date
            portfolio.updatePortValue()
            checkStopLoss()
            #checkMarginCall()
            dates.append(date)
            cash_value.append(portfolio.cash_value)
            port_value.append(portfolio.port_value)
            fees.append(portfolio.fees)
            indu_count.append(len(portfolio.industries.keys()))
            positions = 0
            for ticker in portfolio.stocks.keys():
                if portfolio.stocks[ticker].position != 0:
                    positions += 1
            pos_count.append(positions)
        # update last date
        last_date=date
    # Final portfolio settle
    portfolio.updatePortValue()
    #checkStopLoss()
    #checkMarginCall()
    return


# Load data, convert date, sort by date
files = sorted(os.listdir(files_dir))
# Initialize portfolio
portfolio = Portfolio("main", start_cash)
last_date = '1900-01-01'
dates = []
fees = []
cash_value = []
port_value = []
indu_count = []
pos_count  = []

# Work through file for each year
for file in files:
    print file
    runYearFile(file)
    stats = pd.DataFrame()
    stats['Date'] = dates
    stats['cash_value'] = cash_value
    stats['port_value'] = port_value
    stats['fees'] = fees
    stats['indu'] = indu_count
    stats['postions'] = pos_count
    stats.to_csv(files_dir + file.split('.')[0] + "portfolio_stats.csv")




# TO DO:
# - Industry Lookup
# - Set file path
