# Maintain portfolio, analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University
                                                                   
# Load useful python libraries
import numpy as np
import pandas as pd

# Cash and portfolio values
cash_value = 1000000.0
port_value = 0.0

# Portfolio dictionary
# stock:[Position, Amount Bought, Amount Sold, fees, pnl, max position value]
portfolio = dict()

# Sector lookup
sectors = dict()
# ADD SECTOR INFO

# Parameters
max_stock_percent = 0.05


# Add valid trade and update portfolio
# side: "BUY" or "SELL"
# size: number of shares
# price: share price
def updatePortfolio(stock, side, size, price):
    global portfolio
    position_update = size if (side == "BUY") else -size
    buy_update  = size*price if (side == "BUY")  else 0.0
    sell_update = side*price if (side == "SELL") else 0.0
    # Fee includes exchange, broker, and SEC components
    fee_update  = (size*-0.002) +\
                  (size*price*-0.0002) +\
                  (size*price*(-20.0/1000000.0) if side == "SELL" else 0.0)
    if stock in portfolio.keys():
        # Update portfolio
        portfolio[stock][0] += position_update
        portfolio[stock][1] += buy_update
        portfolio[stock][2] += sell_update
        portfolio[stock][3] += fee_update
    else:
        portfolio[stock] = [position_update, buy_update, sell_update, fee_update, 0.0, 0.0]
    return

# Update pnl for a stock
def updatePNL(stock, closing_price):
    global portfolio
    if stock in portfolio.keys():
        info = portfolio[stock]
        # PNL = cash in - cash out + fees + market value of position
        portfolio[stock][4] = info[2] - info[1] + info[3] + info[0]*closing_price
    else:
        print "Error:", stock, "not in portfolio"
    return

# Check trade signal for valid portfolio inclusion
# Set trade size based on risk/portfolio settings
def checkTrade(stock, side, price):
    global portfolio
    # Check available portfolio space
    
    # Set trade size

    # Set price to be worse than closing price
    slip_rate = 5    # bps
    adjust = round(max(0.01, price*slip_rate/10000.0), 2)
    price  = price + (adjust if (side == "BUY") else -adjust)
    
    # Add trade to portfolio
    updatePortfolio(stock, side, size, price)
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
# Interest?
