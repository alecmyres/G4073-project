# Maintain portfolio, analyze trading (pnl, max drawdown, etc.)
# Alec Myres
# MATH G4073 - Columbia University
                                                                   
# Load useful python libraries
import numpy as np
import pandas as pd


# Portfolio dictionary
# stock:[Position, Amount Bought, Amount Sold, fees, pnl, max position value]
portfolio = dict()

# Add trade and update portfolio
# side: "BUY" or "SELL"
# size: number of shares
# price: share price
def updatePortfolio(stock, side, size, price):
    global portfolio
    position_update = size if (side == "BUY") else -size
    buy_update  = size*price if (side == "BUY")  else 0.0
    sell_update = side*price if (side == "SELL") else 0.0
    # Fee includes exchange, broker, and SEC components
    # TO DO: historical corrections
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

# Update pnl for a stock
def updatePNL(stock, closing_price):
    global portfolio
    if stock in portfolio.keys():
        info = portfolio[stock]
        # PNL = cash in - cash out + fees + market value of position
        portfolio[stock][4] = info[2] - info[1] + info[3] + info[0]*closing_price
    else:
        print "Error:", stock, "not in portfolio"


