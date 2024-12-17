import robin_stocks.robinhood as rs
import os
import datetime
import time
import csv
import pandas as pd
import talib

# Replace these with your Robinhood credentials
username = 'nsloan91@outlook.com'
password = 'marine2010'

# Logging in to Robinhood
rs.login(username=username, password=password)

# Fetching stock data for a specific ticker, e.g., Apple (AAPL)
stock_data = rs.stocks.get_stock_quote_by_symbol('AAPL')

# Displaying stock data
print(stock_data)

# Logging out after operations
rs.logout()