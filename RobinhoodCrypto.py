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

# Login
rs.login(username=username, password=password)

# Start time
start_time = time.time()

try:
    while True:
        # Check if 5 minutes have passed
        if time.time() - start_time > 300:
            break

        # Get current crypto price
        crypto_data = rs.crypto.get_crypto_quote('BTC')
        price = crypto_data['mark_price']
        timestamp = datetime.datetime.now()

        # Print or store the data
        print(f"{timestamp}: BTC price is {price}")

        # Wait for 1 minute
        time.sleep(60)
except KeyboardInterrupt:
    print("Script stopped.")
finally:
    rs.logout()