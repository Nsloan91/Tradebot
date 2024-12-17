import robin_stocks.robinhood as rs
import pandas as pd
import datetime
import time
import talib

# Login
username = 'nsloan91@outlook.com'
password = 'marine2010'
rs.login(username=username, password=password)

# Setup DataFrame to store data
data = pd.DataFrame(columns=['Timestamp', 'Price', 'MA', 'RSI', 'MACD'])

# Duration of the script run
total_duration = 20 * 60  # 20 minutes in seconds

# Loop to collect data
start_time = time.time()
try:
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = total_duration - elapsed_time

        # Break the loop if time is up
        if remaining_time <= 0:
            break

        # Fetch ETC price
        # ... Fetching and processing logic ...

        # Print time remaining
        print(f"Time remaining: {remaining_time // 60} minutes and {int(remaining_time % 60)} seconds")


        # Fetch ETC price
        price_info = rs.crypto.get_crypto_quote('ETC')
        price = float(price_info['mark_price'])
        timestamp = datetime.datetime.now()

        # Append new data
        new_row = pd.DataFrame({'Timestamp': [timestamp], 'Price': [price]})
        data = pd.concat([data, new_row], ignore_index=True)

        # Calculate indicators with enough data
        if len(data) >= 30:
            data['MA'] = data['Price'].rolling(window=10).mean()
            data['RSI'] = talib.RSI(data['Price'].values, timeperiod=14)
            macd, signal, hist = talib.MACD(data['Price'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            data['MACD'] = macd
 
# Print time remaining
        print(f"Time remaining: {remaining_time // 60} minutes and {int(remaining_time % 60)} seconds")

        time.sleep(60)  # wait for 1 minute
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Export to CSV
    data.to_csv('etc_crypto_data.csv', index=False)
    rs.logout()