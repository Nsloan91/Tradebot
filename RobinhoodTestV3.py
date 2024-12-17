import robin_stocks.robinhood as rs
import pandas as pd
import datetime
import time
import talib
from scipy.stats import linregress
import os
import logging
import numpy as np 

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

def fetch_latest_price():
    price_info = rs.crypto.get_crypto_quote('ETC')
    price = float(price_info['mark_price'])
    timestamp = datetime.datetime.now()
    return timestamp, price

def fetch_historical_price(symbol, interval, span):
    """
    Fetch historical price data for a given symbol.

    Args:
        symbol (str): The symbol or trading pair (e.g., 'ETC' for Ethereum Classic).
        interval (str): The interval for each data point ('5minute', '10minute', 'hour', 'day', etc.).
        span (str): The time span for the data ('day', 'week', 'month', '3month', 'year', '5year').

    Returns:
        pd.DataFrame: A DataFrame containing historical price data.
    """
    # Use the robin_stocks library to fetch historical price data
    historical_data = rs.crypto.get_crypto_historicals(symbol, interval=interval, span=span)

    # Convert the data to a DataFrame
    historical_df = pd.DataFrame(historical_data)

    # Rename columns for clarity
    historical_df.rename(columns={
        'begins_at': 'Timestamp',
        'close_price': 'Price',
        'high_price': 'High',
        'low_price': 'Low',
        'open_price': 'Open',
        'volume': 'Volume',
    }, inplace=True)

    # Convert Timestamp to datetime format
    historical_df['Timestamp'] = pd.to_datetime(historical_df['Timestamp'])

    return historical_df

def calculate_moving_averages(data, window_short, window_long):
    """
    Calculate Exponential Moving Averages (EMA) for short and long periods.
    
    Args:
    data (pd.DataFrame): DataFrame with 'Close' column.
    window_short (int): Short EMA period.
    window_long (int): Long EMA period.
    
    Returns:
    pd.DataFrame: DataFrame with 'EMA_Short' and 'EMA_Long' columns.
    """
    data['EMA_Short'] = data['Close'].ewm(span=window_short, adjust=False).mean()
    data['EMA_Long'] = data['Close'].ewm(span=window_long, adjust=False).mean()
    return data

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
    data (pd.DataFrame): DataFrame with 'Close' column.
    window (int): RSI calculation period.
    
    Returns:
    pd.DataFrame: DataFrame with 'RSI' column.
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
    data (pd.DataFrame): DataFrame with 'Close' column.
    fast_period (int): Fast EMA period.
    slow_period (int): Slow EMA period.
    signal_period (int): Signal EMA period.
    
    Returns:
    pd.DataFrame: DataFrame with 'MACD', 'MACD_Signal', and 'MACD_Hist' columns.
    """
    data['EMA_Fast'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
    data['EMA_Slow'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
    data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    return data

def calculate_indicators(data, historical_df): ##Use the function to get historical data
# Example usage
historical_data = pd.read_csv('your_historical_data.csv')  # Load historical data

# Calculate moving averages
historical_data = calculate_moving_averages(historical_data, window_short=10, window_long=50)

# Calculate RSI
historical_data = calculate_rsi(historical_data, window=14)

# Calculate MACD
historical_data = calculate_macd(historical_data, fast_period=12, slow_period=26, signal_period=9)
    return data

def analyze_trend(data):
    # Generate an array of x values from 0 to the number of data points - 1
    x_values = range(len(data) - 30, len(data))
    # Extract the last 30 price data points
    y_values = data['Price'][-30:]
    # Calculate the linear regression parameters
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    # Return the slope as the trend indicator
    return slope

def calculate_volatility(prices):
    if len(prices) < 2:
        return 0.0  # Volatility is zero if there is not enough data
    returns = np.diff(prices) / prices[:-1]  # Calculate daily returns
    volatility = np.std(returns)  # Calculate standard deviation of returns

    return volatility


def make_trading_decision(slope, slope_threshold, prediction, volatility, volatility_threshold, in_position):
    trend_strength = abs(slope)
    is_strong_trend = trend_strength > slope_threshold
    is_low_volatility = volatility < volatility_threshold
    
    if slope > 0 and prediction == 1 and not in_position and is_strong_trend and is_low_volatility:
        return "Buy", True
    elif (slope <= 0 or not is_low_volatility) and prediction == 0 and in_position:
        return "Sell", False
    else:
        return "Hold", in_position

def calculate_profit(data, buy_price, sell_price):
    profit = sell_price - buy_price
    return profit

# Main Script
username = 'nsloan91@outlook.com'
password = 'marine2010'
rs.login(username=username, password=password)

# Fetch historical price data
historical_df = fetch_historical_price('ETC', 'hour', 'month')

data = pd.DataFrame(columns=['Timestamp', 'Price', 'MA', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist'])
total_duration = 20 * 60  # 20 minutes in seconds
in_position = False
buy_price = 0.0
sell_price = 0.0
action = ''
predicted_price = 0.0
historical_prices = []
slope_threshold = 0.5
volatility_threshold = 0.2

# Get the current datetime
now = datetime.datetime.now()

# Format the datetime as a string in the desired format
formatted_now = now.strftime("%Y%m%d_%H%M%S")

# Create the file name by appending the formatted datetime to "Results_"
file_name = f"Results_{formatted_now}.csv"

#Directory
directory = "D:/Nick/Source/Results"

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(directory):
    os.makedirs(directory)

# Append the directory to the file name
full_path = os.path.join(directory, file_name)

# Initialize the scaler
scaler = StandardScaler()
scaler_fitted = False

# Initialize the SGDClassifier
sgd_model = SGDClassifier()

# Create a log file name by appending "_logging" before the extension
log_file_name = file_name.replace('.csv', '_logging.csv')

# Set up basic logging
logging.basicConfig(level=logging.INFO, filename=log_file_name)


try:
    max_data_length = 500  # Maximum number of rows to keep in the DataFrame
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = total_duration - elapsed_time

        if remaining_time <= 0:
            break

        timestamp, price = fetch_latest_price()
        new_row = pd.DataFrame({'Timestamp': [timestamp], 'Price': [price]})
        data = pd.concat([data, new_row], ignore_index=True)
            
        # Limit the size of the DataFrame
        if len(data) > max_data_length:
            data = data.tail(max_data_length)

        historical_prices.append(price)  # Store the current price for volatility calculation
        if len(historical_prices) > 30:
            historical_prices.pop(0)  # Remove the oldest price to keep a rolling window of 30 prices

        volatility = calculate_volatility(historical_prices)  # Calculate volatility

        if len(data) >= 30:
            data = calculate_indicators(data, historical_df)
            slope = analyze_trend(data)
            
	    # Calculate future price and generate labels
            lookahead = 1
            data['Future_Price'] = data['Price'].shift(-lookahead)
            data['Label'] = (data['Future_Price'] > data['Price']).astype(int)
            data['Label'] = data['Label'].replace({0: -1})
            data.dropna(inplace=True)

            # Prepare features for scaling and online learning
            features = data[['MA', 'RSI', 'MACD']].values
            labels = data['Label'].values

            # Scale features if not already done
            if not scaler_fitted:
                scaler.fit(features)
                scaler_fitted = True
            features_scaled = scaler.transform(features)

            # Online learning
            sgd_model.partial_fit(features_scaled, labels, classes=np.unique(labels))

            # Make prediction
            prediction, predicted_price = make_prediction(sgd_model, data, scaler)
            action, in_position = make_trading_decision(slope, slope_threshold, prediction, volatility, volatility_threshold, in_position)            

        if action == 'Buy':
            buy_price = data.loc[data.index[-1], 'Price']
        elif action == 'Sell' and buy_price > 0:
            sell_price = data.loc[data.index[-1], 'Price']
            profit = calculate_profit(data, buy_price, sell_price)
            data.loc[data.index[-1], 'Profit'] = profit
            buy_price = 0  # Reset buy price after selling

        data.to_csv(full_path, index=False)
        time.sleep(10)

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    rs.logout()
