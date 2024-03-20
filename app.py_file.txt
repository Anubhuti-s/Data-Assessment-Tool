import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import date, timedelta

# Define the date range
today= date.today()
start = (today - timedelta(days=365*5)).strftime('%Y-%m-%d') 
end = date.today().strftime('%Y-%m-%d')
st.title('Stock Price Prediction')
st.subheader('(With dataset quality assessment)')
st.write('The data is from Yahoo Finance.')
user_input = st.text_input('Enter the ticker of the stock (e.g. AAPL for Apple):', 'AAPL')
user_input = user_input.upper()


# # Load the data
# df = data.DataReader(user_input, 'yahoo', start, end)
# # df = df.reset_index()
# Use yfinance to fetch stock data
try:
    df = yf.download(user_input, start=start, end=end)
except Exception as e:
    st.error(f"An error occurred while fetching data: {str(e)}")


# Describing the data
if not df.empty:
    st.subheader(f'Data for {user_input} from 2010 - 2023')
    st.write(df.describe())


# Visualizing the data
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 50MA')
ma50 = df.Close.rolling(50).mean()  
fig = plt.figure(figsize=(12,6))
plt.plot(ma50, 'r', label='50MA')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA')
ma200 = df.Close.rolling(200).mean() 
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, 'g', label='200MA') 
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 50MA and 200MA')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Close')
plt.plot(ma50, 'r', label='50MA')
plt.plot(ma200, 'g', label='200MA')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
st.pyplot(fig)


# Create a new figure for the candlestick chart
fig_candlestick = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
# Set the layout for the candlestick chart
fig_candlestick.update_layout(
    title=f'Candlestick Chart for {user_input}',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    showlegend=False
)
# Display the candlestick chart
st.subheader('Candlestick Chart')
st.plotly_chart(fig_candlestick)
# Calculate and display common technical indicators
st.subheader('Technical Indicators')
# Calculate Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
# Calculate Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
rsi = 100 - (100 / (1 + rs))
# Display Moving Averages and RSI
fig_technical_indicators = go.Figure()
fig_technical_indicators.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-day SMA'))
fig_technical_indicators.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='200-day SMA'))
fig_technical_indicators.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name='RSI'))
fig_technical_indicators.update_layout(
    title='Technical Indicators',
    xaxis_title='Date',
    yaxis_title='Value',
    xaxis_rangeslider_visible=True
)
st.plotly_chart(fig_technical_indicators)
# Describe the candlestick model
st.subheader('Candlestick Model Description')

# Analyze candlestick patterns
def analyze_candlestick_patterns(df):
    patterns = []
    for i in range(2, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i - 1]
        prev_open = previous['Open']
        prev_close = previous['Close']
        current_open = current['Open']
        current_close = current['Close']
        prev_high = previous['High']
        prev_low = previous['Low']
        current_high = current['High']
        current_low = current['Low']
        
         # Bullish Engulfing Pattern
        if current_open < prev_close and current_close > prev_open:
            patterns.append("Bullish Engulfing")

        # Bearish Engulfing Pattern
        if current_open > prev_close and current_close < prev_open:
            patterns.append("Bearish Engulfing")

        # Hammer Pattern
        if (prev_close - prev_open) > 3 * (prev_high - prev_low) and \
           (current_close - current_open) < 0.1 * (current_high - current_low) and \
           (current_low - current_open) < 0.01 * (current_high - current_low):
            patterns.append("Hammer")

        # Inverted Hammer Pattern
        if (prev_close - prev_open) > 3 * (prev_high - prev_low) and \
           (current_close - current_open) > 0.1 * (current_high - current_low) and \
           (current_high - current_open) < 0.01 * (current_high - current_low):
            patterns.append("Inverted Hammer")
    return patterns

candlestick_patterns = analyze_candlestick_patterns(df)
# Display the detected patterns
st.subheader('Detected Candlestick Patterns')
if candlestick_patterns:
    st.write("Candlestick patterns detected:")
    st.write(candlestick_patterns)
else:
    st.write("No significant candlestick patterns detected.")


# Splitting data into training and testing data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


# Load the model
model = load_model('keras_model.h5')


# Testing data
past_100_days = data_training.head(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])  
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler= scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# Create a new scaler for inverse scaling
# scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
# scaler_for_inverse.fit(data_training['Close'].values.reshape(-1, 1))
# # Inverse scaling of predictions and actual values
# y_predicted = scaler_for_inverse.inverse_transform(y_predicted)
# y_test = scaler_for_inverse.inverse_transform(y_test)


# Visualizing the results
st.subheader('Predictions vs Actual')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Actual Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# bullish trend (when the 50-day moving average is above the 200-day moving average) or a bearish trend
# If the current price is higher than the previous price, it's considered bullish, and if it's lower, it's considered bearish.
# Calculate daily returns for actual and predicted prices
actual_returns = np.diff(y_test)
predicted_returns = np.diff(y_predicted)

# Determine if it's bullish or bearish
bullish_count = len([ret for ret in actual_returns if ret > 0 and predicted_returns[actual_returns.tolist().index(ret)] > 0])
bearish_count = len([ret for ret in actual_returns if ret < 0 and predicted_returns[actual_returns.tolist().index(ret)] < 0])

if bullish_count > bearish_count:
    st.write("The market is currently in a bullish state.")
elif bearish_count > bullish_count:
    st.write("The market is currently in a bearish state.")
else:
    st.write("The market is relatively stable.")


# DATA QUALITY ASSESSMENT
st.subheader('DATA QUALITY ASSESSMENT')
# Checking for missing values in the DataFrame
missing_values = df.isnull().sum()
st.subheader('Missing Values')
st.write(missing_values)

# Check for duplicate rows in the DataFrame
duplicate_rows = df[df.duplicated()]
st.subheader('Duplicate Rows')
st.write(duplicate_rows)

# Check for data types of columns
data_types = df.dtypes
st.subheader('Data Types')
st.write(data_types)

# Checking for outliers in the 'Close' column using a box plot
st.subheader('Outliers in Closing Price')
fig3 = plt.figure(figsize=(6, 4))
plt.boxplot(df['Close'], vert=False)
plt.xlabel('Closing Price')
st.pyplot(fig3)

# Checking for data distribution of the 'Close' column using a histogram
st.subheader('Distribution of Closing Price')
fig4 = plt.figure(figsize=(8, 6))
plt.hist(df['Close'], bins=20, edgecolor='k')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
st.pyplot(fig4)

# Calculate and display basic statistics of the 'Close' column
statistics = df['Close'].describe()
st.subheader('Basic Statistics of Closing Price')
st.write(statistics)

# Check for any abnormal values or patterns in the 'Close' column
abnormal_values = df[df['Close'] < 0]  # Example: Identify negative closing prices
st.subheader('Abnormal Values in Closing Price')
st.write(abnormal_values)

# Check for data quality and consistency in the date range
date_range = df.index
date_range_consistency = date_range.is_unique and (date_range[0] == pd.to_datetime(start)) and (date_range[-1] == pd.to_datetime(end))
st.subheader('Date Range Consistency')
st.write(f"Is the date range consistent? {date_range_consistency}")

# Check for any gaps or missing dates in the date range
date_range_gaps = pd.date_range(start=start, end=end).difference(date_range)
st.subheader('Date Range Gaps (Missing Dates)')
st.write(date_range_gaps)

# Check for any issues in the 50MA and 200MA calculations
if ma50.isna().any() or ma200.isna().any():
    st.subheader('Issues in Moving Averages Calculation')
    st.write("There are missing or NaN values in the moving averages.")

# Check for any issues in data_training and data_testing splits
if data_training.isnull().values.any() or data_testing.isnull().values.any():
    st.subheader('Data Splitting Issues')
    st.write("There are missing values in data_training or data_testing.")

# Check for scaling issues
if np.isnan(data_training_array).any():
    st.subheader('Scaling Issues')
    st.write("There are NaN values in the scaled data_training_array.")

# Check for model loading issues
if model is None:
    st.subheader('Model Loading Issue')
    st.write("The model failed to load.")

# Check for issues in the testing data
if x_test is None or y_test is None:
    st.subheader('Testing Data Issue')
    st.write("There are issues with the testing data.")

# Check for issues in the predictions
if np.isnan(y_predicted).any() or np.isnan(y_test).any():
    st.subheader('Prediction Issues')
    st.write("There are NaN values in the predictions or testing data.")

# Display a summary of data quality
data_quality_summary = "Data quality appears to be generally good."
if missing_values.any():
    data_quality_summary = "There are missing values in the dataset."
if duplicate_rows.shape[0] > 0:
    data_quality_summary = "There are duplicate rows in the dataset."
st.subheader('Data Quality Summary')
st.write(data_quality_summary)



# BACKTESTING AND RISK ASSESSMENT
# Define initial portfolio values
initial_balance = 100000  # Initial portfolio balance
balance = initial_balance
shares_owned = 0

# Lists to track portfolio performance
portfolio_balance = [initial_balance]
daily_returns = []

# Define trading strategy
for i in range(len(y_predicted)):
    prediction = y_predicted[i]
    actual_price = y_test[i]

    # Example: Implement a simple moving average crossover strategy
    if i > 0 and y_predicted[i - 1] < y_test[i - 1] and prediction >= actual_price:
        # Buy signal: cross above
        shares_to_buy = balance // actual_price
        cost = shares_to_buy * actual_price
        balance -= cost
        shares_owned += shares_to_buy

    if i > 0 and y_predicted[i - 1] > y_test[i - 1] and prediction <= actual_price:
        # Sell signal: cross below
        sale_value = shares_owned * actual_price
        balance += sale_value
        shares_owned = 0

    # Calculate daily returns
    daily_return = (balance + shares_owned * actual_price - initial_balance) / initial_balance
    daily_returns.append(daily_return)
    portfolio_balance.append(balance + shares_owned * actual_price)
# Calculate risk metrics
max_drawdown = min(portfolio_balance) - initial_balance
annualized_volatility = np.std(daily_returns) * np.sqrt(252)  # Assuming 252 trading days in a year
# Display results and risk assessment
st.subheader('Backtesting and Risk Assessment')
st.write(f"Initial Portfolio Balance: ${initial_balance:.2f}")
st.write(f"Final Portfolio Balance: ${balance + shares_owned * actual_price:.2f}")
st.write(f"Maximum Drawdown: ${max_drawdown:.2f}")
st.write(f"Annualized Volatility: {annualized_volatility:.2%}")



# Display the model's performance metrics
st.subheader('Model Performance Metrics')
# Calculate Mean Squared Error (MSE), MAE, R2_SCORE between actual and predicted prices
mae = mean_absolute_error(y_test, y_predicted)
rmse = mean_squared_error(y_test, y_predicted, squared=False)
r2 = r2_score(y_test, y_predicted)

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R-squared (R2): {r2:.2f}")
# Plot the residuals (the difference between actual and predicted prices)
residuals = y_test - y_predicted
fig_residuals = plt.figure(figsize=(12, 6))
plt.plot(residuals, 'b', label='Residuals')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Residuals')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
st.pyplot(fig_residuals)

# Visualize the distribution of residuals
st.subheader('Distribution of Residuals')
fig_residuals_hist = plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, edgecolor='k')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
st.pyplot(fig_residuals_hist)

# Display the portfolio balance over time
st.subheader('Portfolio Balance Over Time')
fig_portfolio_balance = plt.figure(figsize=(12, 6))
plt.plot(portfolio_balance, label='Portfolio Balance')
plt.xlabel('Time')
plt.ylabel('Portfolio Balance')
st.pyplot(fig_portfolio_balance)

# Display risk assessment metrics
st.subheader('Risk Assessment Metrics')
st.write(f"Initial Portfolio Balance: ${initial_balance:.2f}")
st.write(f"Final Portfolio Balance: ${portfolio_balance[-1]:.2f}")
st.write(f"Maximum Drawdown: ${max_drawdown:.2f}")
st.write(f"Annualized Volatility: {annualized_volatility:.2%}")


# streamlit run app.py




