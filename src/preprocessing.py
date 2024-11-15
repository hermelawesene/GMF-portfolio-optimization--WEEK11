import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def clean_data(df):
    # Step 1: Check for initial missing values
    print('Initial check for missing values:')
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Step 2: Handle numeric columns and convert to appropriate data types
    if not df.empty:
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coerce errors to NaN if any non-numeric values are found
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f'Warning: Column "{col}" not found in the DataFrame.')

        # Step 3: Forward fill missing values, then backward fill any remaining ones
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill for any still-missing values
        
        # Step 4: Check if any missing values remain after filling
        remaining_missing = df.isnull().sum()
        if remaining_missing.any():
            print('Remaining missing values after fill operations:')
            print(remaining_missing[remaining_missing > 0])
        else:
            print('No missing values remain after fill operations.')

        # Set 'Date' as the index name if applicable
        #df.index.name = 'Date'
    else:
        print('Warning: DataFrame is empty.')

    return df
def closing_price(df, title):
    # Convert 'Date' column to datetime, coercing errors to NaT (Not a Time)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Plot Close price with Date as index
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label=f'{title} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.title(f'{title} Closing Price Over Time')
    plt.legend()
    plt.show()

def rolling_volatility(df, title):
    # Ensure 'Date' is in datetime format if not already
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the daily percentage change (if not already calculated)
    df['Daily Return'] = df['Adj Close'].pct_change()

    # Calculate rolling volatility over a 21-day window (approximately 1 month)
    df['Volatility'] = df['Daily Return'].rolling(window=21).std() * (252 ** 0.5)  # Annualized volatility

    # Plot rolling volatility with Date as x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Volatility'], label=f'{title} Rolling Volatility', color='red', alpha=0.6)
    plt.title(f'Rolling Volatility (21-day) - {title}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

def rolling_mean_volatility(df,title):
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the daily percentage change in 'Adj Close' prices (daily return)
    df['Daily Return'] = df['Adj Close'].pct_change()

    # Define the rolling window size, here 21 days
    rolling_window = 21

    # Calculate the rolling mean of daily returns
    df['Rolling Mean'] = df['Daily Return'].rolling(window=rolling_window).mean()

    # Calculate the rolling standard deviation of daily returns (volatility)
    df['Rolling Std Dev'] = df['Daily Return'].rolling(window=rolling_window).std() #* (252 ** 0.5)  # Annualized

    # Plot both the a mean and rolling standard deviation (volatility)
    plt.figure(figsize=(14, 7))

    # Plot Rolling Mean of Daily Returns to show trend
    plt.plot(df['Date'], df['Rolling Mean'], label='21-Day Rolling Mean of Daily Returns', color='blue', linewidth=1.5)

    # Plot Rolling Standard Deviation of Daily Returns to show volatility
    plt.plot(df['Date'], df['Rolling Std Dev'], label='21-Day Rolling Volatility (Std Dev)', color='red', linewidth=1.5)

    # Plot labels and title
    plt.title(f'Rolling Mean and Volatility (21-Day) of {title} Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()

def anomaly_detection(df):
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate daily returns
    df['Daily Return'] = df['Adj Close'].pct_change()

    # Drop NaN values caused by pct_change()
    df = df.dropna()

    # 1. Outlier Detection Using Z-Score
    mean_return = df['Daily Return'].mean()
    std_return = df['Daily Return'].std()

    # Calculate Z-scores
    df['Z-Score'] = (df['Daily Return'] - mean_return) / std_return

    # Identify anomalies (|Z| > 3)
    anomalies_z = df[np.abs(df['Z-Score']) > 3]

    # 2. Outlier Detection Using IQR
    Q1 = df['Daily Return'].quantile(0.25)
    Q3 = df['Daily Return'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies_iqr = df[(df['Daily Return'] < lower_bound) | (df['Daily Return'] > upper_bound)]

    # Plotting Anomalies
    plt.figure(figsize=(14, 7))

    # Plot Daily Returns
    sns.lineplot(x=df['Date'], y=df['Daily Return'], label='Daily Returns', color='blue', linewidth=1.5)

    # Highlight Anomalies (Z-Score)
    plt.scatter(anomalies_z['Date'], anomalies_z['Daily Return'], color='red', label='Anomalies (Z-Score)', s=50, edgecolor='black')

    # Highlight Anomalies (IQR)
    plt.scatter(anomalies_iqr['Date'], anomalies_iqr['Daily Return'], color='orange', label='Anomalies (IQR)', s=50, edgecolor='black')

    # Plot labels and title
    plt.title('Daily Returns with Anomalies (Z-Score and IQR)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Daily Return', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show plot
    plt.show()

    # Analyze Days with Unusual Returns
    print("Anomalies Detected Using Z-Score:")
    print(anomalies_z[['Date', 'Daily Return', 'Z-Score']])

    print("\nAnomalies Detected Using IQR:")
    print(anomalies_iqr[['Date', 'Daily Return']])


def decompose_timeseries(df):

    # Ensure 'Date' is in datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Decompose the time series (use 'Adj Close' as the series to decompose)
    result = seasonal_decompose(df['Adj Close'], model='additive', period=365)  # Assuming daily data with a yearly seasonality

    # Plot the decomposition
    plt.figure(figsize=(14, 10))

    # Observed
    plt.subplot(4, 1, 1)
    plt.plot(result.observed, label='Observed', color='blue')
    plt.title('Observed')
    plt.legend(loc='upper left')

    # Trend
    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend', color='orange')
    plt.title('Trend')
    plt.legend(loc='upper left')

    # Seasonal
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal', color='green')
    plt.title('Seasonal')
    plt.legend(loc='upper left')

    # Residual
    plt.subplot(4, 1, 4)
    plt.plot(result.resid, label='Residual', color='red')
    plt.title('Residual')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
