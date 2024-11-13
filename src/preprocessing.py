import pandas as pd
import matplotlib.pyplot as plt

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
