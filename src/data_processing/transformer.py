import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw stock data by resampling, filling missing values,
    and adding technical indicators.

    Args:
        df (pd.DataFrame): The raw stock data with a DatetimeIndex.

    Returns:
        pd.DataFrame: The transformed DataFrame with technical indicators.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the data we need while preserving the original structure
        close_col = ('Close', df.columns.get_level_values(1)[0])
        
        # Create a simplified DataFrame with single-level columns
        df_processed = pd.DataFrame({
            'Close': df[close_col],
            'Open': df[('Open', df.columns.get_level_values(1)[0])],
            'High': df[('High', df.columns.get_level_values(1)[0])],
            'Low': df[('Low', df.columns.get_level_values(1)[0])],
            'Volume': df[('Volume', df.columns.get_level_values(1)[0])]
        }, index=df.index)
    
    # Ensure daily frequency and fill missing values
    df_processed = df_processed.asfreq('D').ffill()

    # Calculate technical indicators
    # Simple Moving Averages (SMA)
    df_processed['sma_20'] = SMAIndicator(close=df_processed['Close'], window=20).sma_indicator()
    df_processed['sma_50'] = SMAIndicator(close=df_processed['Close'], window=50).sma_indicator()

    # Relative Strength Index (RSI)
    df_processed['rsi_14'] = RSIIndicator(close=df_processed['Close'], window=14).rsi()

    # Drop rows with NaN values created by indicators
    df_processed.dropna(inplace=True)

    return df_processed

if __name__ == '__main__':

    data = {
        'Open': [150.0, 151.2, 153.4, 152.1, 155.0],
        'High': [152.5, 153.4, 155.0, 154.5, 156.2],
        'Low': [149.5, 150.8, 152.0, 151.5, 154.0],
        'Close': [151.8, 152.9, 154.5, 153.8, 155.5],
        'Adj Close': [151.8, 152.9, 154.5, 153.8, 155.5],
        'Volume': [1000000, 1200000, 1100000, 1300000, 1050000]
    }
    sample_df = pd.DataFrame(data, index=pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-06', '2023-01-07']))

    # To test indicators, we need more data
    from src.data_ingestion.yfinance_extractor import fetch_stock_data
    from datetime import date, timedelta
    
    end_date = date.today()
    start_date = end_date - timedelta(days=100)
    
    ticker = 'MSFT'
    msft_data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if not msft_data.empty:
        transformed_df = transform_data(msft_data)
        print(f"Transformed data for {ticker}:")
        print(transformed_df.head())
        print("\nData Info:")
        transformed_df.info()
        print("\nColumns with NaN values:")
        print(transformed_df.isnull().sum())
    else:
        print(f"Could not fetch data for {ticker} to run transformation test.") 