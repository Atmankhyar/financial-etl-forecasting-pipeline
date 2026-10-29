import yfinance as yf
import pandas as pd
from datetime import date

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
                      Returns an empty DataFrame if no data is found.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if stock_data.empty:
        print(f"No data found for ticker {ticker} from {start_date} to {end_date}")
    return stock_data

if __name__ == '__main__':
    # Example usage:
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = date.today().strftime('%Y-%m-%d')
    
    aapl_data = fetch_stock_data(ticker, start_date, end_date)
    
    if not aapl_data.empty:
        print(f"Successfully fetched data for {ticker}:")
        print(aapl_data.head())
        print("\nData Info:")
        aapl_data.info() 