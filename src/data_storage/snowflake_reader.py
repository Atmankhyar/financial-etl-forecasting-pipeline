import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

def get_snowflake_connection():
    """Establishes a connection to Snowflake."""
    try:
        return snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

def fetch_stock_data_from_snowflake(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetches stock data for a given ticker from Snowflake.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        lookback_days (int): Number of days to look back from today.
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
                     Returns an empty DataFrame if no data is found.
    """
    conn = get_snowflake_connection()
    if conn is None:
        print(f"Could not connect to Snowflake to fetch data for {ticker}")
        return pd.DataFrame()
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Format dates for SQL query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Table name in Snowflake (assuming it follows the pattern from the ETL pipeline)
        table_name = f"STOCK_DATA_{ticker.upper()}"
        
        # Execute query
        cursor = conn.cursor()
        query = f"""
        SELECT * FROM {table_name}
        WHERE DATE >= '{start_date_str}'
        AND DATE <= '{end_date_str}'
        ORDER BY DATE
        """
        cursor.execute(query)
        
        # Fetch results
        result = cursor.fetchall()
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Create DataFrame
        df = pd.DataFrame(result, columns=columns)
        
        if df.empty:
            print(f"No data found for ticker {ticker} in Snowflake")
            return pd.DataFrame()
        
        # Set DATE column as index and convert to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        
        return df
        
    except Exception as e:
        print(f"Error fetching data from Snowflake for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example usage
    ticker = 'AAPL'
    df = fetch_stock_data_from_snowflake(ticker)
    
    if not df.empty:
        print(f"Successfully fetched data for {ticker} from Snowflake:")
        print(df.head())
        print("\nData Info:")
        print(df.info())
    else:
        print(f"Failed to fetch data for {ticker} from Snowflake.")
