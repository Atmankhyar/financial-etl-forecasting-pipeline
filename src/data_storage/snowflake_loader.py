
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

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

def load_df_to_snowflake(df: pd.DataFrame, table_name: str):
    """
    Loads a DataFrame into a specified Snowflake table.

    The table will be created if it does not exist. The function is idempotent.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        table_name (str): The name of the target table in Snowflake.
    """
    conn = get_snowflake_connection()
    if conn is None:
        return

    try:
        # Snowflake table names are case-insensitive by default and often stored as uppercase.
        table_name = table_name.upper()
        
        # Reset index to make the date index a column
        df_to_load = df.reset_index()
        df_to_load.rename(columns={'index': 'date'}, inplace=True)
        
        # Use the write_pandas utility
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df_to_load,
            table_name=table_name,
            auto_create_table=True,
            overwrite=True  # Overwrite table if it exists to ensure fresh data
        )
        if success:
            print(f"Successfully loaded {nrows} rows into {table_name}.")
        else:
            print(f"Failed to load data into {table_name}.")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # This is an example of how to use the loader.
    # NOTE: This will only work if you have a .env file with valid Snowflake credentials.
    
    # Create a sample DataFrame
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'Open': [100, 102, 101],
        'High': [103, 104, 102],
        'Low': [99, 101, 100],
        'Close': [102, 103, 101],
        'Volume': [1000, 1200, 1100],
        'sma_20': [101, 101.5, 101.2],
        'rsi_14': [50, 55, 52]
    }
    sample_df = pd.DataFrame(data).set_index('date')
    
    table_name = 'stock_data_test'
    
    print("Attempting to load sample data to Snowflake...")
    load_df_to_snowflake(sample_df, table_name)
    print("Sample data loading process finished.") 