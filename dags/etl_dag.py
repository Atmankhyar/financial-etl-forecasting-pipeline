import sys
from pathlib import Path
from datetime import datetime, timedelta


project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from airflow.decorators import dag, task
from src.data_ingestion.yfinance_extractor import fetch_stock_data
from src.data_processing.transformer import transform_data
from src.data_storage.snowflake_loader import load_df_to_snowflake


TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

@dag(
    dag_id='financial_etl_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['finance', 'etl'],
    default_args={
        'owner': 'airflow',
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
    }
)
def financial_etl_dag():
    """
    This DAG performs the ETL process for financial time series data.
    It extracts data from Yahoo Finance, transforms it, and loads it into Snowflake.
    """

    @task
    def extract_data(ticker: str):
        """
        Extracts historical stock data for a given ticker.
        The data is stored in XComs to be passed to the next task.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        df = fetch_stock_data(ticker, start_date, end_date)
        # Convert DataFrame to JSON to pass it through XComs
        return df.to_json()

    @task
    def transform(json_df: str):
        """
        Transforms the extracted data.
        It converts the JSON string back to a DataFrame, applies transformations,
        and then returns the processed DataFrame as JSON.
        """
        import pandas as pd
        df = pd.read_json(json_df)
        df.index = pd.to_datetime(df.index)
        transformed_df = transform_data(df)
        return transformed_df.to_json()

    @task
    def load(json_df: str, ticker: str):
        """
        Loads the transformed data into a Snowflake table.
        The table name is dynamically created based on the ticker.
        """
        import pandas as pd
        df = pd.read_json(json_df)
        df.index = pd.to_datetime(df.index)
        table_name = f'stock_data_{ticker}'
        load_df_to_snowflake(df, table_name)


    for ticker in TICKERS:
        extracted_data_json = extract_data.override(task_id=f'extract_{ticker}')(ticker)
        transformed_data_json = transform.override(task_id=f'transform_{ticker}')(extracted_data_json)
        load.override(task_id=f'load_{ticker}')(transformed_data_json, ticker)


financial_etl_dag_instance = financial_etl_dag() 