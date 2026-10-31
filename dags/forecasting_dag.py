import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
import pandas as pd
import json

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

from src.data_processing.transformer import transform_data
from src.data_storage.snowflake_reader import fetch_stock_data_from_snowflake
from src.forecasting.models import get_forecaster
from src.forecasting.evaluation import backtest_model, ModelEvaluator

# Define the list of tickers to forecast
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
# Define the list of models to use
MODELS = ['arima', 'prophet']
# Define the forecast horizon (in days)
FORECAST_HORIZON = 30
# Define the lookback period (in days)
LOOKBACK_DAYS = 365

@dag(
    dag_id='financial_forecasting_dag',
    start_date=days_ago(1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['finance', 'forecasting'],
    default_args={
        'owner': 'airflow',
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }
)
def financial_forecasting_dag():
    """
    This DAG performs time series forecasting for financial data.
    It fetches stock data from Snowflake, trains forecasting models, 
    and evaluates their performance.
    """

    @task
    def extract_stock_data_from_snowflake(ticker: str):
        """
        Extract historical stock data for a given ticker from Snowflake.
        """
        df = fetch_stock_data_from_snowflake(ticker, lookback_days=LOOKBACK_DAYS)
        
        # If data is not available in Snowflake, return empty DataFrame
        if df.empty:
            print(f"No data found in Snowflake for {ticker}. Forecasting will be skipped.")
            
        return df.to_json()

    @task
    def check_data_availability(json_df: str, ticker: str):
        """
        Check if data is available for forecasting.
        """
        df = pd.read_json(json_df)
        
        if df.empty:
            print(f"Skipping forecasting for {ticker} due to missing data.")
            return False
        
        # Check if we have enough data points for forecasting
        min_required_points = 50  # Minimum data points needed for reliable forecasting
        if len(df) < min_required_points:
            print(f"Not enough data points for {ticker}. Found {len(df)}, need at least {min_required_points}.")
            return False
            
        return True

    @task
    def transform_stock_data(json_df: str):
        """
        Transform the stock data by adding technical indicators.
        """
        df = pd.read_json(json_df)
        
        # Skip transformation if DataFrame is empty
        if df.empty:
            return df.to_json()
            
        df.index = pd.to_datetime(df.index)
        transformed_df = transform_data(df)
        return transformed_df.to_json()

    @task
    def train_and_forecast(json_df: str, ticker: str, model_type: str):
        """
        Train a forecasting model and generate predictions.
        """
        df = pd.read_json(json_df)
        
        # Skip if DataFrame is empty
        if df.empty:
            return json.dumps({
                'ticker': ticker,
                'model_type': model_type,
                'status': 'skipped',
                'reason': 'No data available'
            })
            
        df.index = pd.to_datetime(df.index)
        
        # Get the appropriate forecaster
        forecaster = get_forecaster(model_type)
        
        # Train the model
        forecaster.train(df, target_col='Close')
        
        # Generate forecast
        forecast = forecaster.predict(periods=FORECAST_HORIZON)
        
        # Save the model
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f'{ticker}_{model_type}_model.pkl')
        forecaster.save_model(model_path)
        
        # Return the forecast
        forecast_dict = {
            'ticker': ticker,
            'model_type': model_type,
            'forecast_dates': forecast.index.strftime('%Y-%m-%d').tolist(),
            'forecast_values': forecast.values.tolist(),
            'model_path': model_path,
            'status': 'success'
        }
        return json.dumps(forecast_dict)

    @task
    def evaluate_models(json_df: str, ticker: str, model_forecasts: list):
        """
        Evaluate and compare different forecasting models.
        """
        df = pd.read_json(json_df)
        
        # Skip if DataFrame is empty
        if df.empty:
            return json.dumps({
                'ticker': ticker,
                'status': 'skipped',
                'reason': 'No data available'
            })
            
        df.index = pd.to_datetime(df.index)
        
        # Filter out skipped models
        valid_forecasts = []
        for model_forecast_json in model_forecasts:
            model_forecast = json.loads(model_forecast_json)
            if model_forecast.get('status') == 'success':
                valid_forecasts.append(model_forecast_json)
        
        # If no valid forecasts, skip evaluation
        if not valid_forecasts:
            return json.dumps({
                'ticker': ticker,
                'status': 'skipped',
                'reason': 'No successful model forecasts'
            })
        
        # Split data for backtesting
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate each model
        results = {}
        for model_forecast_json in valid_forecasts:
            model_forecast = json.loads(model_forecast_json)
            model_type = model_forecast['model_type']
            
            # Get the appropriate forecaster
            forecaster = get_forecaster(model_type)
            
            # Backtest the model
            backtest_results = backtest_model(forecaster, df)
            
            # Add to evaluator
            evaluator.add_forecast(model_type, backtest_results['y_true'], backtest_results['y_pred'])
            
            # Store results
            results[model_type] = backtest_results['metrics']
        
        # Get metrics as DataFrame
        metrics_df = evaluator.get_metrics_df()
        
        # Get best model
        best_model = evaluator.get_best_model()
        
        # Save evaluation results
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(results_dir, f'{ticker}_model_metrics.csv')
        metrics_df.to_csv(metrics_path)
        
        # Return evaluation summary
        evaluation_summary = {
            'ticker': ticker,
            'best_model': best_model,
            'metrics_path': metrics_path,
            'model_metrics': {model: {k: float(v) for k, v in metrics.items()} 
                             for model, metrics in results.items()},
            'status': 'success'
        }
        return json.dumps(evaluation_summary)

    # Workflow for each ticker
    for ticker in TICKERS:
        # Extract data from Snowflake
        json_df = extract_stock_data_from_snowflake(ticker)
        
        # Check data availability
        data_available = check_data_availability(json_df, ticker)
        
        # Transform data if available
        transformed_json_df = transform_stock_data(json_df)
        
        # Train models and generate forecasts
        model_forecasts = []
        for model_type in MODELS:
            forecast_json = train_and_forecast(
                transformed_json_df, 
                ticker, 
                model_type
            )
            model_forecasts.append(forecast_json)
        
        # Evaluate models
        evaluation_summary = evaluate_models(
            transformed_json_df,
            ticker,
            model_forecasts
        )

# Instantiate the DAG
financial_forecasting_dag_instance = financial_forecasting_dag()