import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

class ModelEvaluator:
    """Class for evaluating time series forecasting models."""
    
    def __init__(self):
        self.metrics = {}
        self.forecasts = {}
    
    def add_forecast(self, model_name, y_true, y_pred):
        """Add a forecast to the evaluator."""
        self.forecasts[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def calculate_metrics(self):
        """Calculate evaluation metrics for all forecasts."""
        for model_name, data in self.forecasts.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            self.metrics[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }
    
    def get_best_model(self, metric='RMSE'):
        """Get the best performing model based on a specific metric."""
        if not self.metrics:
            self.calculate_metrics()
        
        # Lower is better for error metrics
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            best_model = min(self.metrics.items(), key=lambda x: x[1][metric])
        # Higher is better for R²
        elif metric == 'R²':
            best_model = max(self.metrics.items(), key=lambda x: x[1][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model[0]
    
    def get_metrics_df(self):
        """Return metrics as a DataFrame."""
        if not self.metrics:
            self.calculate_metrics()
        
        return pd.DataFrame(self.metrics).T
    
    def plot_forecasts(self, title="Model Forecasts Comparison"):
        """Plot actual vs predicted values for all models."""
        if not self.forecasts:
            raise ValueError("No forecasts available to plot.")
        
        plt.figure(figsize=(12, 6))
        
        # Get a sample forecast to determine the actual values
        sample_model = list(self.forecasts.keys())[0]
        y_true = self.forecasts[sample_model]['y_true']
        
        # Plot actual values
        plt.plot(y_true.index, y_true.values, 'k-', label='Actual')
        
        # Plot predictions for each model
        colors = ['b', 'r', 'g', 'c', 'm', 'y']
        for i, (model_name, data) in enumerate(self.forecasts.items()):
            color = colors[i % len(colors)]
            y_pred = data['y_pred']
            plt.plot(y_pred.index, y_pred.values, f'{color}--', label=f'{model_name} Forecast')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_comparison_{timestamp}.png"
        plt.savefig(filename)
        
        plt.close()
        return filename


def backtest_model(model, df, target_col='Close', test_size=0.2, periods=30):
    """
    Backtest a forecasting model on historical data.
    
    Args:
        model: A forecaster instance with train and predict methods
        df: DataFrame with time series data
        target_col: Target column for forecasting
        test_size: Proportion of data to use for testing
        periods: Number of periods to forecast
    
    Returns:
        dict: Dictionary with actual and predicted values, and evaluation metrics
    """
    # Split data into train and test sets
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Train the model
    model.train(train_df, target_col=target_col)
    
    # Make predictions
    predictions = model.predict(periods=len(test_df))
    
    # Evaluate the model
    y_true = test_df[target_col]
    y_pred = predictions
    
    # Ensure indices match for comparison
    if not y_true.index.equals(y_pred.index):
        # Align indices for fair comparison
        common_dates = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_dates]
        y_pred = y_pred.loc[common_dates]
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': metrics
    }


if __name__ == '__main__':
    # Example usage
    from src.data_ingestion.yfinance_extractor import fetch_stock_data
    from src.data_processing.transformer import transform_data
    from src.forecasting.models import get_forecaster
    from datetime import datetime, timedelta
    
    # Fetch and transform some data
    ticker = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = fetch_stock_data(ticker, start_date, end_date)
    transformed_df = transform_data(df)
    
    # Backtest ARIMA model
    arima_forecaster = get_forecaster('arima')
    arima_results = backtest_model(arima_forecaster, transformed_df)
    
    # Backtest Prophet model
    prophet_forecaster = get_forecaster('prophet')
    prophet_results = backtest_model(prophet_forecaster, transformed_df)
    
    # Compare models
    evaluator = ModelEvaluator()
    evaluator.add_forecast('ARIMA', arima_results['y_true'], arima_results['y_pred'])
    evaluator.add_forecast('Prophet', prophet_results['y_true'], prophet_results['y_pred'])
    
    # Get metrics
    metrics_df = evaluator.get_metrics_df()
    print("Model Performance Metrics:")
    print(metrics_df)
    
    # Get best model
    best_model = evaluator.get_best_model()
    print(f"\nBest model based on RMSE: {best_model}")
    
    # Plot forecasts
    plot_file = evaluator.plot_forecasts(title=f"{ticker} Forecast Comparison")
    print(f"\nPlot saved as: {plot_file}")
