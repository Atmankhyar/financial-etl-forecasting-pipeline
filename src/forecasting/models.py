import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from datetime import datetime, timedelta

class TimeSeriesForecaster:
    """Base class for time series forecasting models."""
    
    def __init__(self, model_name='base'):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
    
    def preprocess(self, df):
        """Preprocess data for forecasting."""
        raise NotImplementedError
    
    def train(self, df, target_col):
        """Train the forecasting model."""
        raise NotImplementedError
    
    def predict(self, periods):
        """Generate forecasts for future periods."""
        raise NotImplementedError
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance."""
        results = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
        return results
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


class ARIMAForecaster(TimeSeriesForecaster):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order=(5,1,0)):
        super().__init__(model_name='arima')
        self.order = order
        self.train_data = None
    
    def preprocess(self, df, target_col='Close'):
        """Preprocess data for ARIMA model."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Extract the target column
        target_series = df[target_col]
        
        return target_series
    
    def train(self, df, target_col='Close'):
        """Train ARIMA model on the provided data."""
        target_series = self.preprocess(df, target_col)
        self.train_data = target_series
        
        # Fit ARIMA model
        self.model = ARIMA(target_series, order=self.order)
        self.fitted_model = self.model.fit()
        
        return self.fitted_model
    
    def predict(self, periods=30):
        """Generate forecasts for future periods."""
        if self.fitted_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=periods)
        
        # Create date index for forecast
        last_date = self.train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Return forecast as Series with date index
        forecast_series = pd.Series(forecast, index=forecast_index)
        return forecast_series


class ProphetForecaster(TimeSeriesForecaster):
    """Facebook Prophet model for time series forecasting."""
    
    def __init__(self):
        super().__init__(model_name='prophet')
        self.last_date = None
    
    def preprocess(self, df, target_col='Close'):
        """Preprocess data for Prophet model."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        
        # Prophet requires a specific dataframe format with 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_col]
        })
        
        self.last_date = df.index[-1]
        return prophet_df
    
    def train(self, df, target_col='Close'):
        """Train Prophet model on the provided data."""
        prophet_df = self.preprocess(df, target_col)
        
        # Initialize and fit Prophet model
        self.model = Prophet(daily_seasonality=True)
        self.model.fit(prophet_df)
        
        return self.model
    
    def predict(self, periods=30):
        """Generate forecasts for future periods."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Create future dataframe for predictions
        future = self.model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract only the future predictions
        future_forecast = forecast.iloc[-periods:]
        
        # Return forecast as Series with date index
        forecast_series = pd.Series(future_forecast['yhat'].values, 
                                   index=pd.DatetimeIndex(future_forecast['ds']))
        return forecast_series


def get_forecaster(model_type='arima', **kwargs):
    """Factory function to get the appropriate forecaster."""
    if model_type.lower() == 'arima':
        return ARIMAForecaster(**kwargs)
    elif model_type.lower() == 'prophet':
        return ProphetForecaster(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Example usage
    from src.data_ingestion.yfinance_extractor import fetch_stock_data
    from src.data_processing.transformer import transform_data
    
    # Fetch and transform some data
    ticker = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    df = fetch_stock_data(ticker, start_date, end_date)
    transformed_df = transform_data(df)
    
    # Train ARIMA model
    arima_forecaster = get_forecaster('arima', order=(5,1,0))
    arima_forecaster.train(transformed_df)
    arima_forecast = arima_forecaster.predict(periods=30)
    
    print(f"ARIMA Forecast for {ticker} (next 30 days):")
    print(arima_forecast.head())
    
    # Train Prophet model
    prophet_forecaster = get_forecaster('prophet')
    prophet_forecaster.train(transformed_df)
    prophet_forecast = prophet_forecaster.predict(periods=30)
    
    print(f"\nProphet Forecast for {ticker} (next 30 days):")
    print(prophet_forecast.head())
