import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class StockVisualizer:
    """Class for creating stock data visualizations."""
    
    def __init__(self, output_dir='./visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set(rc={'figure.figsize': (12, 8)})
    
    def plot_price_history(self, df, ticker, show_volume=True, save=True):
        """
        Plot price history with volume.
        
        Args:
            df (pd.DataFrame): Stock data DataFrame
            ticker (str): Stock ticker symbol
            show_volume (bool): Whether to include volume subplot
            save (bool): Whether to save the plot to file
            
        Returns:
            str: Path to saved file if save=True, None otherwise
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            fig.subplots_adjust(hspace=0.1)
        else:
            fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
        
        # Add moving averages if available
        if 'sma_20' in df.columns:
            ax1.plot(df.index, df['sma_20'], label='20-day SMA', linestyle='--', color='orange')
        if 'sma_50' in df.columns:
            ax1.plot(df.index, df['sma_50'], label='50-day SMA', linestyle='-.', color='green')
        
        ax1.set_title(f'{ticker} Price History', fontsize=16)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if show_volume:
            # Plot volume
            ax2.bar(df.index, df['Volume'], color='gray', alpha=0.7)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis to show dates nicely
            plt.gcf().autofmt_xdate()
        else:
            ax1.set_xlabel('Date', fontsize=12)
            plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_price_history_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            return filepath
        else:
            plt.tight_layout()
            plt.show()
            return None
    
    def plot_technical_indicators(self, df, ticker, indicators=None, save=True):
        """
        Plot technical indicators.
        
        Args:
            df (pd.DataFrame): Stock data DataFrame with indicators
            ticker (str): Stock ticker symbol
            indicators (list): List of indicators to plot, defaults to ['rsi_14']
            save (bool): Whether to save the plot to file
            
        Returns:
            str: Path to saved file if save=True, None otherwise
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")
        
        if indicators is None:
            indicators = ['rsi_14']
        
        fig, axes = plt.subplots(len(indicators), 1, figsize=(14, 6 * len(indicators)))
        
        # Handle single indicator case
        if len(indicators) == 1:
            axes = [axes]
        
        for i, indicator in enumerate(indicators):
            if indicator not in df.columns:
                print(f"Warning: {indicator} not found in DataFrame")
                continue
            
            axes[i].plot(df.index, df[indicator], label=indicator)
            axes[i].set_title(f'{ticker} - {indicator}', fontsize=14)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Add special formatting for specific indicators
            if indicator == 'rsi_14':
                axes[i].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                axes[i].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                axes[i].fill_between(df.index, 70, 100, color='r', alpha=0.1)
                axes[i].fill_between(df.index, 0, 30, color='g', alpha=0.1)
                axes[i].set_ylim(0, 100)
        
        # Set common x-label
        fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
        plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            indicators_str = '_'.join(indicators)
            filename = f"{ticker}_indicators_{indicators_str}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            return filepath
        else:
            plt.tight_layout()
            plt.show()
            return None
    
    def plot_forecast(self, historical_df, forecast_df, ticker, model_name='Model', save=True):
        """
        Plot historical data with forecast.
        
        Args:
            historical_df (pd.DataFrame or pd.Series): Historical price data
            forecast_df (pd.DataFrame or pd.Series): Forecast data
            ticker (str): Stock ticker symbol
            model_name (str): Name of the forecasting model
            save (bool): Whether to save the plot to file
            
        Returns:
            str: Path to saved file if save=True, None otherwise
        """
        # Convert to Series if DataFrames were provided
        if isinstance(historical_df, pd.DataFrame):
            if 'Close' in historical_df.columns:
                historical_series = historical_df['Close']
            else:
                historical_series = historical_df.iloc[:, 0]
        else:
            historical_series = historical_df
            
        if isinstance(forecast_df, pd.DataFrame):
            forecast_series = forecast_df.iloc[:, 0]
        else:
            forecast_series = forecast_df
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(historical_series.index, historical_series.values, 
                 label='Historical Data', color='blue')
        
        # Plot forecast data
        plt.plot(forecast_series.index, forecast_series.values, 
                 label=f'{model_name} Forecast', color='red', linestyle='--')
        
        # Add shading to forecast area
        min_forecast_date = forecast_series.index.min()
        plt.axvspan(min_forecast_date, forecast_series.index.max(), 
                   alpha=0.1, color='red')
        
        # Add vertical line at forecast start
        plt.axvline(x=min_forecast_date, color='black', linestyle='-.')
        
        plt.title(f'{ticker} Price Forecast - {model_name}', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_forecast_{model_name}_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            return filepath
        else:
            plt.tight_layout()
            plt.show()
            return None
    
    def plot_forecast_comparison(self, historical_df, forecasts_dict, ticker, save=True):
        """
        Plot historical data with multiple forecasts for comparison.
        
        Args:
            historical_df (pd.DataFrame or pd.Series): Historical price data
            forecasts_dict (dict): Dictionary of forecast Series with model names as keys
            ticker (str): Stock ticker symbol
            save (bool): Whether to save the plot to file
            
        Returns:
            str: Path to saved file if save=True, None otherwise
        """
        # Convert to Series if DataFrame was provided
        if isinstance(historical_df, pd.DataFrame):
            if 'Close' in historical_df.columns:
                historical_series = historical_df['Close']
            else:
                historical_series = historical_df.iloc[:, 0]
        else:
            historical_series = historical_df
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(historical_series.index, historical_series.values, 
                 label='Historical Data', color='blue')
        
        # Plot each forecast with a different color
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        for i, (model_name, forecast_series) in enumerate(forecasts_dict.items()):
            color = colors[i % len(colors)]
            plt.plot(forecast_series.index, forecast_series.values, 
                     label=f'{model_name} Forecast', color=color, linestyle='--')
        
        # Find the earliest forecast date
        min_forecast_date = min([forecast.index.min() for forecast in forecasts_dict.values()])
        
        # Add shading to forecast area
        plt.axvspan(min_forecast_date, max([forecast.index.max() for forecast in forecasts_dict.values()]), 
                   alpha=0.1, color='gray')
        
        # Add vertical line at forecast start
        plt.axvline(x=min_forecast_date, color='black', linestyle='-.')
        
        plt.title(f'{ticker} Price Forecast Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_forecast_comparison_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            return filepath
        else:
            plt.tight_layout()
            plt.show()
            return None


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
    
    # Create visualizer
    visualizer = StockVisualizer()
    
    # Plot price history
    visualizer.plot_price_history(transformed_df, ticker)
    
    # Plot technical indicators
    visualizer.plot_technical_indicators(transformed_df, ticker, indicators=['rsi_14', 'sma_20', 'sma_50'])
    
    # Generate forecasts
    arima_forecaster = get_forecaster('arima')
    arima_forecaster.train(transformed_df)
    arima_forecast = arima_forecaster.predict(periods=30)
    
    prophet_forecaster = get_forecaster('prophet')
    prophet_forecaster.train(transformed_df)
    prophet_forecast = prophet_forecaster.predict(periods=30)
    
    # Plot individual forecasts
    visualizer.plot_forecast(transformed_df, arima_forecast, ticker, model_name='ARIMA')
    visualizer.plot_forecast(transformed_df, prophet_forecast, ticker, model_name='Prophet')
    
    # Plot forecast comparison
    forecasts = {
        'ARIMA': arima_forecast,
        'Prophet': prophet_forecast
    }
    visualizer.plot_forecast_comparison(transformed_df, forecasts, ticker)
