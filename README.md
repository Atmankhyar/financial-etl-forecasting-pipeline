# Financial Data Engineering Pipeline

**Tech Stack:** Python • Apache Airflow • Snowflake • Power BI • Docker • Yahoo Finance API

A complete data engineering solution that automates financial data collection, processing, and analysis. Built to demonstrate modern ETL practices, time series forecasting, and business intelligence integration.

## Architecture Overview

This pipeline handles the full data lifecycle:

1. **Data Extraction** - Automated daily pulls from Yahoo Finance API
2. **Transformation** - Technical indicator calculations and data quality checks  
3. **Storage** - Structured loading into Snowflake data warehouse
4. **Forecasting** - Weekly ARIMA and Prophet model execution
5. **Visualization** - Real-time Power BI dashboards with DirectQuery

![System Architecture](images/architecture.png)

The complete system architecture shows the data flow from source to visualization, with Snowflake as the central data repository connecting both ML models and Power BI dashboards.

![Airflow Pipeline Management](images/airflow-ui.png)

The Airflow orchestration manages two main workflows: the daily ETL process (`financial_etl_dag`) and weekly forecasting pipeline (`financial_forecasting_dag`). This separation allows for efficient resource utilization while maintaining clear data dependencies.

## Technical Implementation

### Data Processing Pipeline

```
Raw Yahoo Finance Data → Technical Indicators → Snowflake Tables → BI Dashboards
```

**Key Features:**
- Resilient error handling with automatic retries
- Data validation at each transformation step
- Scalable architecture supporting multiple stock symbols
- Containerized deployment with Docker Compose

### Technical Indicators Implemented

- **Simple Moving Averages** (SMA-20, SMA-50)
- **Relative Strength Index** (RSI)
- **Trend Classification** (Bullish/Bearish signals)
- **Volume Analysis** patterns

### Forecasting Models

- **ARIMA** - Classical time series analysis for trend prediction
- **Prophet** - Facebook's forecasting tool for seasonal patterns
- **Performance Metrics** - RMSE, MAE evaluation and backtesting

## Business Intelligence Dashboard

![Financial Analytics Dashboard](images/powerBI-dashboard.png)

**Dashboard Components:**
- **Price Evolution**: OHLC candlestick patterns with volume correlation
- **Technical Analysis**: Moving average crossovers and trend indicators  
- **Volume Patterns**: Trading activity analysis for market sentiment
- **Trend Signals**: Automated buy/sell signal generation

The Power BI implementation uses DirectQuery to Snowflake, ensuring real-time data access without manual refreshes.

## System Requirements

**Infrastructure:**
- Docker & Docker Compose
- Python 3.8+ runtime
- Snowflake data warehouse access
- Minimum 4GB RAM for Airflow services

**External Dependencies:**
- Yahoo Finance API (via yfinance library)
- Power BI Desktop for dashboard access

## Setup & Configuration

**Environment Configuration:**
```bash
# Core Snowflake connection
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=password  
SNOWFLAKE_ACCOUNT=account_identifier
SNOWFLAKE_WAREHOUSE=compute_warehouse
SNOWFLAKE_DATABASE=finance_db
SNOWFLAKE_SCHEMA=stock_data

# Pipeline parameters
STOCK_SYMBOLS=AAPL,GOOGL,MSFT,TSLA
```

**Deployment:**
```bash
# Initialize environment
git clone [repository]
pip install -r requirements.txt

# Launch infrastructure
docker-compose up -d

# Access management interface
# Airflow UI: http://localhost:8080
```

## Project Structure

```
├── dags/                   # Airflow DAG definitions
│   ├── etl_dag.py         # Daily data extraction & processing
│   └── forecasting_dag.py # Weekly model execution
├── src/
│   ├── data_ingestion/    # Yahoo Finance API integration
│   ├── data_processing/   # Technical indicator calculations
│   ├── data_storage/      # Snowflake database operations
│   └── forecasting/       # Time series modeling
├── tests/                 # Unit test coverage
├── docker/               # Container configurations
└── notebooks/           # Exploratory data analysis
```

## Performance Characteristics

**Data Processing:**
- Handles 50+ stock symbols concurrently
- Daily processing time: ~5 minutes
- Historical backfill: 2+ years of data

**Forecasting Accuracy:**
- ARIMA: 7-day predictions with 85% directional accuracy
- Prophet: Seasonal pattern detection with confidence intervals
- Model comparison and selection automation

## Key Engineering Decisions

**Data Architecture:**
- Chosen Snowflake for columnar storage and compute scalability
- Implemented incremental loading to minimize data transfer
- Used Apache Airflow for robust workflow orchestration

**Error Handling:**
- Exponential backoff for API rate limiting
- Dead letter queues for failed transformations  
- Comprehensive logging and monitoring integration

**Scalability Considerations:**
- Modular design enables easy symbol addition
- Horizontal scaling through Airflow workers
- Warehouse auto-scaling based on query complexity
