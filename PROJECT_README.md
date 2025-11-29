# Energy Demand and Renewable Generation Forecasting

A modular Python-based time series forecasting pipeline for energy demand prediction using ARIMA, SARIMA, SARIMAX, and Facebook Prophet models.

## 📁 Project Structure

```
Energy_Demand_Forecasting/
├── config.py                    # Central configuration
├── run.py                       # Main execution script
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── outputs/                     # Generated plots and results
├── data/                        # Data directory
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading utilities
│   ├── preprocessing.py        # Data preprocessing
│   ├── eda.py                  # Exploratory analysis
│   ├── evaluation.py           # Model evaluation metrics
│   ├── visualization.py        # Plotting functions
│   └── models/
│       ├── __init__.py
│       ├── arima.py            # ARIMA implementation
│       ├── sarima.py           # SARIMA/SARIMAX implementation
│       └── prophet_model.py    # Prophet implementation
└── Notebooks/
    └── timeseries_data_forecasting_v2.ipynb
```

## 🚀 Features

- **Modular Architecture**: Function-based modules for easy testing and maintenance
- **Multiple Models**: ARIMA, SARIMA, SARIMAX, and Prophet
- **Automated Pipeline**: End-to-end workflow from data loading to evaluation
- **Interactive Visualizations**: Plotly-based interactive charts
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and R² scores
- **Configurable**: Centralized configuration for easy parameter tuning

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

For auto-ARIMA functionality (optional):
```bash
pip install pmdarima
```

## 🔧 Configuration

Edit `config.py` to customize:
- Data paths
- Target variable and exogenous features
- Train/test date ranges
- Model parameters (ARIMA orders, seasonality)
- Output settings

```python
from config import Config

config = Config()
config.data_path = "path/to/your/data.csv"
config.train_start = '2018-01-01'
config.train_end = '2019-12-31'
# ... customize other parameters
```

## 🏃 Usage

### Run Complete Pipeline

```bash
python run.py
```

This will:
1. Load and preprocess data
2. Optionally run EDA (you'll be prompted)
3. Train SARIMA, SARIMAX, and Prophet models
4. Generate forecasts
5. Evaluate models on validation data
6. Save plots and metrics to `outputs/`

### Use Individual Modules

```python
from config import Config
from src.data_loader import load_and_prepare_data
from src.models.sarima import train_sarima, forecast_sarima

# Load configuration
config = Config()

# Load data
df = load_and_prepare_data(config)

# Train model
model = train_sarima(
    df[config.target], 
    order=(2,1,2), 
    seasonal_order=(1,1,1,24)
)

# Generate forecast
forecast = forecast_sarima(model, steps=100)
```

## 📊 Models

### SARIMA
- Seasonal AutoRegressive Integrated Moving Average
- Captures trend and seasonality
- No external variables

### SARIMAX
- SARIMA with eXogenous variables
- Incorporates external factors (price, solar, wind generation)
- Better performance with relevant predictors

### Prophet
- Facebook's time series forecasting tool
- Handles holidays and multiple seasonality
- Configured with multiplicative seasonality for energy data

## 📈 Evaluation Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error): Average magnitude of errors
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **R²** (R-squared): Variance explained by model

## 📝 Outputs

The pipeline generates:
- Interactive HTML plots (`outputs/*.html`)
- Model metrics CSV (`outputs/model_metrics.csv`)
- Static plots (if enabled)

## 🛠️ Development

### Running Tests
```bash
# Add test framework if needed
pytest tests/
```

### Adding New Models

1. Create new module in `src/models/`
2. Implement `train_<model>()` and `forecast_<model>()` functions
3. Update `run.py` to include new model
4. Update `src/models/__init__.py`

Example:
```python
# src/models/lstm.py
def train_lstm(data, config):
    # Implementation
    pass

def forecast_lstm(model, steps):
    # Implementation
    pass
```

## 📖 Data Format

Expected CSV format:
```csv
utc_timestamp,AT_load_actual_entsoe_transparency,AT_price_day_ahead,...
2015-01-01 00:00:00,6284.75,52.38,...
2015-01-01 01:00:00,5997.0,50.91,...
```

Required columns (Austria example):
- `utc_timestamp`: Datetime index
- `AT_load_actual_entsoe_transparency`: Target variable (energy demand)
- `AT_price_day_ahead`: Day-ahead price (exogenous)
- `AT_solar_generation_actual`: Solar generation (exogenous)
- `AT_wind_onshore_generation_actual`: Wind generation (exogenous)

## 🔍 Troubleshooting

### Common Issues

**Import Error: No module named 'pmdarima'**
```bash
pip install pmdarima
```

**Prophet Installation Issues**
```bash
# Try conda if pip fails
conda install -c conda-forge prophet
```

**Memory Error with Large Datasets**
- Reduce `auto_arima_sample_fraction` in config
- Use smaller date ranges for training

## 📚 References

- Data Source: [Open Power System Data](https://data.open-power-system-data.org/time_series/)
- ARIMA: [statsmodels documentation](https://www.statsmodels.org/)
- Prophet: [Facebook Prophet](https://facebook.github.io/prophet/)

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: Update the `data_path` in `config.py` to point to your actual data file before running the pipeline.
