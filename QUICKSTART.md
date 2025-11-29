# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Update Data Path

Edit `config.py` (line 16):
```python
data_path: str = r"YOUR_DATA_PATH_HERE/time_series_60min_singleindex.csv"
```

### 3. Run the Pipeline

```bash
python run.py
```

When prompted "Run EDA? (y/n):", type `n` for faster execution (or `y` to see exploratory plots).

### 4. Check Results

Results will be saved in the `outputs/` folder:
- `sarima_forecast.html` - SARIMA model visualization
- `sarimax_forecast.html` - SARIMAX model visualization  
- `prophet_forecast.html` - Prophet model visualization
- `multi_model_comparison.html` - All models on one chart
- `metrics_comparison.html` - Performance metrics
- `validation_comparison.html` - Validation period comparison
- `model_metrics.csv` - Metrics table

---

## 📊 Example Usage

### Basic Pipeline

```python
from config import Config
from src.data_loader import load_and_prepare_data
from src.preprocessing import preprocess_pipeline

# Initialize
config = Config()

# Load data
df = load_and_prepare_data(config)

# Preprocess
df, target_series = preprocess_pipeline(df, config)

print(f"Data loaded: {df.shape}")
print(f"Target series: {len(target_series)} samples")
```

### Train Single Model

```python
from src.models.sarima import train_sarima, forecast_sarima
from src.preprocessing import prepare_train_test_split

# Prepare data
train, test = prepare_train_test_split(df, config)

# Train SARIMA
model = train_sarima(
    train[config.target],
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24)
)

# Forecast
forecast = forecast_sarima(model, steps=len(test))

print(f"Forecast generated: {len(forecast)} steps")
```

### Evaluate Model

```python
from src.evaluation import calculate_metrics

# Calculate metrics
metrics = calculate_metrics(
    test[config.target],  # Actual values
    forecast,              # Predictions
    'SARIMA'              # Model name
)

print(metrics)
```

### Visualize Results

```python
from src.visualization import plot_time_series_comparison

plot_time_series_comparison(
    train_data=train[config.target],
    test_data=test[config.target],
    forecast=forecast,
    model_name='SARIMA',
    title='My SARIMA Forecast',
    save_path='outputs/my_forecast.html'
)
```

---

## 🔧 Configuration Examples

### Change Train/Test Periods

```python
# In config.py
train_start: str = '2017-01-01'
train_end: str = '2019-12-31'
test_start: str = '2020-01-01'
test_end: str = '2021-12-31'
```

### Modify Model Parameters

```python
# In config.py
sarima_order: Tuple[int, int, int] = (1, 1, 1)  # Simpler model
seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 168)  # Weekly seasonality
```

### Add/Remove Exogenous Variables

```python
# In config.py
exog_vars: List[str] = field(default_factory=lambda: [
    'AT_price_day_ahead',
    'AT_solar_generation_actual',
    # 'AT_wind_onshore_generation_actual',  # Commented out
    'AT_temperature'  # Added new variable
])
```

---

## 🐛 Troubleshooting

### Error: "File not found"
➡️ Update `data_path` in `config.py`

### Error: "No module named 'pmdarima'"
➡️ Run: `pip install pmdarima`

### Error: "Prophet installation failed"
➡️ Try: `conda install -c conda-forge prophet`

### Warning: "Could not infer frequency"
➡️ Check if your data has regular time intervals (hourly/daily)

### Memory Error
➡️ Reduce `auto_arima_sample_fraction` in config.py to 0.1 or 0.05

---

## 📁 Project Structure Explained

```
├── config.py           → All settings in one place
├── run.py              → Execute full pipeline
├── requirements.txt    → Dependencies
├── src/
│   ├── data_loader.py     → Load CSV files
│   ├── preprocessing.py   → Clean & prepare data
│   ├── eda.py            → Exploratory analysis
│   ├── evaluation.py     → Calculate metrics
│   ├── visualization.py  → Create plots
│   └── models/           → Model implementations
│       ├── arima.py
│       ├── sarima.py
│       └── prophet_model.py
└── outputs/           → Generated results
```

---

## 🎯 Next Steps

1. **Experiment with parameters**: Try different ARIMA orders in `config.py`
2. **Add more models**: Extend `src/models/` with LSTM, XGBoost, etc.
3. **Cross-validation**: Implement time series CV for robust evaluation
4. **Feature engineering**: Add lag features, rolling statistics
5. **Deployment**: Create Flask/FastAPI endpoint for real-time forecasts

---

## 💡 Tips

- Start with small date ranges for faster testing
- Use `n` for EDA prompt during development
- Check `outputs/model_metrics.csv` for quick comparison
- Interactive plots allow zooming and hovering for details
- All functions have logging - check console for progress

---

**Happy Forecasting! 📈**
