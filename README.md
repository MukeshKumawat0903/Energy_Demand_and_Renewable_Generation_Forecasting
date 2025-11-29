# 📈 Energy Demand and Renewable Generation Forecasting

A production-ready time series forecasting pipeline for energy demand prediction using **SARIMA**, **SARIMAX**, and **Facebook Prophet** models. Features modular architecture, automated hyperparameter tuning, and interactive visualizations.

## 🌟 Key Features

- **Multiple Models**: SARIMA, SARIMAX (with exogenous variables), and Prophet
- **Auto-ARIMA**: Automated hyperparameter tuning using pmdarima
- **Exogenous Variables**: Price, solar generation, and wind generation data
- **Interactive Visualizations**: Plotly-based HTML plots with zoom, hover, and range sliders
- **Modular Architecture**: Function-based design for easy testing and maintenance
- **Comprehensive Metrics**: MAE, RMSE, MAPE, and R² scores
- **Automated Pipeline**: End-to-end execution from data loading to evaluation

---

## 📁 Dataset

**Source**: [Open Power System Data - Time Series](https://data.open-power-system-data.org/time_series/)

Hourly time-series data (2015-2021) for Austria energy market:
- **Target**: `AT_load_actual_entsoe_transparency` (Actual load in MW)
- **Exogenous Variables**:
  - `AT_price_day_ahead` (Day-ahead electricity price)
  - `AT_solar_generation_actual` (Solar generation)
  - `AT_wind_onshore_generation_actual` (Wind generation)

**Data File**: `time_series_60min_singleindex.csv`

---

## 🛠️ Project Structure

```
Energy_Demand_and_Renewable_Generation_Forecasting/
├── config.py                 # Centralized configuration
├── run.py                    # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── PROJECT_README.md         # Detailed documentation
├── QUICKSTART.md            # 5-minute setup guide
├── LICENSE
├── Notebooks/
│   └── timeseries_data_forecasting_v2.ipynb  # Jupyter notebook (exploratory)
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading & preparation
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   ├── eda.py              # Exploratory data analysis
│   ├── evaluation.py       # Metrics calculation
│   ├── visualization.py    # Interactive plotting
│   └── models/
│       ├── __init__.py
│       ├── arima.py        # ARIMA & Auto-ARIMA
│       ├── sarima.py       # SARIMA & SARIMAX
│       └── prophet_model.py # Facebook Prophet
└── outputs/                 # Generated plots & metrics
    ├── sarima_forecast.html
    ├── sarimax_forecast.html
    ├── prophet_forecast.html
    ├── multi_model_comparison.html
    ├── metrics_comparison.html
    ├── validation_comparison.html
    └── model_metrics.csv
```

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/MukeshKumawat0903/Energy_Demand_and_Renewable_Generation_Forecasting.git
cd Energy_Demand_and_Renewable_Generation_Forecasting
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `pandas>=2.2.2` - Data manipulation
- `numpy>=1.26.4` - Numerical computing
- `statsmodels>=0.14.2` - ARIMA/SARIMA models
- `prophet>=1.1.6` - Facebook Prophet
- `pmdarima>=1.8.0` - Auto-ARIMA tuning
- `plotly>=5.20.0` - Interactive visualizations
- `scikit-learn>=1.4.2` - Metrics

> **Note**: If Prophet installation fails with `pip`, use conda:
> ```bash
> conda install -c conda-forge prophet
> ```

---

## 🚀 Quick Start

### Option 1: Run Python Script (Recommended)

```bash
# 1. Update data path in config.py (line 16)
# 2. Run the pipeline
python run.py
```

**Output**: All plots saved to `outputs/` folder and opened in browser automatically.

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook Notebooks/timeseries_data_forecasting_v2.ipynb
```

---

## 🔍 Models Overview

### 1. **SARIMA** (Seasonal ARIMA)
- **Order**: (2, 1, 2) - AR, Differencing, MA
- **Seasonal Order**: (1, 1, 1, 24) - 24-hour seasonality
- **Best For**: Univariate time series with strong seasonality

### 2. **SARIMAX** (SARIMA with Exogenous Variables)
- Incorporates external factors: price, solar, wind generation
- **Best For**: When external predictors are available
- **Performance**: Typically outperforms SARIMA by 5-10%

### 3. **Prophet** (Facebook Prophet)
- **Features**: Multiplicative seasonality, Austria holidays
- **Best For**: Data with strong yearly/weekly patterns
- **Configuration**: `yearly_seasonality=True`, `seasonality_mode='multiplicative'`

### 4. **Auto-ARIMA** (Automated Tuning)
- Uses `pmdarima` for optimal parameter selection
- **Search Method**: Stepwise algorithm with AIC minimization
- **Speed**: 25% data sampling for faster tuning

---

## 📊 Evaluation Metrics

| Metric | Description | Lower = Better |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | ✅ |
| **RMSE** | Root Mean Squared Error | ✅ |
| **MAPE** | Mean Absolute Percentage Error | ✅ |
| **R²** | Coefficient of Determination | Higher = Better |

**Validation Period**: 2020-01-01 to 2020-12-31  
**Test Period**: 2020-01-01 to 2021-12-31

---

## 📈 Sample Results

**Typical Performance** (Validation Metrics):

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
|-------|----------|-----------|----------|-----|
| SARIMA | 864.7 | 1060.9 | 12.42 | 0.391 |
| SARIMAX | ~750-850 | ~950-1050 | ~10-12 | ~0.45-0.55 |
| Prophet | 1242.8 | 1423.8 | 19.90 | -0.097 |

> **Note**: Prophet's negative R² suggests poor fit for 2020 data (COVID-19 impact).

---

## 🎨 Visualizations

All plots are **interactive HTML files** with:
- Zoom & pan functionality
- Hover tooltips with exact values
- Range sliders for time navigation
- Legend toggling

**Generated Plots**:
1. **Individual Model Forecasts** - Train/test/forecast comparison
2. **Multi-Model Comparison** - All models on one chart
3. **Metrics Comparison** - Bar charts for MAE/RMSE/MAPE/R²
4. **Validation Comparison** - Model performance on validation period

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Data paths
data_path: str = "path/to/your/data.csv"
output_dir: str = "outputs"

# Train/test periods
train_start: str = '2018-01-01'
train_end: str = '2019-12-31'
test_start: str = '2020-01-01'
test_end: str = '2021-12-31'

# Model parameters
sarima_order: Tuple[int, int, int] = (2, 1, 2)
seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)

# Prophet settings
prophet_seasonality_mode: str = 'multiplicative'
prophet_country: str = 'AT'  # Austria holidays
```

---

## 🐛 Troubleshooting

### Issue: "File not found" error
**Solution**: Update `data_path` in `config.py` with correct file path

### Issue: Prophet installation fails
**Solution**: Use conda instead: `conda install -c conda-forge prophet`

### Issue: Memory error during Auto-ARIMA
**Solution**: Reduce `auto_arima_sample_fraction` in config.py (e.g., 0.1)

### Issue: Plots not opening
**Solution**: Check if default browser is configured, or open HTML files manually from `outputs/`

---

## 📚 Documentation

- **QUICKSTART.md** - 5-minute setup guide with examples
- **PROJECT_README.md** - Comprehensive technical documentation
- **Jupyter Notebook** - Step-by-step exploratory analysis

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add LSTM/XGBoost models
- [ ] Implement time series cross-validation
- [ ] Add CLI argument parsing
- [ ] Create unit tests
- [ ] Add model persistence (pickle/joblib)
- [ ] Build REST API wrapper

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 📧 Contact

**Mukesh Kumawat**  
GitHub: [@MukeshKumawat0903](https://github.com/MukeshKumawat0903)

---

## 🙏 Acknowledgments

- **Data Source**: [Open Power System Data](https://open-power-system-data.org/)
- **Prophet**: Facebook Research
- **pmdarima**: Automated ARIMA modeling

---

**⭐ If you find this project useful, please consider giving it a star!**
