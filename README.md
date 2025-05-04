# 📈 Energy Demand and Renewable Generation Forecasting

This project provides a comprehensive comparison of time series forecasting techniques — including **ARIMA**, **SARIMA** and **Facebook Prophet** — using real-world energy data. It includes:

* Data preprocessing
* Forecasting with ARIMA, SARIMA and Prophet
* Visualization of predictions
* Model evaluation and comparison

---

### 📁 Dataset

The dataset used contains hourly time-series data related to energy metrics such as load and renewable generation. The file used:

```
time_series_60min_singleindex_filtered.csv
```

It contains a `datetime` column and several numeric columns like:

* `DE_load_actual_entsoe_transparency` (Germany load)
* `DE_solar_generation_actual`
* `DE_wind_generation_actual`
* And more (per-country data available)

Data Source: [opsd-time_series-2020-10-06](https://data.open-power-system-data.org/time_series/)

---

### 🛠️ Project Structure

```
forecast_comparison/
├── forecast_comparison.ipynb  # Jupyter notebook with full analysis
├── time_series_60min_singleindex_filtered.csv  # Input dataset
└── README.md                  # Project documentation
```

---

### 📦 Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Minimal requirements:

```txt
pandas
numpy
matplotlib
seaborn
statsmodels
prophet
scikit-learn
```

> If Prophet fails to install via `pip`, try:
>
> ```bash
> conda install -c conda-forge prophet
> ```

---

### 🔍 Forecasting Models

* **SARIMA**: Captures both trend and seasonality using seasonal autoregressive modeling.
* **Prophet**: Decomposable time series model from Facebook that handles holidays, trends, and seasonality.

---

### 📊 Evaluation Metrics

Each model is evaluated using:

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Squared Error)

Plots and error metrics help visually and quantitatively compare forecast performance.

---

### 🚀 How to Run

1. Place the CSV file in the same directory.
2. Open the `forecast_comparison.ipynb` notebook.
3. Update the source datapath, can download from provided [link](https://data.open-power-system-data.org/time_series/).
4. Follow the cells in order to process, model, and compare the forecasts.

---
