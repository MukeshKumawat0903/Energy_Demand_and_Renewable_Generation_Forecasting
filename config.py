"""
Configuration Module for Energy Demand Forecasting
Centralized configuration for paths, parameters, and constants
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class Config:
    """Configuration dataclass for the forecasting pipeline"""
    
    # Data paths
    data_path: str = r"D:\Learnings\My_Projects\Master_Projects\Energy Demand & Renewable Generation Forecasting\data\row\opsd-time_series-2020-10-06\time_series_60min_singleindex.csv"
    output_dir: str = "outputs"
    
    # Target & features
    target: str = 'AT_load_actual_entsoe_transparency'
    exog_vars: List[str] = field(default_factory=lambda: [
        'AT_price_day_ahead', 
        'AT_solar_generation_actual', 
        'AT_wind_onshore_generation_actual'
    ])
    country_prefix: str = 'AT_'
    
    # Time periods
    train_start: str = '2018-01-01'
    train_end: str = '2019-12-31'
    test_start: str = '2020-01-01'
    test_end: str = '2021-12-31'
    
    # Validation period
    val_start: str = '2020-01-01'
    val_end: str = '2020-12-31'
    
    # Model parameters
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    sarima_order: Tuple[int, int, int] = (2, 1, 2)
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)
    seasonality_period: int = 24 * 7  # Weekly for decomposition
    
    # Auto-ARIMA parameters
    auto_arima_seasonal: bool = True
    auto_arima_m: int = 24  # Daily seasonality for hourly data
    auto_arima_sample_fraction: float = 0.25  # Use 25% of data for speed
    
    # Prophet parameters
    prophet_seasonality_mode: str = 'multiplicative'
    prophet_country: str = 'AT'
    
    # Visualization settings
    plot_save: bool = True # Set to False to disable saving
    plot_format: str = 'html'
    
    def __post_init__(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)


# Create default config instance
default_config = Config()
