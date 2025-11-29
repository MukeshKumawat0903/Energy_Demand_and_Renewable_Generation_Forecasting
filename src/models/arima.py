"""
ARIMA Model Module
Handles ARIMA model training and automatic parameter tuning
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def auto_tune_arima(data: pd.Series, seasonal: bool = True, m: int = 24,
                    sample_fraction: float = 0.25) -> Dict:
    """
    Automatically find optimal ARIMA parameters using pmdarima
    
    Args:
        data: Training time series
        seasonal: Whether to include seasonal components
        m: Seasonal period
        sample_fraction: Fraction of data to use for faster tuning
        
    Returns:
        Dictionary with best order, seasonal_order, and AIC
    """
    try:
        import pmdarima as pm
    except ImportError:
        logger.error("pmdarima not installed. Install with: pip install pmdarima")
        raise
    
    logger.info("Running Auto-ARIMA to find optimal parameters...")
    
    # Use subset for faster tuning
    sample_size = int(len(data) * sample_fraction)
    data_sample = data.iloc[:sample_size]
    
    logger.info(f"Using {sample_size} samples ({sample_fraction*100:.0f}%) for tuning")
    
    # Run auto_arima
    auto_model = pm.auto_arima(
        data_sample,
        seasonal=seasonal,
        m=m,
        d=1,  # First difference for stationarity
        D=1,  # Seasonal difference
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        random_state=42,
        n_fits=50
    )
    
    results = {
        'order': auto_model.order,
        'seasonal_order': auto_model.seasonal_order,
        'aic': auto_model.aic()
    }
    
    logger.info("=" * 60)
    logger.info("OPTIMAL ARIMA PARAMETERS FOUND:")
    logger.info("=" * 60)
    logger.info(f"Order (p,d,q): {results['order']}")
    logger.info(f"Seasonal Order (P,D,Q,m): {results['seasonal_order']}")
    logger.info(f"AIC: {results['aic']:.2f}")
    logger.info("=" * 60)
    
    return results


def train_arima(data: pd.Series, order: Tuple[int, int, int]):
    """
    Train ARIMA model
    
    Args:
        data: Training time series
        order: ARIMA order (p, d, q)
        
    Returns:
        Fitted ARIMA model
    """
    logger.info(f"Training ARIMA model with order={order}")
    
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    
    logger.info(f"ARIMA model trained successfully. AIC: {model_fit.aic:.2f}")
    
    return model_fit


def forecast_arima(model, steps: int) -> pd.Series:
    """
    Generate forecasts from fitted ARIMA model
    
    Args:
        model: Fitted ARIMA model
        steps: Number of steps to forecast
        
    Returns:
        Forecast series
    """
    logger.info(f"Generating {steps}-step forecast")
    
    forecast = model.forecast(steps=steps)
    
    return forecast
