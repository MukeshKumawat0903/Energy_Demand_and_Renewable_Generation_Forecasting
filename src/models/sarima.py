"""
SARIMA/SARIMAX Model Module
Handles SARIMA and SARIMAX model training with optional exogenous variables
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def train_sarima(data: pd.Series, order: Tuple[int, int, int], 
                 seasonal_order: Tuple[int, int, int, int]):
    """
    Train SARIMA model without exogenous variables
    
    Args:
        data: Training time series
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m)
        
    Returns:
        Fitted SARIMAX model
    """
    logger.info(f"Training SARIMA model")
    logger.info(f"  Order: {order}")
    logger.info(f"  Seasonal Order: {seasonal_order}")
    
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    logger.info(f"SARIMA model trained successfully. AIC: {model_fit.aic:.2f}")
    
    return model_fit


def train_sarimax(data: pd.Series, exog: pd.DataFrame, 
                  order: Tuple[int, int, int],
                  seasonal_order: Tuple[int, int, int, int]):
    """
    Train SARIMAX model with exogenous variables
    
    Args:
        data: Training time series
        exog: Exogenous variables DataFrame
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, m)
        
    Returns:
        Fitted SARIMAX model
    """
    logger.info(f"Training SARIMAX model with exogenous variables")
    logger.info(f"  Order: {order}")
    logger.info(f"  Seasonal Order: {seasonal_order}")
    logger.info(f"  Exogenous variables: {list(exog.columns)}")
    
    model = SARIMAX(data, exog=exog, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    logger.info(f"SARIMAX model trained successfully. AIC: {model_fit.aic:.2f}")
    
    return model_fit


def forecast_sarima(model, steps: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Generate forecasts from fitted SARIMA/SARIMAX model
    
    Args:
        model: Fitted SARIMAX model
        steps: Number of steps to forecast
        exog: Exogenous variables for forecast period (required for SARIMAX)
        
    Returns:
        Forecast series
    """
    logger.info(f"Generating {steps}-step forecast")
    
    if exog is not None:
        logger.info(f"Using exogenous variables for forecast")
        forecast = model.get_forecast(steps=steps, exog=exog).predicted_mean
    else:
        forecast = model.get_forecast(steps=steps).predicted_mean
    
    return forecast
