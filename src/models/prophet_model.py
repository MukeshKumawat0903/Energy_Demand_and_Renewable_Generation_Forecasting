"""
Prophet Model Module
Handles Facebook Prophet model training with enhanced configuration
"""

import pandas as pd
from prophet import Prophet
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def train_prophet(data: pd.Series, country: str = 'AT',
                  seasonality_mode: str = 'multiplicative',
                  yearly_seasonality: bool = True,
                  weekly_seasonality: bool = True,
                  daily_seasonality: bool = True) -> Prophet:
    """
    Train Prophet model with enhanced configuration
    
    Args:
        data: Training time series
        country: Country code for holiday calendar
        seasonality_mode: 'additive' or 'multiplicative'
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
        daily_seasonality: Include daily seasonality
        
    Returns:
        Fitted Prophet model
    """
    logger.info(f"Training Prophet model")
    logger.info(f"  Country: {country}")
    logger.info(f"  Seasonality mode: {seasonality_mode}")
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = data.reset_index()
    prophet_df.columns = ['ds', 'y']
    
    # Initialize Prophet model
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode=seasonality_mode
    )
    
    # Add country holidays
    m.add_country_holidays(country_name=country)
    logger.info(f"Added {country} holidays to model")
    
    # Fit model
    m.fit(prophet_df)
    
    logger.info("Prophet model trained successfully")
    
    return m


def forecast_prophet(model: Prophet, periods: int, freq: str = 'H') -> pd.DataFrame:
    """
    Generate forecasts from fitted Prophet model
    
    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast
        freq: Frequency of forecasts ('H' for hourly, 'D' for daily, etc.)
        
    Returns:
        DataFrame with forecast and components
    """
    logger.info(f"Generating {periods}-period forecast with frequency='{freq}'")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return forecast


def extract_forecast_values(forecast_df: pd.DataFrame, 
                           start_date: str, 
                           end_date: str) -> pd.Series:
    """
    Extract forecast values for a specific date range
    
    Args:
        forecast_df: Prophet forecast DataFrame
        start_date: Start date for extraction
        end_date: End date for extraction
        
    Returns:
        Series with forecast values
    """
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_indexed = forecast_df.set_index('ds')
    
    forecast_values = forecast_indexed[
        (forecast_indexed.index >= start_date) & 
        (forecast_indexed.index < end_date)
    ]['yhat']
    
    logger.info(f"Extracted {len(forecast_values)} forecast values for period {start_date} to {end_date}")
    
    return forecast_values
