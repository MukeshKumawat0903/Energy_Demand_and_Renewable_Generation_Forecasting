"""
Preprocessing Module
Handles missing values, feature engineering, and data preparation
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, solar_col: str = 'AT_solar_generation_actual') -> pd.DataFrame:
    """
    Handle missing values in the dataset
    - Fill solar generation with 0 (nighttime = no generation)
    - Forward fill other columns
    
    Args:
        df: Input DataFrame
        solar_col: Solar generation column name
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    # Solar generation: missing values are nighttime (= 0)
    if solar_col in df.columns:
        missing_solar = df[solar_col].isnull().sum()
        df[solar_col] = df[solar_col].fillna(0)
        logger.info(f"Filled {missing_solar} missing values in {solar_col} with 0")
    
    # Forward fill other columns
    df = df.ffill()
    
    remaining_missing = df.isnull().sum().sum()
    logger.info(f"Remaining missing values after preprocessing: {remaining_missing}")
    
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime index
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    logger.info("Created time features: hour, day_of_week, month")
    
    return df


def interpolate_target(series: pd.Series, method: str = 'linear') -> pd.Series:
    """
    Interpolate missing values in target series
    
    Args:
        series: Target time series
        method: Interpolation method
        
    Returns:
        Interpolated series
    """
    series = series.copy()
    missing_before = series.isnull().sum()
    
    series = series.interpolate(method=method)
    series = series.dropna()
    
    missing_after = series.isnull().sum()
    logger.info(f"Interpolated target: {missing_before} → {missing_after} missing values")
    
    return series


def prepare_train_test_split(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on date ranges
    
    Args:
        df: Input DataFrame
        config: Configuration object with date ranges
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train = df[config.train_start:config.train_end]
    test = df[config.test_start:config.test_end]
    
    logger.info(f"Train set: {len(train)} samples ({train.index[0]} to {train.index[-1]})")
    logger.info(f"Test set: {len(test)} samples ({test.index[0]} to {test.index[-1]})")
    
    return train, test


def preprocess_pipeline(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline
    
    Args:
        df: Raw DataFrame
        config: Configuration object
        
    Returns:
        Tuple of (processed_df, target_series)
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Engineer time features
    df = engineer_time_features(df)
    
    # Extract and interpolate target
    target_series = df[config.target]
    target_series = interpolate_target(target_series)
    
    logger.info("Preprocessing pipeline completed")
    
    return df, target_series
