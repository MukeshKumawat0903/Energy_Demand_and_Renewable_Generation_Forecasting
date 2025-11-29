"""
Data Loading Module
Handles CSV loading, datetime parsing, timezone handling, and frequency inference
"""

import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(file_path: str, timestamp_col: str = "utc_timestamp") -> pd.DataFrame:
    """
    Load CSV dataset with timestamp parsing and indexing
    
    Args:
        file_path: Path to the CSV file
        timestamp_col: Name of the timestamp column
        
    Returns:
        DataFrame with timestamp index
    """
    logger.info(f"Loading dataset from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, parse_dates=[timestamp_col])
        df = df.set_index(timestamp_col)
        df.index = df.index.tz_localize(None)  # Remove timezone
        
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def filter_country_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Filter DataFrame columns by country prefix
    
    Args:
        df: Input DataFrame
        prefix: Country prefix (e.g., 'AT_')
        
    Returns:
        DataFrame with filtered columns
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    logger.info(f"Filtered {len(cols)} columns with prefix '{prefix}'")
    
    return df[cols]


def infer_and_set_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and set time series frequency
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        DataFrame with frequency set
    """
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq:
        df.index.freq = inferred_freq
        logger.info(f"Inferred frequency: {df.index.freq}")
    else:
        logger.warning("Could not infer frequency. Setting manually may be required.")
    
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        Series with missing value counts per column
    """
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    logger.info(f"Total missing values: {total_missing}")
    
    if total_missing > 0:
        logger.warning(f"Columns with missing values:\n{missing[missing > 0]}")
    
    return missing


def load_and_prepare_data(config) -> pd.DataFrame:
    """
    Complete data loading pipeline
    
    Args:
        config: Configuration object
        
    Returns:
        Prepared DataFrame
    """
    # Load dataset
    df = load_dataset(config.data_path)
    
    # Filter country columns
    df = filter_country_columns(df, config.country_prefix)
    
    # Infer frequency
    df = infer_and_set_frequency(df)
    
    # Check missing values
    check_missing_values(df)
    
    return df
