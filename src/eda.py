"""
Exploratory Data Analysis Module
Statistical tests, decomposition, and diagnostic plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def test_stationarity(series: pd.Series) -> Dict[str, float]:
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Args:
        series: Time series to test
        
    Returns:
        Dictionary with test results
    """
    dftest = adfuller(series, autolag='AIC')
    
    results = {
        'adf_statistic': dftest[0],
        'p_value': dftest[1],
        'used_lag': dftest[2],
        'n_observations': dftest[3]
    }
    
    logger.info(f"ADF Statistic: {results['adf_statistic']:.6f}")
    logger.info(f"p-value: {results['p_value']:.6f}")
    
    if results['p_value'] < 0.05:
        logger.info("Series is stationary (reject null hypothesis)")
    else:
        logger.warning("Series is non-stationary (fail to reject null hypothesis)")
    
    return results


def plot_seasonal_decomposition(series: pd.Series, period: int, 
                                save_path: Optional[str] = None) -> None:
    """
    Decompose time series into trend, seasonal, and residual components
    
    Args:
        series: Time series to decompose
        period: Seasonality period
        save_path: Optional path to save the plot
    """
    logger.info(f"Decomposing series with period={period}")
    
    result = seasonal_decompose(series, period=period)
    
    fig = result.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Decomposition plot saved to: {save_path}")
    
    # Non-blocking display
    plt.show(block=False)
    plt.pause(0.1)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap for DataFrame
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to: {save_path}")
    
    # Non-blocking display
    plt.show(block=False)
    plt.pause(0.1)


def plot_acf_pacf(series: pd.Series, lags: int = 48, 
                  save_path: Optional[str] = None) -> None:
    """
    Plot ACF and PACF for time series
    
    Args:
        series: Time series
        lags: Number of lags to display
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ACF/PACF plot saved to: {save_path}")
    
    # Non-blocking display
    plt.show(block=False)
    plt.pause(0.1)


def plot_time_series(series: pd.Series, title: str = 'Time Series', 
                     ylabel: str = 'Value', save_path: Optional[str] = None) -> None:
    """
    Plot time series
    
    Args:
        series: Time series to plot
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(14, 6))
    series.plot()
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Time series plot saved to: {save_path}")
    
    # Non-blocking display
    plt.show(block=False)
    plt.pause(0.1)


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for DataFrame
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with summary statistics
    """
    summary = df.describe()
    logger.info("Summary statistics generated")
    
    return summary


def run_eda_pipeline(df: pd.DataFrame, target_series: pd.Series, config, 
                     save_plots: bool = False) -> Dict:
    """
    Run complete EDA pipeline
    
    Args:
        df: Full DataFrame
        target_series: Target time series
        config: Configuration object
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with EDA results
    """
    results = {}
    
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)
    
    # Summary statistics
    results['summary_stats'] = generate_summary_statistics(df)
    
    # Stationarity test
    results['stationarity'] = test_stationarity(target_series)
    
    # Plots (optional save)
    output_dir = config.output_dir if save_plots else None
    
    plot_time_series(target_series, 
                    title='Actual Load Over Time',
                    ylabel='MW',
                    save_path=f"{output_dir}/time_series.png" if output_dir else None)
    
    plot_seasonal_decomposition(target_series, 
                               period=config.seasonality_period,
                               save_path=f"{output_dir}/decomposition.png" if output_dir else None)
    
    plot_correlation_heatmap(df,
                            save_path=f"{output_dir}/correlation.png" if output_dir else None)
    
    plot_acf_pacf(target_series,
                 save_path=f"{output_dir}/acf_pacf.png" if output_dir else None)
    
    logger.info("EDA pipeline completed")
    
    return results
