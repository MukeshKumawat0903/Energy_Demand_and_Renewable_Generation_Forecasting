"""
Evaluation Module
Model performance metrics and validation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> Dict:
    """
    Calculate evaluation metrics for model predictions
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        
    Returns:
        Dictionary with metric values
    """
    metrics = {
        'Model': model_name,
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'MAPE': round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        'R²': round(r2_score(y_true, y_pred), 3)
    }
    
    logger.info(f"Metrics for {model_name}:")
    logger.info(f"  MAE: {metrics['MAE']} MW")
    logger.info(f"  RMSE: {metrics['RMSE']} MW")
    logger.info(f"  MAPE: {metrics['MAPE']}%")
    logger.info(f"  R²: {metrics['R²']}")
    
    return metrics


def create_metrics_dataframe(metrics_list: List[Dict]) -> pd.DataFrame:
    """
    Create DataFrame from list of metric dictionaries
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        DataFrame with metrics indexed by model name
    """
    metrics_df = pd.DataFrame(metrics_list).set_index('Model')
    
    logger.info("Metrics DataFrame created")
    logger.info(f"\n{metrics_df}")
    
    return metrics_df


def generate_forecast_error_estimates(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate forecast error estimates from metrics
    
    Args:
        metrics_df: DataFrame with model metrics
        
    Returns:
        Styled DataFrame with error estimates
    """
    forecast_metrics = {}
    
    for index, row in metrics_df.iterrows():
        model_name = row.name if hasattr(row, 'name') else index
        forecast_metrics[model_name] = {
            'MAE': f"±{round(row['MAE'])} MW",
            'MAPE': f"{round(row['MAPE'], 1)}%"
        }
    
    df_forecast = pd.DataFrame(forecast_metrics).T
    
    logger.info("Forecast error estimates generated")
    
    return df_forecast


def compare_models(metrics_df: pd.DataFrame) -> str:
    """
    Generate comparison report for models
    
    Args:
        metrics_df: DataFrame with model metrics
        
    Returns:
        String with comparison analysis
    """
    best_mae = metrics_df['MAE'].idxmin()
    best_rmse = metrics_df['RMSE'].idxmin()
    best_mape = metrics_df['MAPE'].idxmin()
    best_r2 = metrics_df['R²'].idxmax()
    
    report = f"""
Model Comparison Summary:
{'=' * 60}
Best MAE:   {best_mae} ({metrics_df.loc[best_mae, 'MAE']:.2f} MW)
Best RMSE:  {best_rmse} ({metrics_df.loc[best_rmse, 'RMSE']:.2f} MW)
Best MAPE:  {best_mape} ({metrics_df.loc[best_mape, 'MAPE']:.2f}%)
Best R²:    {best_r2} ({metrics_df.loc[best_r2, 'R²']:.3f})
{'=' * 60}
"""
    
    logger.info(report)
    
    return report


def evaluate_models(models_dict: Dict, validation_data: pd.DataFrame, 
                   forecasts_dict: Dict, config) -> pd.DataFrame:
    """
    Evaluate multiple models on validation data
    
    Args:
        models_dict: Dictionary of fitted models
        validation_data: Validation DataFrame
        forecasts_dict: Dictionary of model forecasts
        config: Configuration object
        
    Returns:
        DataFrame with evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    metrics_list = []
    
    for model_name, forecast in forecasts_dict.items():
        metrics = calculate_metrics(
            validation_data[config.target],
            forecast,
            model_name
        )
        metrics_list.append(metrics)
    
    metrics_df = create_metrics_dataframe(metrics_list)
    
    # Generate comparison
    compare_models(metrics_df)
    
    return metrics_df
