"""
Visualization Module
Interactive and static plotting functions for forecasts and metrics
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
from typing import Optional, Dict
import logging
import webbrowser
import os

# Suppress plotly verbose output
pio.renderers.default = "browser"

logger = logging.getLogger(__name__)


def plot_time_series_comparison(train_data: Optional[pd.Series] = None,
                                test_data: Optional[pd.Series] = None,
                                forecast: Optional[pd.Series] = None,
                                model_name: str = "Forecast",
                                title: str = "Time Series Comparison",
                                actual_label: str = "Test Data",
                                train_label: str = "Train Data",
                                forecast_label: str = "Forecasted Data",
                                save_path: Optional[str] = None):
    """
    Interactive Plotly comparison plot for train, test, and forecast
    
    Args:
        train_data: Training data series
        test_data: Test data series
        forecast: Forecast series
        model_name: Name of the model
        title: Plot title
        actual_label: Label for actual data
        train_label: Label for training data
        forecast_label: Label for forecast data
        save_path: Optional path to save HTML file
    """
    fig = go.Figure()
    all_values = []
    
    # Add train data
    if train_data is not None:
        train_data = train_data.sort_index()
        all_values.extend(train_data.values)
        fig.add_trace(go.Scatter(
            x=train_data.index, y=train_data.values,
            mode='lines', name=train_label,
            line=dict(color='blue', width=2)
        ))
    
    # Add test data
    if test_data is not None:
        test_data = test_data.sort_index()
        all_values.extend(test_data.values)
        fig.add_trace(go.Scatter(
            x=test_data.index, y=test_data.values,
            mode='lines', name=actual_label,
            line=dict(color='green', width=2)
        ))
    
    # Add forecast
    if forecast is not None:
        forecast = forecast.sort_index()
        all_values.extend(forecast.values)
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            mode='lines', name=f"{model_name} {forecast_label}",
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Calculate y-axis range
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range_buffer = (y_max - y_min) * 0.1 if y_max != y_min else 1
        y_range = [y_min, y_max + y_range_buffer]
    else:
        y_range = None
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title='Datetime',
        yaxis_title='Value',
        yaxis=dict(range=y_range),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.5)',
            borderwidth=0
        ),
        template='plotly_white',
        hovermode='x unified',
        autosize=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Plot saved to: {save_path}")
        # Open in browser silently
        webbrowser.open('file://' + os.path.abspath(save_path), new=2)
    else:
        fig.show(config={'displaylogo': False})


def plot_multi_model_comparison(comparison_df: pd.DataFrame, 
                                title: str = "Multi-Model Forecast Comparison",
                                save_path: Optional[str] = None):
    """
    Plot multiple model forecasts on the same chart
    
    Args:
        comparison_df: DataFrame with Actual and model forecasts
        title: Plot title
        save_path: Optional path to save HTML file
    """
    fig = go.Figure()
    
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=comparison_df.index, 
        y=comparison_df['Actual'],
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Plot each model forecast
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    model_cols = [col for col in comparison_df.columns if col != 'Actual']
    
    for i, model in enumerate(model_cols):
        fig.add_trace(go.Scatter(
            x=comparison_df.index,
            y=comparison_df[model],
            name=f'{model} Forecast',
            line=dict(color=colors[i % len(colors)], width=2, dash='dot')
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Load (MW)',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Multi-model comparison saved to: {save_path}")
        # Open in browser silently
        webbrowser.open('file://' + os.path.abspath(save_path), new=2)
    else:
        fig.show(config={'displaylogo': False})


def plot_metrics_comparison(metrics_df: pd.DataFrame,
                           title: str = "Model Performance Metrics Comparison",
                           save_path: Optional[str] = None):
    """
    Create subplot visualization of model metrics
    
    Args:
        metrics_df: DataFrame with model metrics
        title: Plot title
        save_path: Optional path to save HTML file
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('MAE Comparison', 'RMSE Comparison', 
                       'MAPE Comparison', 'R² Comparison')
    )
    
    metric_cols = ['MAE', 'RMSE', 'MAPE', 'R²']
    
    for i, metric in enumerate(metric_cols):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(
            go.Bar(x=metrics_df.index, y=metrics_df[metric], name=metric,
                  showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_dark'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Metrics comparison saved to: {save_path}")
        # Open in browser silently
        webbrowser.open('file://' + os.path.abspath(save_path), new=2)
    else:
        fig.show(config={'displaylogo': False})


def plot_validation_comparison(validation_data: pd.DataFrame,
                               forecasts_dict: Dict,
                               target_col: str,
                               title: str = "Validation Period Forecast Comparison",
                               save_path: Optional[str] = None):
    """
    Plot actual vs forecasts for validation period
    
    Args:
        validation_data: Validation DataFrame
        forecasts_dict: Dictionary of model forecasts
        target_col: Target column name
        title: Plot title
        save_path: Optional path to save HTML file
    """
    fig = go.Figure()
    
    # Plot actual
    fig.add_trace(go.Scatter(
        x=validation_data.index,
        y=validation_data[target_col],
        name='Actual Load',
        line=dict(color='white', width=2)
    ))
    
    # Plot forecasts
    colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00']
    
    for i, (model_name, forecast) in enumerate(forecasts_dict.items()):
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=1.5)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Load (MW)',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Validation comparison saved to: {save_path}")
        # Open in browser silently
        webbrowser.open('file://' + os.path.abspath(save_path), new=2)
    else:
        fig.show(config={'displaylogo': False})
