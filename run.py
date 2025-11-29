"""
Main Runner Script
Orchestrates the complete forecasting pipeline
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.data_loader import load_and_prepare_data
from src.preprocessing import preprocess_pipeline, prepare_train_test_split
from src.eda import run_eda_pipeline
from src.models.arima import train_arima, forecast_arima
from src.models.sarima import train_sarima, train_sarimax, forecast_sarima
from src.models.prophet_model import train_prophet, forecast_prophet, extract_forecast_values
from src.evaluation import evaluate_models, calculate_metrics, create_metrics_dataframe
from src.visualization import (
    plot_time_series_comparison, 
    plot_multi_model_comparison,
    plot_metrics_comparison,
    plot_validation_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution pipeline"""
    
    logger.info("=" * 60)
    logger.info("ENERGY DEMAND AND RENEWABLE GENERATION FORECASTING")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config()
    
    # 1. LOAD DATA
    logger.info("\n>>> STEP 1: Loading Data")
    df = load_and_prepare_data(config)
    
    # 2. PREPROCESS DATA
    logger.info("\n>>> STEP 2: Preprocessing Data")
    df, target_series = preprocess_pipeline(df, config)
    
    # 3. EXPLORATORY DATA ANALYSIS (Optional - skipped for faster execution)
    logger.info("\n>>> STEP 3: Exploratory Data Analysis (Skipped)")
    logger.info("To run EDA, set run_eda=True in the code")
    run_eda = True  # Set to True if you want EDA plots
    
    if run_eda:
        eda_results = run_eda_pipeline(df, target_series, config, save_plots=True)
    
    # 4. PREPARE TRAIN/TEST SPLIT
    logger.info("\n>>> STEP 4: Preparing Train/Test Split")
    train, test = prepare_train_test_split(df, config)
    
    # 5. TRAIN MODELS
    logger.info("\n>>> STEP 5: Training Models")
    models = {}
    forecasts = {}
    
    # SARIMA
    logger.info("\n--- Training SARIMA ---")
    sarima_model = train_sarima(
        train[config.target], 
        config.sarima_order, 
        config.seasonal_order
    )
    models['SARIMA'] = sarima_model
    forecasts['SARIMA'] = forecast_sarima(sarima_model, len(test))
    
    # SARIMAX
    logger.info("\n--- Training SARIMAX ---")
    sarimax_model = train_sarimax(
        train[config.target],
        train[config.exog_vars],
        config.sarima_order,
        config.seasonal_order
    )
    models['SARIMAX'] = sarimax_model
    forecasts['SARIMAX'] = forecast_sarima(sarimax_model, len(test), test[config.exog_vars])
    
    # Prophet
    logger.info("\n--- Training Prophet ---")
    prophet_model = train_prophet(
        train[config.target],
        country=config.prophet_country,
        seasonality_mode=config.prophet_seasonality_mode
    )
    models['Prophet'] = prophet_model
    prophet_forecast_full = forecast_prophet(prophet_model, len(test), freq='H')
    forecasts['Prophet'] = extract_forecast_values(
        prophet_forecast_full,
        config.test_start,
        config.test_end
    )
    
    # 6. VISUALIZE FORECASTS
    logger.info("\n>>> STEP 6: Visualizing Forecasts")
    
    for model_name, forecast in forecasts.items():
        plot_time_series_comparison(
            train_data=train[config.target],
            test_data=test[config.target],
            forecast=forecast,
            model_name=model_name,
            title=f'{model_name} Model Performance',
            save_path=f"{config.output_dir}/{model_name.lower()}_forecast.html" if config.plot_save else None
        )
    
    # Multi-model comparison
    comparison_df = test[config.target].to_frame(name='Actual')
    for model_name, forecast in forecasts.items():
        comparison_df[model_name] = forecast
    
    plot_multi_model_comparison(
        comparison_df,
        save_path=f"{config.output_dir}/multi_model_comparison.html" if config.plot_save else None
    )
    
    # 7. MODEL EVALUATION
    logger.info("\n>>> STEP 7: Model Evaluation")
    validation = df[config.val_start:config.val_end]
    
    # Generate validation forecasts
    val_forecasts = {}
    val_forecasts['SARIMA'] = forecast_sarima(sarima_model, len(validation))
    val_forecasts['SARIMAX'] = forecast_sarima(sarimax_model, len(validation), validation[config.exog_vars])
    
    prophet_val_forecast = forecast_prophet(prophet_model, len(validation), freq='H')
    val_forecasts['Prophet'] = extract_forecast_values(
        prophet_val_forecast,
        config.val_start,
        config.val_end
    )
    
    # Calculate metrics
    metrics_list = []
    for model_name, val_forecast in val_forecasts.items():
        metrics = calculate_metrics(
            validation[config.target],
            val_forecast,
            model_name
        )
        metrics_list.append(metrics)
    
    metrics_df = create_metrics_dataframe(metrics_list)
    
    # Plot metrics
    plot_metrics_comparison(
        metrics_df,
        save_path=f"{config.output_dir}/metrics_comparison.html" if config.plot_save else None
    )
    
    # Plot validation comparison
    plot_validation_comparison(
        validation,
        val_forecasts,
        config.target,
        save_path=f"{config.output_dir}/validation_comparison.html" if config.plot_save else None
    )
    
    # 8. SAVE RESULTS
    logger.info("\n>>> STEP 8: Saving Results")
    metrics_df.to_csv(f"{config.output_dir}/model_metrics.csv")
    logger.info(f"Metrics saved to {config.output_dir}/model_metrics.csv")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    
    return models, forecasts, metrics_df


if __name__ == "__main__":
    try:
        models, forecasts, metrics = main()
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
