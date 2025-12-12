"""
API Configuration
Centralized settings for the FastAPI application
"""

import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = 'models/production/lightgbm_full_optimized.pkl'

# API configuration
API_TITLE = "Loan Default Prediction API"
API_DESCRIPTION = """
Predict loan default risk using machine learning with enhanced features.

## Features

* **Single Prediction**: Predict default probability for one application
* **Batch Prediction**: Process multiple applications at once
* **Model Information**: Get details about the current model
* **Health Check**: Verify API and model status

## Model Performance

* AUC-ROC: 0.79+ (optimized with 500 Optuna trials)
* Features: 361 (including aggregations from 7 data tables)
* Algorithm: LightGBM with Optuna-optimized hyperparameters
* Training: Full dataset with enhanced feature engineering
"""
API_VERSION = "2.0.0"

# Risk thresholds
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_HIGH = 0.6