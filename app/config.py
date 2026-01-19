"""
API Configuration
Centralized settings for the FastAPI application
"""

import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = 'models/production/lightgbm_smote_high_performance.pkl'
METRICS_PATH = 'models/production/metrics_smote_high_performance.json'
THRESHOLD_PATH = 'models/production/optimal_threshold_smote.json'

# API configuration
API_TITLE = "Loan Default Prediction API"
API_DESCRIPTION = """
Predict loan default risk using machine learning.

## Features

* **Single Prediction**: Predict default probability for one application
* **Batch Prediction**: Process multiple applications at once
* **Model Information**: Get details about the current model
* **Health Check**: Verify API and model status

## Model Performance

* AUC-ROC: 91.6%
* Precision: 87.4%
* Recall: 92.17%
* F1-Score: 90.0%
* Algorithm: LightGBM with SMOTE oversampling and Optuna-optimized hyperparameters
"""
API_VERSION = "1.0.0"

# Risk thresholds
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_HIGH = 0.6