"""
FastAPI Loan Default Prediction API
Main application file with Prometheus monitoring
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge, Info
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import json

from schemas import (
    LoanApplicationSimplified,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    RiskLevel,
    Recommendation
)
from config import (
    MODEL_PATH,
    METRICS_PATH,
    THRESHOLD_PATH,
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH
)
from feature_preparation import prepare_features_for_prediction, load_expected_feature_names

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= PROMETHEUS METRICS =============

# Initialize Prometheus instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Instrument the app
instrumentator.instrument(app)

# Custom business metrics
prediction_counter = Counter(
    'loan_predictions_total',
    'Total number of loan predictions made',
    ['prediction_type', 'risk_level', 'recommendation']
)

prediction_probability = Histogram(
    'loan_prediction_probability',
    'Distribution of default probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

prediction_duration = Histogram(
    'loan_prediction_duration_seconds',
    'Time spent processing prediction requests',
    ['prediction_type']
)

batch_size = Histogram(
    'loan_batch_prediction_size',
    'Number of applications in batch predictions',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

model_load_time = Gauge(
    'model_load_timestamp_seconds',
    'Timestamp when the model was loaded'
)

model_features_count = Gauge(
    'model_features_count',
    'Number of features in the loaded model'
)

model_info_metric = Info(
    'model_info',
    'Information about the ML model'
)

api_errors = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type']
)

# Global variable for model
model = None
model_loaded_at = None
expected_feature_names = None
model_metrics = None
optimal_threshold = 0.5


@app.on_event("startup")
async def load_model():
    """Load ML model when API starts"""
    global model, model_loaded_at, expected_feature_names, model_metrics, optimal_threshold
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        start_time = time.time()
        
        model = joblib.load(MODEL_PATH)
        load_duration = time.time() - start_time
        
        model_loaded_at = datetime.now().isoformat()
        print(f"Model loaded successfully at {model_loaded_at}")
        print(f"Model type: {type(model).__name__}")
        print(f"Model features: {model.n_features_}")
        print(f"Load time: {load_duration:.2f}s")
        
        # Load expected feature names for proper feature alignment
        expected_feature_names = load_expected_feature_names(MODEL_PATH)
        if expected_feature_names:
            print(f"Loaded {len(expected_feature_names)} expected feature names")
            if len(expected_feature_names) != model.n_features_:
                print(f"WARNING: Feature count mismatch! Expected {len(expected_feature_names)}, model has {model.n_features_}")
        else:
            print("WARNING: Could not load expected feature names. Using fallback feature preparation.")
        
        # Load model metrics
        try:
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, 'r') as f:
                    model_metrics = json.load(f)
                print(f"Loaded model metrics from: {METRICS_PATH}")
                print(f"  AUC-ROC: {model_metrics.get('auc_roc', 'N/A')}")
                print(f"  Precision: {model_metrics.get('precision', 'N/A')}")
                print(f"  Recall: {model_metrics.get('recall', 'N/A')}")
                print(f"  F1-Score: {model_metrics.get('f1_score', 'N/A')}")
            else:
                print(f"WARNING: Metrics file not found: {METRICS_PATH}")
                model_metrics = None
        except Exception as e:
            print(f"WARNING: Could not load metrics: {e}")
            model_metrics = None
        
        # Load optimal threshold
        try:
            if os.path.exists(THRESHOLD_PATH):
                with open(THRESHOLD_PATH, 'r') as f:
                    threshold_data = json.load(f)
                optimal_threshold = threshold_data.get('threshold', 0.5)
                print(f"Loaded optimal threshold: {optimal_threshold}")
            else:
                print(f"WARNING: Threshold file not found: {THRESHOLD_PATH}, using default 0.5")
                optimal_threshold = 0.5
        except Exception as e:
            print(f"WARNING: Could not load threshold: {e}, using default 0.5")
            optimal_threshold = 0.5
        
        # Update Prometheus metrics
        model_load_time.set(time.time())
        model_features_count.set(model.n_features_)
        model_info_metric.info({
            'model_type': 'LightGBM',
            'version': '1.0.0',
            'features': str(model.n_features_),
            'loaded_at': model_loaded_at
        })
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        model = None
        api_errors.labels(endpoint='startup', error_type='model_load_failed').inc()


@app.on_event("startup")
async def startup():
    """Expose Prometheus metrics endpoint"""
    instrumentator.expose(app, endpoint="/metrics")


@app.get("/", tags=["General"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "Loan Default Prediction API",
        "version": API_VERSION,
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        api_version=API_VERSION,
        model_loaded=model is not None,
        model_path=MODEL_PATH
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def get_model_info():
    """Get information about the current ML model"""
    if model is None:
        api_errors.labels(endpoint='model_info', error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Use loaded metrics if available, otherwise use defaults
    if model_metrics:
        performance_metrics = {
            "auc_roc": model_metrics.get('auc_roc', 0.0),
            "precision": model_metrics.get('precision', 0.0),
            "recall": model_metrics.get('recall', 0.0),
            "f1_score": model_metrics.get('f1_score', 0.0),
            "optimal_threshold": model_metrics.get('optimal_threshold', 0.5)
        }
        trained_at = model_metrics.get('trained_at', model_loaded_at)
    else:
        performance_metrics = {
            "auc_roc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "optimal_threshold": 0.5
        }
        trained_at = model_loaded_at
    
    return ModelInfo(
        model_type="LightGBM Classifier (SMOTE)",
        model_version="1.0.0",
        n_features=model.n_features_,
        performance_metrics=performance_metrics,
        last_trained=trained_at
    )


def prepare_features(application: LoanApplicationSimplified) -> pd.DataFrame:
    """
    Prepare features for model prediction matching the training pipeline.
    This now properly handles all features expected by the model.
    """
    # Convert Pydantic model to dictionary
    application_dict = {
        'AMT_INCOME_TOTAL': application.AMT_INCOME_TOTAL,
        'AMT_CREDIT': application.AMT_CREDIT,
        'AMT_ANNUITY': application.AMT_ANNUITY,
        'AMT_GOODS_PRICE': application.AMT_GOODS_PRICE,
        'DAYS_BIRTH': application.DAYS_BIRTH,
        'DAYS_EMPLOYED': application.DAYS_EMPLOYED,
        'EXT_SOURCE_1': application.EXT_SOURCE_1 if application.EXT_SOURCE_1 is not None else 0.5,
        'EXT_SOURCE_2': application.EXT_SOURCE_2 if application.EXT_SOURCE_2 is not None else 0.5,
        'EXT_SOURCE_3': application.EXT_SOURCE_3 if application.EXT_SOURCE_3 is not None else 0.5,
    }
    
    # Use the proper feature preparation function
    features_df = prepare_features_for_prediction(
        application_dict,
        expected_feature_names=expected_feature_names
    )
    
    return features_df


def calculate_risk(probability: float) -> tuple:
    """Calculate risk level and recommendation"""
    if probability < RISK_THRESHOLD_LOW:
        risk_level = RiskLevel.LOW
        recommendation = Recommendation.APPROVE
        message = f"Low risk of default ({probability:.1%}). Loan application approved."
    elif probability < RISK_THRESHOLD_HIGH:
        risk_level = RiskLevel.MEDIUM
        recommendation = Recommendation.REVIEW
        message = f"Medium risk of default ({probability:.1%}). Manual review recommended."
    else:
        risk_level = RiskLevel.HIGH
        recommendation = Recommendation.REJECT
        message = f"High risk of default ({probability:.1%}). Loan application rejected."
    
    return risk_level, recommendation, message


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplicationSimplified):
    """Predict default probability for a single loan application"""
    if model is None:
        api_errors.labels(endpoint='predict', error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = time.time()
    
    try:
        features_df = prepare_features(application)
        probability = float(model.predict_proba(features_df)[0][1])
        risk_level, recommendation, message = calculate_risk(probability)
        
        # Record metrics
        duration = time.time() - start_time
        prediction_duration.labels(prediction_type='single').observe(duration)
        prediction_probability.observe(probability)
        prediction_counter.labels(
            prediction_type='single',
            risk_level=risk_level.value,
            recommendation=recommendation.value
        ).inc()
        
        return PredictionResponse(
            success=True,
            default_probability=probability,
            risk_level=risk_level,
            recommendation=recommendation,
            message=message
        )
    
    except Exception as e:
        api_errors.labels(endpoint='predict', error_type='prediction_failed').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Predict default probability for multiple loan applications"""
    if model is None:
        api_errors.labels(endpoint='predict_batch', error_type='model_not_loaded').inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if len(request.applications) > 1000:
        api_errors.labels(endpoint='predict_batch', error_type='batch_too_large').inc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 applications per batch"
        )
    
    start_time = time.time()
    predictions = []
    
    try:
        batch_size.observe(len(request.applications))
        
        for app in request.applications:
            features_df = prepare_features(app)
            probability = float(model.predict_proba(features_df)[0][1])
            risk_level, recommendation, message = calculate_risk(probability)
            
            # Record metrics for each prediction
            prediction_probability.observe(probability)
            prediction_counter.labels(
                prediction_type='batch',
                risk_level=risk_level.value,
                recommendation=recommendation.value
            ).inc()
            
            predictions.append(PredictionResponse(
                success=True,
                default_probability=probability,
                risk_level=risk_level,
                recommendation=recommendation,
                message=message
            ))
        
        processing_time = time.time() - start_time
        prediction_duration.labels(prediction_type='batch').observe(processing_time)
        
        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_seconds=round(processing_time, 3)
        )
    
    except Exception as e:
        api_errors.labels(endpoint='predict_batch', error_type='prediction_failed').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )