"""
FastAPI Loan Default Prediction API
Main application file
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

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
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH
)

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

# Global variable for model
model = None
model_loaded_at = None


@app.on_event("startup")
async def load_model():
    """Load ML model when API starts"""
    global model, model_loaded_at
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        model_loaded_at = datetime.now().isoformat()
        print(f"Model loaded successfully at {model_loaded_at}")
        print(f"Model type: {type(model).__name__}")
        print(f"Model features: {model.n_features_}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        model = None


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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_type="LightGBM Classifier",
        model_version="1.0.0",
        n_features=model.n_features_,
        performance_metrics={
            "auc_roc": 0.7712,
            "recall": 0.6359,
            "precision": 0.1907,
            "f1_score": 0.2934
        },
        last_trained=model_loaded_at
    )


def prepare_features(application: LoanApplicationSimplified) -> pd.DataFrame:
    """
    Prepare features for model prediction
    NOTE: Simplified for demo - production needs all 135 features
    """
    features = {
        'AMT_INCOME_TOTAL': application.AMT_INCOME_TOTAL,
        'AMT_CREDIT': application.AMT_CREDIT,
        'AMT_ANNUITY': application.AMT_ANNUITY,
        'AMT_GOODS_PRICE': application.AMT_GOODS_PRICE,
        'DAYS_BIRTH': application.DAYS_BIRTH,
        'DAYS_EMPLOYED': application.DAYS_EMPLOYED,
        'EXT_SOURCE_1': application.EXT_SOURCE_1 if application.EXT_SOURCE_1 else 0.5,
        'EXT_SOURCE_2': application.EXT_SOURCE_2 if application.EXT_SOURCE_2 else 0.5,
        'EXT_SOURCE_3': application.EXT_SOURCE_3 if application.EXT_SOURCE_3 else 0.5,
    }
    
    # Calculate engineered features
    features['AGE_YEARS'] = abs(application.DAYS_BIRTH) / 365
    features['EMPLOYMENT_YEARS'] = abs(application.DAYS_EMPLOYED) / 365 if application.DAYS_EMPLOYED < 0 else 0
    features['CREDIT_INCOME_RATIO'] = application.AMT_CREDIT / application.AMT_INCOME_TOTAL
    features['ANNUITY_INCOME_RATIO'] = application.AMT_ANNUITY / application.AMT_INCOME_TOTAL
    features['LOAN_TERM_MONTHS'] = application.AMT_CREDIT / application.AMT_ANNUITY if application.AMT_ANNUITY > 0 else 0
    
    # Pad with zeros for remaining features (demo only - DON'T do this in production!)
    for i in range(model.n_features_ - len(features)):
        features[f'feature_{i}'] = 0
    
    df = pd.DataFrame([features])
    return df.iloc[:, :model.n_features_]


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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        features_df = prepare_features(application)
        probability = float(model.predict_proba(features_df)[0][1])
        risk_level, recommendation, message = calculate_risk(probability)
        
        return PredictionResponse(
            success=True,
            default_probability=probability,
            risk_level=risk_level,
            recommendation=recommendation,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Predict default probability for multiple loan applications"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if len(request.applications) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 applications per batch"
        )
    
    start_time = time.time()
    predictions = []
    
    try:
        for app in request.applications:
            features_df = prepare_features(app)
            probability = float(model.predict_proba(features_df)[0][1])
            risk_level, recommendation, message = calculate_risk(probability)
            
            predictions.append(PredictionResponse(
                success=True,
                default_probability=probability,
                risk_level=risk_level,
                recommendation=recommendation,
                message=message
            ))
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_seconds=round(processing_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )