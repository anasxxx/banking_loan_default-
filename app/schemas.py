"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Recommendation(str, Enum):
    """Loan decision recommendations"""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


class LoanApplicationSimplified(BaseModel):
    """
    Simplified loan application for demo purposes
    In production, you'd include all 135 features
    """
    AMT_INCOME_TOTAL: float = Field(..., description="Total income", gt=0)
    AMT_CREDIT: float = Field(..., description="Credit amount", gt=0)
    AMT_ANNUITY: float = Field(..., description="Loan annuity", gt=0)
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods", gt=0)
    DAYS_BIRTH: int = Field(..., description="Days since birth (negative)", lt=0)
    DAYS_EMPLOYED: int = Field(..., description="Days employed (negative)")
    EXT_SOURCE_1: Optional[float] = Field(None, description="External source 1", ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, description="External source 2", ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, description="External source 3", ge=0, le=1)
    
    class Config:
        schema_extra = {
            "example": {
                "AMT_INCOME_TOTAL": 150000.0,
                "AMT_CREDIT": 500000.0,
                "AMT_ANNUITY": 25000.0,
                "AMT_GOODS_PRICE": 450000.0,
                "DAYS_BIRTH": -15000,
                "DAYS_EMPLOYED": -3000,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.7
            }
        }
    
    @validator('DAYS_BIRTH')
    def validate_age(cls, v):
        """Ensure reasonable age (18-100 years)"""
        age_years = abs(v) / 365
        if age_years < 18 or age_years > 100:
            raise ValueError('Age must be between 18 and 100 years')
        return v
    
    @validator('AMT_CREDIT')
    def validate_credit(cls, v, values):
        """Ensure credit amount is reasonable relative to income"""
        if 'AMT_INCOME_TOTAL' in values and v > values['AMT_INCOME_TOTAL'] * 10:
            raise ValueError('Credit amount too high relative to income')
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    success: bool = Field(..., description="Whether prediction succeeded")
    default_probability: float = Field(..., description="Probability of default (0-1)", ge=0, le=1)
    risk_level: RiskLevel = Field(..., description="Risk category")
    recommendation: Recommendation = Field(..., description="Loan decision recommendation")
    message: str = Field(..., description="Human-readable explanation")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    applications: List[LoanApplicationSimplified] = Field(..., description="List of loan applications")
    
    class Config:
        schema_extra = {
            "example": {
                "applications": [
                    {
                        "AMT_INCOME_TOTAL": 150000.0,
                        "AMT_CREDIT": 500000.0,
                        "AMT_ANNUITY": 25000.0,
                        "AMT_GOODS_PRICE": 450000.0,
                        "DAYS_BIRTH": -15000,
                        "DAYS_EMPLOYED": -3000,
                        "EXT_SOURCE_1": 0.5,
                        "EXT_SOURCE_2": 0.6,
                        "EXT_SOURCE_3": 0.7
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    n_features: int
    performance_metrics: dict
    last_trained: Optional[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    api_version: str
    model_loaded: bool
    model_path: str