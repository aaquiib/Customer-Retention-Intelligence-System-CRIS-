"""Pydantic models for API request/response validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# PREDICTION REQUEST/RESPONSE
# ─────────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """Single customer features for prediction."""
    
    # Demographic
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    
    # Tenure & Charges
    tenure: Optional[int] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    
    # Services
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    
    # Contract & Billing
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "MonthlyCharges": 65.50,
                "TotalCharges": 1570.70,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


class PredictionOutput(BaseModel):
    """Single customer prediction result."""
    
    segment: int = Field(..., ge=0, le=3, description="Segment ID (0-3)")
    segment_label: str = Field(..., description="Human-readable segment name")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Churn probability (0.0-1.0)")
    is_churner: bool = Field(..., description="Whether churn_probability exceeds threshold")
    threshold: float = Field(..., description="Churn decision threshold used")
    segment_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of segment assignment")
    input_features: Dict[str, Any] = Field(..., description="Echo of input features after preprocessing")
    engineered_features: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    """Request for single prediction endpoint."""
    
    customer: CustomerInput
    return_features: bool = Field(default=False, description="Include engineered features in response")


class PredictionResponse(BaseModel):
    """Response from single prediction endpoint."""
    
    success: bool
    prediction: Optional[PredictionOutput] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# BATCH PREDICTION REQUEST/RESPONSE
# ─────────────────────────────────────────────────────────────────

class BatchPredictionStatus(BaseModel):
    """Status summary for batch prediction."""
    
    total_rows: int
    rows_processed: int
    rows_failed: int
    churn_rate: float = Field(..., ge=0.0, le=1.0)
    avg_churn_probability: float
    avg_segment_confidence: float
    segment_distribution: Dict[int, int]


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""
    
    success: bool
    status: BatchPredictionStatus
    predictions: List[PredictionOutput]
    message: str


# ─────────────────────────────────────────────────────────────────
# WHAT-IF SIMULATION REQUEST/RESPONSE
# ─────────────────────────────────────────────────────────────────

class PredictionDelta(BaseModel):
    """Delta between original and modified predictions."""
    
    segment_changed: bool
    segment_delta: Optional[Dict[str, Any]] = None
    churn_probability_delta: float = Field(..., description="Modified prob - original prob")
    is_churner_changed: bool
    action_changed: bool = False


class WhatIfRequest(BaseModel):
    """Request for what-if simulation."""
    
    customer: CustomerInput
    modifications: Dict[str, Any] = Field(..., description="Features to modify, e.g., {'tenure': 50}")


class WhatIfResponse(BaseModel):
    """Response from what-if simulation."""
    
    success: bool
    original_prediction: Optional[PredictionOutput] = None
    modified_prediction: Optional[PredictionOutput] = None
    delta: Optional[PredictionDelta] = None
    modified_features: Dict[str, Any]
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# EXPLANATION REQUEST/RESPONSE
# ─────────────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    """Feature importance score."""
    
    feature_name: str
    importance: float = Field(..., description="Importance score (higher = more important)")
    sign: Optional[str] = Field(None, description="Direction of impact: 'positive', 'negative', or None")


class GlobalExplanation(BaseModel):
    """Global feature importance explanation."""
    
    top_features: List[FeatureImportance]
    explainer_type: str = Field(..., description="Type of explainer used (e.g., 'kernel', 'tree')")
    sample_size: int = Field(..., description="Number of samples used for explanation")


class InstanceExplanation(BaseModel):
    """Per-instance SHAP explanation."""
    
    customer_id: Optional[str] = None
    prediction: PredictionOutput
    top_features: List[FeatureImportance]
    base_value: Optional[float] = Field(None, description="Model base prediction value")


class ExplanationResponse(BaseModel):
    """Response from explanation endpoints."""
    
    success: bool
    explanation: Optional[Any] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# ERROR RESPONSE
# ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    success: bool = False
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
