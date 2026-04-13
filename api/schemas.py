"""Pydantic models for API request/response validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# 19-COLUMN PROCESSED INPUT (from processed_df.csv, excluding Churn)
# ─────────────────────────────────────────────────────────────────

class ProcessedCustomerInput(BaseModel):
    """Customer features for prediction - 19 columns from processed_df.csv.
    
    Flow: 19 input columns → engineered features → segment prediction (K-Prototypes)
          → segment label added → churn prediction (LightGBM with segment as feature)
    """
    
    # Demographic (4)
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    
    # Tenure & Charges (3)
    tenure: Optional[int] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    
    # Services (10)
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    
    # Contract & Billing (2)
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


# For backward compatibility
CustomerInput = ProcessedCustomerInput


# ─────────────────────────────────────────────────────────────────
# SHAP FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """Top feature driving churn prediction via SHAP."""
    
    feature_name: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (contribution to prediction)")
    feature_value: Any = Field(..., description="Actual value of this feature in the instance")
    impact_direction: str = Field(
        ..., 
        description="Direction of impact: 'increases_churn', 'decreases_churn', or 'neutral'"
    )


# ─────────────────────────────────────────────────────────────────
# RECOMMENDED ACTION
# ─────────────────────────────────────────────────────────────────

class RecommendedAction(BaseModel):
    """Retention action recommendation from business rules."""
    
    action_label: str = Field(..., description="Human-readable action (e.g., 'Retention Call', 'Loyalty Program')")
    priority_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Priority score for outreach (0.0-1.0, higher = more urgent)"
    )
    reason: str = Field(..., description="Explanation for recommending this action")


# ─────────────────────────────────────────────────────────────────
# PREDICTION OUTPUT (ENHANCED)
# ─────────────────────────────────────────────────────────────────

class PredictionOutput(BaseModel):
    """Single customer prediction result with SHAP features and actions."""
    
    # Segment Information
    segment: int = Field(..., ge=0, le=3, description="Segment ID (0-3)")
    segment_label: str = Field(..., description="Human-readable segment name")
    segment_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of segment assignment")
    
    # Churn Prediction
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Churn probability (0.0-1.0)")
    is_churner: bool = Field(..., description="True if churn_probability exceeds threshold")
    threshold: float = Field(..., description="Decision threshold used")
    
    # Feature Importance (SHAP)
    top_features: List[SHAPFeature] = Field(
        ..., 
        description="Top 5 features driving this prediction (SHAP values)"
    )
    
    # Business Action
    recommended_action: RecommendedAction = Field(
        ..., 
        description="Recommended retention action from business rules"
    )
    
    # Input Data (Echo)
    input_features: Dict[str, Any] = Field(..., description="Echo of 19 input features")
    engineered_features: Optional[Dict[str, Any]] = Field(
        None, 
        description="Engineered features (only if return_features=true)"
    )


# ─────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Request for single prediction endpoint."""
    
    customer: ProcessedCustomerInput = Field(..., description="19-column customer data")
    return_features: bool = Field(
        default=False, 
        description="Include engineered features in response"
    )


class PredictionResponse(BaseModel):
    """Response from single prediction endpoint."""
    
    success: bool
    prediction: Optional[PredictionOutput] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────

class BatchPredictionStatus(BaseModel):
    """Status summary for batch prediction."""
    
    total_rows: int = Field(..., description="Total rows in batch")
    rows_processed: int = Field(..., description="Rows successfully processed")
    rows_failed: int = Field(..., description="Rows that failed")
    churn_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of churners")
    avg_churn_probability: float = Field(..., description="Average churn probability")
    avg_segment_confidence: float = Field(..., description="Average segment confidence")
    segment_distribution: Dict[int, int] = Field(..., description="Count per segment (0-3)")
    action_distribution: Dict[str, int] = Field(
        ..., 
        description="Count per recommended action"
    )


class TopFeatureGlobal(BaseModel):
    """Global feature importance across batch."""
    
    feature_name: str = Field(..., description="Feature name")
    avg_shap_value: float = Field(..., description="Average absolute SHAP value across batch")
    frequency: int = Field(..., description="How many instances this feature appeared in top 5")


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""
    
    success: bool
    status: BatchPredictionStatus = Field(..., description="Batch processing summary")
    predictions: List[PredictionOutput] = Field(..., description="Per-row predictions")
    top_features_global: Optional[List[TopFeatureGlobal]] = Field(
        None, 
        description="Most impactful features across entire batch"
    )
    message: str = Field(..., description="Summary message")


# ─────────────────────────────────────────────────────────────────
# WHAT-IF SIMULATION (BACKWARD COMPATIBLE)
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
    
    customer: ProcessedCustomerInput
    modifications: Dict[str, Any] = Field(..., description="Features to modify")


class WhatIfResponse(BaseModel):
    """Response from what-if simulation."""
    
    success: bool
    original_prediction: Optional[PredictionOutput] = None
    modified_prediction: Optional[PredictionOutput] = None
    delta: Optional[PredictionDelta] = None
    modified_features: Dict[str, Any]
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# EXPLANATION (BACKWARD COMPATIBLE)
# ─────────────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    """Feature importance score."""
    
    feature_name: str
    importance: float = Field(..., description="Importance score")
    sign: Optional[str] = Field(None, description="Direction: 'positive', 'negative'")


class GlobalExplanation(BaseModel):
    """Global feature importance."""
    
    top_features: List[FeatureImportance]
    explainer_type: str = Field(..., description="Type of explainer")
    sample_size: int = Field(..., description="Number of samples used")


class InstanceExplanation(BaseModel):
    """Per-instance explanation."""
    
    customer_id: Optional[str] = None
    prediction: PredictionOutput
    top_features: List[FeatureImportance]
    base_value: Optional[float] = None


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
