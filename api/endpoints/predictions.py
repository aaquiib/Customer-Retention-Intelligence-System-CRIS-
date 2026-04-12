"""Prediction endpoints for single and batch customer scoring."""

import io
import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd

from api.app import ModelCache
from api.schemas import (
    BatchPredictionResponse,
    BatchPredictionStatus,
    CustomerInput,
    PredictionOutput,
    PredictionRequest,
    PredictionResponse,
)
from inference.business_rules import RetentionActionDecider

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_single(request: PredictionRequest):
    """
    Predict segment and churn for a single customer.
    
    Returns:
        - segment (0-3)
        - segment_label
        - churn_probability
        - is_churner (based on optimal threshold)
        - recommended action and priority score
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Convert customer input dict to standard format
        customer_dict = request.customer.model_dump(exclude_none=True)
        
        # Run inference
        pred = model_cache.pipeline.predict_single(
            customer_dict,
            return_intermediate=request.return_features
        )
        
        # Decide action
        action_key, action_label, priority_score, reason = model_cache.action_decider.decide_action(
            segment=pred['segment'],
            segment_label=pred['segment_label'],
            churn_probability=pred['churn_probability'],
            customer_features=customer_dict,
            segment_confidence=pred['segment_confidence']
        )
        
        # Build output
        output = PredictionOutput(
            segment=pred['segment'],
            segment_label=pred['segment_label'],
            churn_probability=pred['churn_probability'],
            is_churner=pred['is_churner'],
            threshold=pred['threshold'],
            segment_confidence=pred['segment_confidence'],
            input_features=customer_dict,
            engineered_features=pred.get('engineered_features')
        )
        
        response = {
            'success': True,
            'prediction': output,
            'recommended_action': action_label,
            'priority_score': priority_score,
            'action_reason': reason
        }
        
        return response
    
    except ValueError as e:
        logger.warning(f"Validation error in single prediction: {e}")
        return PredictionResponse(success=False, error=str(e))
    
    except Exception as e:
        logger.error(f"Error in single prediction: {e}", exc_info=True)
        return PredictionResponse(success=False, error="Prediction failed")


# ─────────────────────────────────────────────────────────────────
# BATCH PREDICTION (CSV UPLOAD)
# ─────────────────────────────────────────────────────────────────

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """
    Batch predict on CSV file.
    
    Expected CSV format:
        - Columns matching customer features (gender, tenure, MonthlyCharges, etc.)
        - One customer per row
    
    Returns:
        - List of predictions with segment, churn_prob, action
        - Summary statistics (churn_rate, distribution, etc.)
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Read CSV
        if not file.filename.endswith('.csv'):
            raise ValueError("File must be CSV format")
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        original_rows = len(df)
        if original_rows == 0:
            raise ValueError("CSV file is empty")
        
        if original_rows > 50000:
            raise ValueError("Batch size exceeds maximum of 50,000 rows")
        
        logger.info(f"Processing batch prediction with {original_rows} customers")
        
        # Run batch inference
        results_df, summary = model_cache.pipeline.predict_batch(df)
        
        # Add actions for each prediction
        action_results = []
        for idx, row in results_df.iterrows():
            pred_dict = row.to_dict()
            
            action_key, action_label, priority_score, reason = model_cache.action_decider.decide_action(
                segment=int(pred_dict['segment']),
                segment_label=pred_dict['segment_label'],
                churn_probability=float(pred_dict['churn_probability']),
                customer_features=df.iloc[idx].to_dict()
            )
            
            pred_dict['recommended_action'] = action_label
            pred_dict['priority_score'] = priority_score
            action_results.append(pred_dict)
        
        # Convert results to output format
        predictions = [
            PredictionOutput(
                segment=int(p['segment']),
                segment_label=p['segment_label'],
                churn_probability=float(p['churn_probability']),
                is_churner=bool(p['is_churner']),
                threshold=model_cache.pipeline.churn_threshold,
                segment_confidence=float(p['segment_confidence']),
                input_features=df.iloc[i].to_dict()
            )
            for i, p in enumerate(action_results)
        ]
        
        # Build status
        status = BatchPredictionStatus(
            total_rows=original_rows,
            rows_processed=original_rows,
            rows_failed=0,
            churn_rate=summary['churn_rate'],
            avg_churn_probability=summary['avg_churn_probability'],
            avg_segment_confidence=summary['avg_segment_confidence'],
            segment_distribution=summary['segments_distribution']
        )
        
        return BatchPredictionResponse(
            success=True,
            status=status,
            predictions=predictions,
            message=f"Processed {original_rows} customers successfully"
        )
    
    except ValueError as e:
        logger.warning(f"Validation error in batch prediction: {e}")
        return BatchPredictionResponse(
            success=False,
            status=None,
            predictions=[],
            message=f"Error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}", exc_info=True)
        return BatchPredictionResponse(
            success=False,
            status=None,
            predictions=[],
            message="Batch prediction failed"
        )


@router.get("/predict-batch/template")
async def get_batch_template():
    """
    Return CSV template for batch prediction.
    """
    template_data = {
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['Yes', 'No'],
        'tenure': [12, 24],
        'MonthlyCharges': [65.5, 85.0],
        'TotalCharges': [786.0, 2040.0],
        'PhoneService': ['Yes', 'No'],
        'MultipleLines': ['Yes', 'No', 'No phone service'],
        'InternetService': ['Fiber optic', 'DSL', 'No'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service'],
        'DeviceProtection': ['Yes', 'No', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service'],
        'StreamingTV': ['Yes', 'No', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    }
    
    df_template = pd.DataFrame({k: [v[0]] for k, v in template_data.items()})
    
    csv_buffer = io.StringIO()
    df_template.to_csv(csv_buffer, index=False)
    
    return {
        'template': csv_buffer.getvalue(),
        'columns': list(df_template.columns),
        'expected_format': 'CSV with headers',
        'max_rows': 50000
    }
