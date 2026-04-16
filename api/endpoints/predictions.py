"""Prediction endpoints for single and batch customer scoring."""

import io
import logging
import time
from typing import Dict, List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd
import numpy as np

from api.app import ModelCache
from api.schemas import (
    BatchPredictionResponse,
    BatchPredictionStatus,
    PredictionOutput,
    PredictionRequest,
    PredictionResponse,
    ProcessedCustomerInput,
    RecommendedAction,
    SHAPFeature,
    TopFeatureGlobal,
)
from inference.shap_utils import extract_top_shap_features, compute_batch_top_features

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def normalize_csv_columns(df: pd.DataFrame, expected_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize CSV column names to match expected columns (case-insensitive).
    
    Args:
        df: DataFrame from CSV
        expected_columns: List of expected column names
    
    Returns:
        Tuple of (normalized_df, missing_columns)
    """
    # Create case-insensitive mapping
    expected_lower = {col.lower(): col for col in expected_columns}
    df_lower_cols = {col.lower(): col for col in df.columns}
    
    # Reorder and rename columns
    normalized_df = pd.DataFrame()
    missing = []
    
    for expected_col in expected_columns:
        lower_key = expected_col.lower()
        if lower_key in df_lower_cols:
            actual_col = df_lower_cols[lower_key]
            normalized_df[expected_col] = df[actual_col]
        else:
            missing.append(expected_col)
    
    return normalized_df, missing


def build_shap_features(
    explainer,
    model,
    preprocessed_data: np.ndarray,
    feature_names: List[str],
    top_n: int = 5
) -> List[SHAPFeature]:
    """
    Build SHAPFeature list from SHAP values.
    """
    shap_list = extract_top_shap_features(
        explainer=explainer,
        model=model,
        instance_data=preprocessed_data,
        feature_names=feature_names,
        top_n=top_n,
        explainer_type="tree"
    )
    
    return [
        SHAPFeature(
            feature_name=feat["feature_name"],
            shap_value=feat["shap_value"],
            feature_value=feat["feature_value"],
            impact_direction=feat["impact_direction"]
        )
        for feat in shap_list
    ]


# ─────────────────────────────────────────────────────────────────
# SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Predict segment and churn for a single customer.
    
    Input: 19-column customer data (ProcessedCustomerInput)
    
    Flow:
      1. Engineer features from 19 columns
      2. Predict segment (K-Prototypes)
      3. Add segment label to 19 columns
      4. Predict churn (LightGBM with segment)
      5. Extract top SHAP features driving prediction
      6. Decide retention action via business rules
    
    Returns:
        - segment (0-3)
        - segment_label
        - churn_probability
        - is_churner
        - top_features (SHAP-driven, top 5)
        - recommended_action (from business rules)
        - input_features (echo of 19 input columns)
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Convert customer input dict to standard format
        customer_dict = request.customer.model_dump(exclude_none=True)
        
        # Run inference (returns preprocessed data for SHAP)
        pred = model_cache.pipeline.predict_single(
            customer_dict,
            return_intermediate=request.return_features
        )
        
        # Extract SHAP features (top 5 driving the churn prediction)
        top_shap_features: List[SHAPFeature] = []
        try:
            # Get preprocessed data for SHAP explanation
            if model_cache.shap_explainer and model_cache.shap_explainer.explainer:
                # Prepare data for SHAP
                import pandas as pd
                df_pred = pd.DataFrame([customer_dict])
                from src.data import preprocess_data
                from src.features.engineering import engineer_features
                from src.config import get_config
                
                cfg = get_config()
                df_preprocessed = preprocess_data(df_pred, cfg)
                df_engineered = engineer_features(df_preprocessed, cfg)
                df_with_segment = df_engineered.copy()
                df_with_segment['segment'] = str(pred['segment'])
                
                X = model_cache.pipeline.churn_preprocessor.transform(df_with_segment)
                preprocessed_array = X.to_numpy() if hasattr(X, 'to_numpy') else X
                
                # Get feature names from preprocessor
                from inference.shap_utils import get_feature_names_from_preprocessor
                feature_names = get_feature_names_from_preprocessor(
                    model_cache.pipeline.churn_preprocessor
                )
                
                top_shap_features = build_shap_features(
                    explainer=model_cache.shap_explainer.explainer,
                    model=model_cache.pipeline.lgbm_model,
                    preprocessed_data=preprocessed_array,
                    feature_names=feature_names,
                    top_n=5
                )
        except Exception as e:
            logger.warning(f"Could not compute SHAP features: {e}. Returning empty feature list.")
            top_shap_features = []
        
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
            segment_confidence=pred['segment_confidence'],
            churn_probability=pred['churn_probability'],
            is_churner=pred['is_churner'],
            threshold=pred['threshold'],
            top_features=top_shap_features if top_shap_features else [],
            recommended_action=RecommendedAction(
                action_label=action_label,
                priority_score=priority_score,
                reason=reason
            ),
            input_features=customer_dict,
            engineered_features=pred.get('engineered_features')
        )
        
        return PredictionResponse(success=True, prediction=output, error=None)
    
    except ValueError as e:
        logger.warning(f"Validation error in single prediction: {e}")
        return PredictionResponse(success=False, prediction=None, error=str(e))
    
    except Exception as e:
        logger.error(f"Error in single prediction: {e}", exc_info=True)
        return PredictionResponse(success=False, prediction=None, error="Prediction failed")


# ─────────────────────────────────────────────────────────────────
# BATCH PREDICTION (CSV UPLOAD)
# ─────────────────────────────────────────────────────────────────

@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """
    Batch predict on CSV file with 19-column customer data.
    
    Input: CSV file with 19 columns (case-insensitive column names)
           E.g., gender, SeniorCitizen, Partner, ..., PaymentMethod
    
    Returns:
        - Per-row predictions with segment, churn probability, SHAP features, actions
        - Batch summary: churn_rate, segment_distribution, action_distribution
        - Top features across batch (global feature importance)
    
    Max batch size: 50,000 rows
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Validate file
        if not file.filename.endswith('.csv'):
            raise ValueError("File must be CSV format")
        
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        original_rows = len(df)
        if original_rows == 0:
            raise ValueError("CSV file is empty")
        
        if original_rows > 50000:
            raise ValueError("Batch size exceeds maximum of 50,000 rows")
        
        logger.info(f"Processing batch prediction with {original_rows} customers")
        
        # Normalize column names (case-insensitive matching)
        expected_19_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'MonthlyCharges', 'TotalCharges', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        
        df_normalized, missing_cols = normalize_csv_columns(df, expected_19_cols)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Run batch inference
        results_df, summary = model_cache.pipeline.predict_batch(df_normalized)
        
        batch_t0 = time.time()
        logger.info(f"Starting per-row prediction processing with SHAP explanations...")
        
        # Pre-compute SHAP values for entire batch (vectorized) - BEFORE the per-row loop
        all_shap_values = None
        feature_names = None
        X_batch = None
        
        try:
            if model_cache.shap_explainer and model_cache.shap_explainer.explainer:
                from src.data import preprocess_data
                from src.features.engineering import engineer_features
                from src.config import get_config
                from inference.shap_utils import get_feature_names_from_preprocessor
                
                cfg = get_config()
                
                # Preprocess and engineer features for entire batch
                df_preprocessed_batch = preprocess_data(df_normalized, cfg)
                df_engineered_batch = engineer_features(df_preprocessed_batch, cfg)
                
                # Add segment column (already in results_df)
                df_with_segment_batch = df_engineered_batch.copy()
                df_with_segment_batch['segment'] = results_df['segment'].astype(str)
                
                # Transform entire batch through preprocessor
                X_batch = model_cache.pipeline.churn_preprocessor.transform(df_with_segment_batch)
                X_batch = X_batch.to_numpy() if hasattr(X_batch, 'to_numpy') else X_batch
                
                # Get feature names once (not in loop)
                feature_names = get_feature_names_from_preprocessor(
                    model_cache.pipeline.churn_preprocessor
                )
                
                # Vectorized SHAP computation for entire batch
                logger.info(f"Computing SHAP values for batch of {len(X_batch)} samples...")
                shap_t0 = time.time()
                all_shap_values = model_cache.shap_explainer.explainer.shap_values(X_batch)
                
                # Handle binary classifier: use positive class (class 1)
                if isinstance(all_shap_values, list):
                    all_shap_values = all_shap_values[1]
                
                logger.info(f"Batch SHAP computation took {time.time()-shap_t0:.2f}s for {len(X_batch)} rows")
        except Exception as e:
            logger.debug(f"Could not pre-compute batch SHAP values: {e}")
            all_shap_values = None
        
        # Process per-row predictions with actions and SHAP features
        predictions_list = []
        
        for idx, row in results_df.iterrows():
            try:
                # Get basic prediction
                pred_dict = row.to_dict()
                
                # Get recommend action
                action_key, action_label, priority_score, reason = model_cache.action_decider.decide_action(
                    segment=int(pred_dict['segment']),
                    segment_label=pred_dict['segment_label'],
                    churn_probability=float(pred_dict['churn_probability']),
                    customer_features=df_normalized.iloc[idx].to_dict(),
                    segment_confidence=float(pred_dict['segment_confidence'])
                )
                
                # Extract top SHAP features for this row from pre-computed batch
                top_shap = []
                try:
                    if all_shap_values is not None and feature_names:
                        # Extract this row's SHAP values from the pre-computed batch matrix
                        row_shap = all_shap_values[idx]
                        
                        # Get top-N features by absolute SHAP value
                        top_n = 5
                        top_idx = np.argsort(np.abs(row_shap))[-top_n:][::-1]
                        
                        # Build feature list with SHAP contributions
                        top_shap = [
                            SHAPFeature(
                                feature_name=str(feature_names[j]),
                                shap_value=float(row_shap[j]),
                                feature_value=float(X_batch[idx, j]) if X_batch is not None and j < X_batch.shape[1] else None,
                                impact_direction="increases_churn" if row_shap[j] > 0 else "decreases_churn"
                            )
                            for j in top_idx
                        ]
                except Exception as e:
                    logger.debug(f"Could not extract SHAP features for row {idx}: {e}")
                
                # Build prediction output
                prediction = PredictionOutput(
                    segment=int(pred_dict['segment']),
                    segment_label=pred_dict['segment_label'],
                    segment_confidence=float(pred_dict['segment_confidence']),
                    churn_probability=float(pred_dict['churn_probability']),
                    is_churner=bool(pred_dict['is_churner']),
                    threshold=model_cache.pipeline.churn_threshold,
                    top_features=top_shap,
                    recommended_action=RecommendedAction(
                        action_label=action_label,
                        priority_score=priority_score,
                        reason=reason
                    ),
                    input_features=df_normalized.iloc[idx].to_dict(),
                    engineered_features=None
                )
                
                predictions_list.append(prediction)
            
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Skip this row, track as failed
        
        # Use cached global SHAP features instead of recomputing
        top_features_global = []
        try:
            model_cache = ModelCache.get_instance()
            cached_global_shap = model_cache.get_global_shap()
            
            if cached_global_shap:
                # Convert cached format to response format
                top_features_global = [
                    TopFeatureGlobal(
                        feature_name=feat['feature_name'],
                        avg_shap_value=feat['importance'],  # Cached importance is already mean absolute SHAP
                        frequency=0  # Set to 0 for cached features (pre-computed aggregate)
                    )
                    for feat in cached_global_shap[:5]
                ]
        except Exception as e:
            logger.debug(f"Could not get cached global SHAP features: {e}")
        
        batch_elapsed = time.time() - batch_t0
        
        # Get action distribution
        action_dist = {}
        for pred in predictions_list:
            action = pred.recommended_action.action_label
            action_dist[action] = action_dist.get(action, 0) + 1
        
        # Build status
        status = BatchPredictionStatus(
            total_rows=original_rows,
            rows_processed=len(predictions_list),
            rows_failed=original_rows - len(predictions_list),
            churn_rate=summary['churn_rate'],
            avg_churn_probability=summary['avg_churn_probability'],
            avg_segment_confidence=summary['avg_segment_confidence'],
            segment_distribution=summary['segments_distribution'],
            action_distribution=action_dist
        )
        
        logger.info(f"Batch of {len(predictions_list)} rows completed in {batch_elapsed:.2f}s (SHAP from cache + vectorized computation)")
        
        return BatchPredictionResponse(
            success=True,
            status=status,
            predictions=predictions_list,
            top_features_global=top_features_global if top_features_global else None,
            message=f"Processed {len(predictions_list)}/{original_rows} customers successfully in {batch_elapsed:.2f}s"
        )
    
    except ValueError as e:
        logger.warning(f"Validation error in batch prediction: {e}")
        return BatchPredictionResponse(
            success=False,
            status=BatchPredictionStatus(
                total_rows=0, rows_processed=0, rows_failed=0,
                churn_rate=0.0, avg_churn_probability=0.0, avg_segment_confidence=0.0,
                segment_distribution={}, action_distribution={}
            ),
            predictions=[],
            top_features_global=None,
            message=f"Error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}", exc_info=True)
        return BatchPredictionResponse(
            success=False,
            status=BatchPredictionStatus(
                total_rows=0, rows_processed=0, rows_failed=0,
                churn_rate=0.0, avg_churn_probability=0.0, avg_segment_confidence=0.0,
                segment_distribution={}, action_distribution={}
            ),
            predictions=[],
            top_features_global=None,
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
        'expected_19_columns': [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'MonthlyCharges', 'TotalCharges', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ],
        'expected_format': 'CSV with headers (column names case-insensitive)',
        'max_rows': 50000,
        'note': 'Column names are case-insensitive. Order does not matter.'
    }

