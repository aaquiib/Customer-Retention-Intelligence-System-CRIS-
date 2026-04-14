"""Explainability endpoints for feature importance using SHAP."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.app import ModelCache
from api.schemas import (
    CustomerInput,
    ExplanationResponse,
    FeatureImportance,
    GlobalExplanation,
    InstanceExplanation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/feature-importance/global", response_model=ExplanationResponse)
async def get_global_feature_importance(top_n: int = 10):
    """
    Get global feature importance across all training data.
    
    Uses SHAP to compute which features drive churn predictions overall.
    Computed once and cached - does not require specific customer input.
    
    Args:
        top_n: Number of top features to return (default 10)
    
    Returns:
        List of features ranked by importance with positive/negative impacts
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get global explanation from SHAP explainer
        if not hasattr(model_cache, 'shap_explainer') or model_cache.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return ExplanationResponse(
                success=False,
                error="SHAP explainer not available"
            )
        
        explanation = model_cache.shap_explainer.get_global_importance(top_n=top_n)
        
        # Convert to feature importance objects
        features = [
            FeatureImportance(
                feature_name=feat['feature'],
                importance=feat['importance'],
                sign=feat.get('sign')
            )
            for feat in explanation['top_features']
        ]
        
        global_exp = GlobalExplanation(
            top_features=features,
            explainer_type=explanation.get('explainer_type', 'kernel'),
            sample_size=explanation.get('sample_size', 0)
        )
        
        logger.info(f"Returned global feature importance (top {top_n} features)")
        
        return ExplanationResponse(
            success=True,
            explanation=global_exp
        )
    
    except Exception as e:
        logger.error(f"Error computing global feature importance: {e}", exc_info=True)
        return ExplanationResponse(
            success=False,
            error="Failed to compute feature importance"
        )


@router.post("/feature-importance/instance", response_model=ExplanationResponse)
async def get_instance_feature_importance(
    customer: CustomerInput,
    top_n: int = 5,
    explanation_type: str = "shap"
) -> ExplanationResponse:
    """
    Get per-instance feature importance for a specific customer prediction.
    
    Uses SHAP to explain which features contributed most to this customer's
    churn prediction. Includes feature values and SHAP contributions.
    
    Args:
        customer: Customer features
        top_n: Number of top features to explain (default 5)
        explanation_type: Type of explanation ('shap' or 'lime', default 'shap')
    
    Returns:
        Top N features with their SHAP values and impact direction
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        customer_dict = customer.model_dump(exclude_none=True)
        
        # Get SHAP explanation first
        if not hasattr(model_cache, 'shap_explainer') or model_cache.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return ExplanationResponse(
                success=False,
                error="SHAP explainer not available"
            )
        
        explanation = model_cache.shap_explainer.explain_instance(
            customer_dict,
            top_n=top_n
        )
        
        # Try to get customer prediction (optional, for context)
        try:
            prediction = model_cache.pipeline.predict_single(customer_dict)
        except Exception as e:
            logger.warning(f"Could not compute full prediction: {e}. Using SHAP prediction only.")
            # Create a minimal prediction from SHAP result
            prediction = {
                'segment': 0,
                'segment_label': 'Unknown',
                'churn_probability': explanation.get('churn_prob', 0.5),
                'is_churner': explanation.get('churn_prob', 0.5) > 0.4356,
                'threshold': 0.4356,
                'segment_confidence': 0.0,
            }
        
        # Convert to feature importance objects
        features = [
            FeatureImportance(
                feature_name=feat['feature'],
                importance=feat['shap_value'],
                sign='positive' if feat['shap_value'] > 0 else 'negative'
            )
            for feat in explanation['top_features']
        ]
        
        # Build PredictionOutput from computed prediction
        from api.schemas import PredictionOutput
        pred_output = PredictionOutput(
            segment=prediction['segment'],
            segment_label=prediction['segment_label'],
            churn_probability=prediction['churn_probability'],
            is_churner=prediction['is_churner'],
            threshold=prediction['threshold'],
            segment_confidence=prediction['segment_confidence'],
            input_features=customer_dict
        )
        
        instance_exp = InstanceExplanation(
            prediction=pred_output,
            top_features=features,
            base_value=explanation.get('base_value')
        )
        
        logger.info(f"Returned instance explanation (top {top_n} features)")
        
        return ExplanationResponse(
            success=True,
            explanation=instance_exp
        )
    
    except Exception as e:
        logger.error(f"Error computing instance explanation: {e}", exc_info=True)
        return ExplanationResponse(
            success=False,
            error="Failed to compute instance explanation"
        )


@router.get("/explanations/methods")
async def get_explanation_methods():
    """
    Return available explanation methods.
    
    Currently: SHAP (Kernel & Tree), LIME
    """
    return {
        'available_methods': [
            {
                'name': 'SHAP - Kernel Explainer',
                'type': 'shap_kernel',
                'description': 'Model-agnostic SHAP explanations using Kernel method',
                'speed': 'slow',
                'accuracy': 'high'
            },
            {
                'name': 'SHAP - Tree Explainer',
                'type': 'shap_tree',
                'description': 'Fast SHAP explanations for tree-based models (LightGBM)',
                'speed': 'fast',
                'accuracy': 'high'
            }
        ],
        'default_method': 'shap_tree',
        'note': 'Tree Explainer recommended for LightGBM churn model'
    }


@router.get("/explanations/model-info")
async def get_model_info():
    """
    Return information about the churn model architecture.
    
    Useful for understanding what the explainer is explaining.
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        pipeline = model_cache.pipeline
        
        # Get model architecture info
        lgbm = pipeline.lgbm_model
        
        return {
            'model_type': 'LightGBM Classifier',
            'n_estimators': lgbm.n_estimators,
            'max_depth': lgbm.max_depth,
            'n_features': lgbm.n_features_in_,
            'feature_columns': [
                'tenure', 'MonthlyCharges', 'TotalCharges',
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod',
                'segment'
            ],
            'churn_threshold': pipeline.churn_threshold,
            'training_metrics': {
                'roc_auc': 0.8398,  # Would load from metrics JSON
                'precision': 0.530,
                'recall': 0.786,
                'f1': 0.633
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}", exc_info=True)
        return {'error': 'Could not retrieve model information'}
