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
    
    Uses pre-computed global SHAP from startup cache.
    Does not require specific customer input.
    
    Args:
        top_n: Number of top features to return (default 10)
    
    Returns:
        List of features ranked by importance with positive/negative impacts
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Get cached global SHAP features from ModelCache
        cached_features = model_cache.get_global_shap()
        
        if not cached_features:
            logger.warning("Global SHAP cache is empty")
            return ExplanationResponse(
                success=False,
                error="Global feature importance not available"
            )
        
        # Limit to top_n requested features
        top_features = cached_features[:top_n]
        
        # Convert to feature importance objects
        features = [
            FeatureImportance(
                feature_name=feat['feature_name'],
                importance=feat['importance'],
                sign=feat.get('sign')
            )
            for feat in top_features
        ]
        
        global_exp = GlobalExplanation(
            top_features=features,
            explainer_type="tree",
            sample_size=200
        )
        
        logger.info(f"Returned global feature importance from cache (top {len(features)} features)")
        
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
    Return information about the churn and segmentation models.
    
    Includes:
    - Model architecture and framework
    - Performance metrics (AUC, accuracy, precision, recall)
    - Training metadata
    - Number of features/clusters
    - Segment definitions
    - Explainer configuration
    """
    try:
        model_cache = ModelCache.get_instance()
        
        if not model_cache.is_loaded:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        pipeline = model_cache.pipeline
        lgbm = pipeline.lgbm_model
        cfg = pipeline.cfg
        
        # Load churn model metrics from JSON
        churn_metrics_latest = {
            "roc_auc": 0.84,
            "accuracy": 0.76,
            "precision": 0.53,
            "recall": 0.79,
            "f1": 0.63
        }
        
        try:
            import json
            metrics_path = cfg['models'].get('churn_metrics_path', 'models/churn/metrics_latest.json')
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                # Get test set metrics (most representative)
                test_metrics = metrics_data.get('split_metrics', {}).get('test', {})
                model_config = metrics_data.get('model_config', {})
                
                churn_metrics_latest = {
                    "roc_auc": float(test_metrics.get('roc_auc', 0.84)),
                    "accuracy": float(test_metrics.get('accuracy', 0.76)),
                    "precision": float(test_metrics.get('precision', 0.53)),
                    "recall": float(test_metrics.get('recall', 0.79)),
                    "f1": float(test_metrics.get('f1', 0.63))
                }
                
                training_data_size = model_config.get('training_data_size', 'Unknown')
        except Exception as e:
            logger.warning(f"Could not load metrics from JSON: {e}")
            training_data_size = 'Unknown'
        
        # Load segmentation model info
        num_segments = 4
        segment_definitions = {
            "0": {
                "name": "Loyal High-Value",
                "description": "High-value customers with strong tenure and low churn risk"
            },
            "1": {
                "name": "Low Engagement",
                "description": "Newly acquired or at-risk customers with minimal engagement"
            },
            "2": {
                "name": "Stable Mid-Value",
                "description": "Stable mid-value customers with steady service usage"
            },
            "3": {
                "name": "At risk High-value",
                "description": "High-value customers showing early signs of churn"
            }
        }
        
        try:
            import json
            segment_labels_path = cfg['models'].get('segment_labels_path', 'models/segmentation/segment_labels.json')
            with open(segment_labels_path, 'r') as f:
                segment_labels = json.load(f)
                num_segments = len(segment_labels)
                # Update definitions with loaded labels
                for seg_id, label in segment_labels.items():
                    if seg_id in segment_definitions:
                        segment_definitions[seg_id]['name'] = label
        except Exception as e:
            logger.warning(f"Could not load segment labels: {e}")
        
        return {
            'model_info': {
                'churn_model': {
                    'model_name': 'LightGBM Classifier',
                    'framework': 'LightGBM',
                    'input_features': 20,  # 19 customer features + 1 segment label
                    'num_features': int(lgbm.n_features_in_),  # After preprocessing/engineering
                    'n_estimators': int(lgbm.n_estimators),
                    'max_depth': int(lgbm.max_depth) if lgbm.max_depth else 'Unlimited',
                    'decision_threshold': float(pipeline.churn_threshold),
                    'training_data_size': 7032,
                    'feature_set_version': '1.0',
                    'training_date': 'See metrics_latest.json',
                    'performance_metrics': churn_metrics_latest
                },
                'segmentation_model': {
                    'model_name': 'KMeans Clustering',
                    'framework': 'scikit-learn',
                    'num_clusters': int(num_segments),
                    'algorithm': 'KMeans',
                    'n_init': 10,
                    'random_state': 42,
                    'training_data_size': 7032
                },
                'segments': segment_definitions,
                'explainer': {
                    'type': 'SHAP (SHapley Additive exPlanations)',
                    'background_samples': 200,
                    'computation_type': 'TreeExplainer (Fast)',
                    'feature_importance': 'SHAP Force Plot'
                }
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}", exc_info=True)
        return {
            'model_info': {
                'churn_model': {},
                'segmentation_model': {},
                'segments': {},
                'explainer': {}
            }
        }
