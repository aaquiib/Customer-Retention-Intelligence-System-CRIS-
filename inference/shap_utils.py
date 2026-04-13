"""SHAP utility functions for feature importance extraction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


def extract_top_shap_features(
    explainer: Optional[Any],
    model: Any,
    instance_data: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
    explainer_type: str = "tree"
) -> List[Dict[str, Any]]:
    """
    Extract top N features driving the prediction via SHAP.
    
    Args:
        explainer: SHAP explainer object (TreeExplainer or KernelExplainer)
        model: The ML model (LightGBM in this case)
        instance_data: Single instance as numpy array (1D)
        feature_names: List of feature names
        top_n: Number of top features to return
        explainer_type: 'tree' or 'kernel'
    
    Returns:
        List of dicts with keys: feature_name, shap_value, feature_value, impact_direction
    """
    if not SHAP_AVAILABLE or explainer is None:
        logger.warning("SHAP not available or explainer is None. Returning empty feature list.")
        return []
    
    try:
        # Ensure instance_data is properly shaped
        if len(instance_data.shape) == 1:
            instance_data = instance_data.reshape(1, -1)
        
        # Get SHAP values for this instance
        if explainer_type == "tree":
            shap_values = explainer.shap_values(instance_data)
        else:  # kernel
            shap_values = explainer.shap_values(instance_data)
        
        # Handle multi-class or binary case
        # For binary classification, shap_values is usually shape (n_samples, n_features) or (2, n_samples, n_features)
        if isinstance(shap_values, list):  # Multi-output
            shap_values = np.array(shap_values)
        
        if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
            # For binary churn prediction, take class 1 (churn)
            shap_values = shap_values[1, 0, :]
        elif len(shap_values.shape) == 2:  # (n_samples, n_features)
            shap_values = shap_values[0, :]
        else:
            shap_values = shap_values.flatten()
        
        # Get absolute SHAP values and sort
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        # Build result
        result = []
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            shap_value = float(shap_values[idx])
            feature_value = float(instance_data[0, idx])
            
            # Determine impact direction
            if abs(shap_value) < 1e-6:
                impact_direction = "neutral"
            elif shap_value > 0:
                impact_direction = "increases_churn"
            else:
                impact_direction = "decreases_churn"
            
            result.append({
                "feature_name": feature_name,
                "shap_value": shap_value,
                "feature_value": feature_value,
                "impact_direction": impact_direction
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error extracting SHAP features: {e}", exc_info=True)
        return []


def compute_batch_top_features(
    explainer: Optional[Any],
    model: Any,
    instances_data: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
    explainer_type: str = "tree"
) -> List[Dict[str, Any]]:
    """
    Compute aggregate SHAP feature importance across a batch.
    
    Args:
        explainer: SHAP explainer object
        model: The ML model
        instances_data: Multiple instances as numpy array (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to return
        explainer_type: 'tree' or 'kernel'
    
    Returns:
        List of dicts with keys: feature_name, avg_shap_value, frequency
    """
    if not SHAP_AVAILABLE or explainer is None:
        logger.warning("SHAP not available. Returning empty batch features.")
        return []
    
    try:
        # Get SHAP values for all instances
        if explainer_type == "tree":
            shap_values = explainer.shap_values(instances_data)
        else:
            shap_values = explainer.shap_values(instances_data)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        if len(shap_values.shape) == 3:  # (n_classes, n_samples, n_features)
            shap_values = shap_values[1, :, :]  # Class 1 for churn
        
        # Compute average absolute SHAP values per feature
        avg_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Track how many times each feature was in top-5
        feature_frequency = {i: 0 for i in range(len(feature_names))}
        
        for sample_shap in np.abs(shap_values):
            top_indices = np.argsort(sample_shap)[-top_n:]
            for idx in top_indices:
                feature_frequency[idx] += 1
        
        # Get top features by average SHAP
        top_indices = np.argsort(avg_abs_shap)[-top_n:][::-1]
        
        result = []
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            result.append({
                "feature_name": feature_name,
                "avg_shap_value": float(avg_abs_shap[idx]),
                "frequency": int(feature_frequency[idx])
            })
        
        return result
    
    except Exception as e:
        logger.error(f"Error computing batch SHAP features: {e}", exc_info=True)
        return []


def get_feature_names_from_preprocessor(preprocessor: Any) -> List[str]:
    """
    Extract output feature names from sklearn ColumnTransformer preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer from pipeline
    
    Returns:
        List of feature names in output order (after all transformations), cleaned of prefixes
    """
    try:
        # Use get_feature_names_out() which returns transformed feature names
        # This handles OneHotEncoder categories, StandardScaler passthrough, etc.
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = list(preprocessor.get_feature_names_out())
            
            # Clean up transformer prefixes (num__, cat__) for readability
            cleaned_names = []
            for name in feature_names:
                # Remove "num__" or "cat__" prefixes
                if name.startswith('num__'):
                    cleaned_name = name[5:]  # Remove "num__"
                elif name.startswith('cat__'):
                    cleaned_name = name[5:]  # Remove "cat__"
                else:
                    cleaned_name = name
                cleaned_names.append(cleaned_name)
            
            logger.debug(f"Extracted and cleaned {len(cleaned_names)} feature names")
            return cleaned_names
        else:
            # Fallback: construct manually from transformers
            logger.warning("get_feature_names_out() not available, using manual extraction")
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if hasattr(columns, '__iter__') and not isinstance(columns, str):
                    feature_names.extend(columns)
                else:
                    feature_names.append(columns)
            return feature_names
    except Exception as e:
        logger.warning(f"Could not extract feature names from preprocessor: {e}")
        return []
