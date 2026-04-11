"""
Artifact Loader and Predictor Classes for Churn Segmentation Decision System

This module provides utilities to load trained models and make predictions
using the saved artifacts from the final model training pipeline.

Classes:
    ModelArtifactLoader: Loads models, preprocessors, and metadata
    ChurnPredictor: Unified inference interface for multiple models
"""

import json
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostClassifier


class ModelArtifactLoader:
    """
    Loads all saved model artifacts including models, preprocessors, 
    hyperparameters, thresholds, and metadata.
    
    Attributes:
        artifact_dir (Path): Directory containing model artifacts
        preprocessor: Sklearn ColumnTransformer for feature preprocessing
        feature_metadata (dict): Information about feature columns
        metadata (dict): Comprehensive project metadata
        models (dict): Dictionary of loaded model objects
        configs (dict): Configuration/hyperparameters for each model
        thresholds (dict): Optional decision thresholds per model
    """
    
    def __init__(self, artifact_dir):
        """
        Initialize loader and load all artifacts from disk.
        
        Args:
            artifact_dir (str or Path): Path to model artifact directory
                (e.g., models/v1.0.0-20260411_224248/)
        """
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
        
        self.preprocessor = None
        self.feature_metadata = {}
        self.metadata = {}
        self.models = {}
        self.configs = {}
        self.thresholds = {}
        
        self._load_all()
    
    def _load_all(self):
        """Load all artifacts from disk."""
        self._load_preprocessing()
        self._load_metadata()
        self._load_models()
        self._load_thresholds()
        print("✅ All artifacts loaded successfully")
    
    def _load_preprocessing(self):
        """Load preprocessing pipeline and feature metadata."""
        # Preprocessor
        preprocessor_path = self.artifact_dir / "preprocessing" / "preprocessor.pkl"
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Feature columns
        feature_path = self.artifact_dir / "preprocessing" / "feature_columns.json"
        with open(feature_path, 'r') as f:
            self.feature_metadata = json.load(f)
    
    def _load_metadata(self):
        """Load comprehensive metadata."""
        metadata_path = self.artifact_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def _load_models(self):
        """Load trained models and their configurations."""
        # LightGBM
        lgbm_path = self.artifact_dir / "lightgbm" / "model.txt"
        if lgbm_path.exists():
            self.models['LightGBM'] = lgb.Booster(model_file=str(lgbm_path))
            config_path = self.artifact_dir / "lightgbm" / "config.yaml"
            with open(config_path, 'r') as f:
                self.configs['LightGBM'] = yaml.safe_load(f)
        
        # CatBoost
        cb_path = self.artifact_dir / "catboost" / "model.cbm"
        if cb_path.exists():
            self.models['CatBoost'] = CatBoostClassifier()
            self.models['CatBoost'].load_model(str(cb_path))
            config_path = self.artifact_dir / "catboost" / "config.yaml"
            with open(config_path, 'r') as f:
                self.configs['CatBoost'] = yaml.safe_load(f)
    
    def _load_thresholds(self):
        """Load decision thresholds if available."""
        threshold_path = self.artifact_dir / "lightgbm" / "threshold.pkl"
        if threshold_path.exists():
            threshold_data = joblib.load(threshold_path)
            self.thresholds['LightGBM'] = threshold_data.get('best_threshold', 0.5)
    
    def get_model(self, model_name):
        """
        Get a specific model.
        
        Args:
            model_name (str): Name of the model ('LightGBM' or 'CatBoost')
        
        Returns:
            Model object or Booster
        """
        return self.models.get(model_name)
    
    def get_threshold(self, model_name, default=0.5):
        """
        Get decision threshold for a model.
        
        Args:
            model_name (str): Model name
            default (float): Default threshold if not found
        
        Returns:
            float: Decision threshold
        """
        return self.thresholds.get(model_name, default)


class ChurnPredictor:
    """
    Unified prediction interface for churn classification.
    
    Supports making predictions with any loaded model, with optional
    threshold application for custom decision boundaries.
    """
    
    def __init__(self, loader):
        """
        Initialize predictor with loaded artifacts.
        
        Args:
            loader (ModelArtifactLoader): Initialized loader with artifacts
        """
        self.loader = loader
        self.num_cols = loader.feature_metadata['num_feature_names']
        self.cat_cols = loader.feature_metadata['cat_feature_names']
    
    def preprocess(self, X):
        """
        Preprocess features using the loaded preprocessor.
        
        Args:
            X (pd.DataFrame): Raw features
        
        Returns:
            np.ndarray: Preprocessed features (encoded/scaled)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Ensure all required columns are present
        missing_cols = set(self.num_cols + self.cat_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X_subset = X[self.num_cols + self.cat_cols]
        return self.loader.preprocessor.transform(X_subset)
    
    def predict_proba(self, X, model_name='LightGBM'):
        """
        Get probability predictions from specified model.
        
        Args:
            X (pd.DataFrame): Raw features
            model_name (str): Model to use ('LightGBM' or 'CatBoost')
        
        Returns:
            np.ndarray: Probability of class 1 (Churn)
        """
        model = self.loader.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        X_processed = self.preprocess(X)
        
        if model_name == 'LightGBM':
            return model.predict(X_processed)
        elif model_name == 'CatBoost':
            # CatBoost needs raw features with categorical indices
            X_cat = X[self.cat_cols].astype(str)
            X_num = X[self.num_cols]
            X_combined = pd.concat([X_num.reset_index(drop=True), 
                                   X_cat.reset_index(drop=True)], axis=1)
            cat_indices = list(range(len(self.num_cols), 
                                    len(self.num_cols) + len(self.cat_cols)))
            
            from catboost import Pool
            pool = Pool(X_combined, cat_features=cat_indices)
            return model.predict_proba(pool)[:, 1]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def predict(self, X, model_name='LightGBM', threshold=None):
        """
        Make binary predictions using specified model.
        
        Args:
            X (pd.DataFrame): Raw features
            model_name (str): Model to use
            threshold (float): Custom decision threshold (None = use config default)
        
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X, model_name=model_name)
        
        if threshold is None:
            threshold = self.loader.get_threshold(model_name, default=0.5)
        
        return (proba >= threshold).astype(int)
    
    def predict_batch(self, X, model_name='LightGBM', return_proba=False, 
                     threshold=None):
        """
        Batch prediction with optional probability output.
        
        Args:
            X (pd.DataFrame): Features for multiple samples
            model_name (str): Model to use
            return_proba (bool): Return probabilities alongside predictions
            threshold (float): Custom decision threshold
        
        Returns:
            dict: Contains 'predictions' and optionally 'probabilities'
        """
        proba = self.predict_proba(X, model_name=model_name)
        
        if threshold is None:
            threshold = self.loader.get_threshold(model_name, default=0.5)
        
        predictions = (proba >= threshold).astype(int)
        
        result = {'predictions': predictions}
        if return_proba:
            result['probabilities'] = proba
            result['threshold_used'] = threshold
        
        return result
    
    def get_feature_importance(self, model_name='LightGBM', top_n=20):
        """
        Get feature importances from the model (if available).
        
        Args:
            model_name (str): Model to use
            top_n (int): Number of top features to return
        
        Returns:
            pd.DataFrame: Feature importances sorted by importance
        """
        model = self.loader.get_model(model_name)
        
        if model_name == 'LightGBM':
            importance_values = model.feature_importance()
            
            # Get feature names from preprocessor
            feature_names = self._get_feature_names_lightgbm()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        else:
            raise NotImplementedError(f"Feature importance not implemented for {model_name}")
    
    def _get_feature_names_lightgbm(self):
        """Get feature names after preprocessing for LightGBM."""
        # Get one-hot encoded feature names from preprocessor
        try:
            feature_names = self.loader.preprocessor.get_feature_names_out()
            return list(feature_names)
        except:
            # Fallback if get_feature_names_out not available
            n_num = len(self.num_cols)
            
            # Estimate categorical features from one-hot encoding
            cat_feature_names = []
            for col in self.cat_cols:
                # This is approximate - would need preprocessor state for exact names
                cat_feature_names.append(f"{col}_*")
            
            return self.num_cols + cat_feature_names
    
    def explain_prediction(self, X_single, model_name='LightGBM', 
                          threshold=None):
        """
        Generate human-readable explanation for a single prediction.
        
        Args:
            X_single (pd.Series or dict): Single sample
            model_name (str): Model to use
            threshold (float): Custom decision threshold
        
        Returns:
            dict: Prediction details and explanation
        """
        # Convert to DataFrame if needed
        if isinstance(X_single, dict):
            X_single = pd.DataFrame([X_single])
        elif isinstance(X_single, pd.Series):
            X_single = X_single.to_frame().T
        
        proba = self.predict_proba(X_single, model_name=model_name)[0]
        
        if threshold is None:
            threshold = self.loader.get_threshold(model_name, default=0.5)
        
        prediction = int(proba >= threshold)
        confidence = max(proba, 1 - proba)
        
        return {
            'prediction': prediction,
            'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
            'churn_probability': float(proba),
            'confidence': float(confidence),
            'threshold_used': threshold,
            'model_used': model_name,
            'features_input': X_single.to_dict('records')[0]
        }


# ============================================================================
# SEGMENTATION MODEL ARTIFACTS
# ============================================================================

class SegmentationModelLoader:
    """
    Loads segmentation model artifacts (KPrototypes clustering with preprocessing).
    
    Attributes:
        artifact_dir (Path): Directory containing model artifacts
        model: Fitted KPrototypes clustering model
        scaler: StandardScaler for numeric feature preprocessing
        feature_config (dict): Feature names, types, and scaler parameters
        metadata (dict): Comprehensive model metadata
        segment_labels (dict): Human-friendly segment names and descriptions
    """
    
    def __init__(self, artifact_dir):
        """
        Initialize loader and load all segmentation artifacts from disk.
        
        Args:
            artifact_dir (str or Path): Path to segmentation artifact directory
                (e.g., models/v1.0.0-segmentation-20260411_224248/)
        """
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")
        
        self.model = None
        self.scaler = None
        self.feature_config = {}
        self.metadata = {}
        self.segment_labels = {}
        
        self._load_all()
    
    def _load_all(self):
        """Load all segmentation artifacts from disk."""
        self._load_model()
        self._load_preprocessor()
        self._load_feature_config()
        self._load_metadata()
        self._load_segment_labels()
        print("✅ Segmentation model artifacts loaded successfully")
    
    def _load_model(self):
        """Load trained KPrototypes model."""
        model_path = self.artifact_dir / "kprototypes" / "model.pkl"
        self.model = joblib.load(model_path)
    
    def _load_preprocessor(self):
        """Load StandardScaler for numeric features."""
        scaler_path = self.artifact_dir / "preprocessing" / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
    
    def _load_feature_config(self):
        """Load feature configuration and metadata."""
        config_path = self.artifact_dir / "preprocessing" / "feature_config.json"
        with open(config_path, 'r') as f:
            self.feature_config = json.load(f)
    
    def _load_metadata(self):
        """Load comprehensive model metadata."""
        metadata_path = self.artifact_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def _load_segment_labels(self):
        """Load human-friendly segment labels."""
        labels_path = self.artifact_dir / "metadata" / "segment_labels.json"
        with open(labels_path, 'r') as f:
            self.segment_labels = json.load(f)
    
    def get_cluster_stats(self):
        """
        Get cluster statistics from metadata.
        
        Returns:
            dict: Cluster statistics (size, churn_rate, medians, etc.)
        """
        return self.metadata.get('cluster_statistics', {})
    
    def get_segment_info(self, segment_id):
        """
        Get human-friendly information about a segment.
        
        Args:
            segment_id (int or str): Segment/cluster ID (0-3)
        
        Returns:
            dict: Segment label, description, and statistics
        """
        segment_id = str(segment_id)
        segment_info = {
            'label': self.segment_labels.get(segment_id, {}).get('label', 'Unknown'),
            'description': self.segment_labels.get(segment_id, {}).get('description', ''),
            'statistics': self.metadata.get('cluster_statistics', {}).get(segment_id, {})
        }
        return segment_info


class SegmentationInference:
    """
    Unified inference interface for customer segmentation.
    
    Performs feature engineering, preprocessing, and clustering to assign
    customers to behavioral segments.
    """
    
    def __init__(self, loader):
        """
        Initialize with loaded segmentation artifacts.
        
        Args:
            loader (SegmentationModelLoader): Loaded segmentation model
        """
        self.loader = loader
        self.model = loader.model
        self.scaler = loader.scaler
        self.feature_config = loader.feature_config
        self.numeric_cols = loader.feature_config.get('numeric_columns', [])
        self.categorical_cols = loader.feature_config.get('categorical_columns', [])
        self.categorical_indices = loader.feature_config.get('categorical_indices', [])
        self.segment_labels = loader.segment_labels
    
    def preprocess(self, X):
        """
        Preprocess features for segmentation model.
        
        Args:
            X (pd.DataFrame): Raw features (must include all segmentation features)
        
        Returns:
            np.ndarray: Preprocessed features (scaled numeric, categorical as-is)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        # Get segmentation features in order
        seg_features = self.feature_config.get('segmentation_features', [])
        missing_cols = set(seg_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        X_subset = X[seg_features].copy()
        
        # Scale numeric columns
        X_scaled = X_subset.copy()
        X_scaled[self.numeric_cols] = self.scaler.transform(X_subset[self.numeric_cols])
        
        # Convert categorical to string for KPrototypes
        for col in self.categorical_cols:
            X_scaled[col] = X_scaled[col].astype(str)
        
        return X_scaled.to_numpy()
    
    def predict_segment(self, X):
        """
        Assign customers to segments (cluster IDs 0-3).
        
        Args:
            X (pd.DataFrame): Raw features for customers
        
        Returns:
            np.ndarray: Segment assignments (integers 0-3)
        """
        X_processed = self.preprocess(X)
        return self.model.predict(X_processed, categorical=self.categorical_indices)
    
    def predict_segment_with_labels(self, X):
        """
        Assign customers to segments with human-friendly labels.
        
        Args:
            X (pd.DataFrame): Raw features for customers
        
        Returns:
            pd.DataFrame: Original data with added 'segment' and 'segment_label' columns
        """
        X_result = X.copy()
        segments = self.predict_segment(X)
        X_result['segment'] = segments
        X_result['segment_label'] = [
            self.segment_labels.get(str(s), {}).get('label', f'Cluster {s}')
            for s in segments
        ]
        return X_result
    
    def predict_batch(self, X, include_labels=True):
        """
        Batch segment assignment with optional label output.
        
        Args:
            X (pd.DataFrame): Features for multiple customers
            include_labels (bool): Include human-friendly labels in output
        
        Returns:
            dict: Contains 'segments' and optionally 'labels'
        """
        segments = self.predict_segment(X)
        
        result = {'segments': segments}
        
        if include_labels:
            labels = [
                self.segment_labels.get(str(s), {}).get('label', f'Cluster {s}')
                for s in segments
            ]
            result['labels'] = labels
            result['descriptions'] = [
                self.segment_labels.get(str(s), {}).get('description', '')
                for s in segments
            ]
        
        return result
    
    def explain_segment_assignment(self, X_single):
        """
        Generate detailed explanation for a single customer's segment assignment.
        
        Args:
            X_single (pd.Series or dict): Single customer record
        
        Returns:
            dict: Segment assignment with business context and profile
        """
        # Convert to DataFrame if needed
        if isinstance(X_single, dict):
            X_single = pd.DataFrame([X_single])
        elif isinstance(X_single, pd.Series):
            X_single = X_single.to_frame().T
        
        segment_id = self.predict_segment(X_single)[0]
        segment_info = self.loader.get_segment_info(segment_id)
        
        return {
            'segment_id': int(segment_id),
            'segment_label': segment_info['label'],
            'segment_description': segment_info['description'],
            'segment_profile': segment_info['statistics'],
            'customer_attributes': X_single.to_dict('records')[0]
        }


# Example usage functions
def load_artifacts(artifact_dir):
    """
    Convenience function to load all artifacts.
    
    Args:
        artifact_dir: Path to artifact directory
    
    Returns:
        tuple: (loader, predictor)
    """
    loader = ModelArtifactLoader(artifact_dir)
    predictor = ChurnPredictor(loader)
    return loader, predictor


def load_segmentation_model(artifact_dir):
    """
    Convenience function to load segmentation model artifacts.
    
    Args:
        artifact_dir: Path to segmentation artifact directory
    
    Returns:
        tuple: (loader, inference)
    """
    loader = SegmentationModelLoader(artifact_dir)
    inference = SegmentationInference(loader)
    return loader, inference


if __name__ == "__main__":
    # Example: Load and make predictions
    print("Artifact Loader - Churn Segmentation Decision System")
    print("=" * 60)
    print("\nChurn Prediction Usage:")
    print("  from artifacts import load_artifacts")
    print("  loader, predictor = load_artifacts('models/v1.0.0-xxx/')")
    print("  predictions = predictor.predict(X_df)")
    print("  probabilities = predictor.predict_proba(X_df)")
    
    print("\nSegmentation Usage:")
    print("  from artifacts import load_segmentation_model")
    print("  loader, inference = load_segmentation_model('models/v1.0.0-segmentation-xxx/')")
    print("  segments = inference.predict_segment(X_df)")
    print("  result = inference.predict_segment_with_labels(X_df)")
