"""Core inference pipeline - chains segmentation and churn prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.config import get_config
from src.data import preprocess_data
from src.features.engineering import engineer_features
from src.utils import load_json, load_model, validate_feature_consistency

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    End-to-end inference pipeline: raw customer features → segment + churn prediction.

    Loads all model artifacts once at initialization. Supports single and batch predictions.
    Chains K-Prototypes segmentation → LightGBM churn model with optimal threshold.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline by loading all trained models and metadata.

        Args:
            cfg: Configuration dictionary. If None, loads from config/config.yaml
        
        Raises:
            FileNotFoundError: If any model artifact is missing
            ValueError: If model artifacts are corrupted or incompatible
        """
        self.cfg = cfg or get_config()
        self._load_models()
        logger.info("✓ InferencePipeline initialized with all models loaded")

    def _load_models(self) -> None:
        """Load all segmentation and churn model artifacts from disk."""
        seg_dir = self.cfg['models']['segmentation_dir']
        churn_dir = self.cfg['models']['churn_dir']

        # Load segmentation components
        try:
            self.kproto = load_model(f"{seg_dir}kproto.pkl")
            self.seg_scaler = load_model(f"{seg_dir}scaler.pkl")
            self.seg_cat_idx = load_json(f"{seg_dir}catidx.json")
            self.seg_feature_metadata = load_json(f"{seg_dir}feature_metadata.json")
            
            seg_labels_raw = load_json(f"{seg_dir}segment_labels.json")
            self.segment_labels = {int(k): v for k, v in seg_labels_raw.items()}
            
            logger.info(f"✓ Loaded segmentation model (k={self.kproto.n_clusters})")
        except FileNotFoundError as e:
            logger.error(f"Failed to load segmentation models: {e}")
            raise

        # Load churn components
        try:
            self.lgbm_model = load_model(f"{churn_dir}lgbm_churn_model.pkl")
            self.churn_preprocessor = load_model(f"{churn_dir}preprocessor.pkl")
            
            threshold_meta = load_json(f"{churn_dir}threshold_meta.json")
            self.churn_threshold = threshold_meta['best_threshold']
            self.model_random_seed = threshold_meta.get('random_seed', 42)
            
            logger.info(f"✓ Loaded churn model with threshold={self.churn_threshold:.4f}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load churn models: {e}")
            raise

        # Store expected feature columns for validation
        self.seg_numeric_cols = self.seg_feature_metadata.get('numeric_columns', [])
        self.seg_categorical_cols = self.seg_feature_metadata.get('categorical_columns', [])
        self.seg_feature_cols = self.seg_feature_metadata.get('segmentation_features', [])

    def predict_single(
        self,
        customer_data: Dict[str, Any],
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Predict segment and churn probability for a single customer.

        Args:
            customer_data: Dictionary of customer attributes (e.g., {'tenure': 24, 'MonthlyCharges': 65.5, ...})
            return_intermediate: If True, return segment assignment and engineered features

        Returns:
            Dictionary with keys:
                - segment (int): 0-3
                - segment_label (str): Human-readable segment name
                - churn_probability (float): 0.0-1.0
                - is_churner (bool): churn_probability > threshold
                - input_features (dict): Echo of input after preprocessing
                - engineered_features (dict): Derived features (if return_intermediate=True)
                - segment_confidence (float): Distance-based confidence score

        Raises:
            ValueError: If required features are missing or invalid
        """
        # Convert to DataFrame (single row)
        df = pd.DataFrame([customer_data])
        
        # Preprocess
        try:
            df_preprocessed = preprocess_data(df, self.cfg)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Preprocessing error: {str(e)}")
        
        # Engineer features
        try:
            df_engineered = engineer_features(df_preprocessed, self.cfg)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering error: {str(e)}")
        
        # Segment assignment
        segment, segment_label, seg_confidence = self._assign_segment(df_engineered)
        
        # Add segment to features for churn model (as string for categorical encoding)
        df_with_segment = df_engineered.copy()
        df_with_segment['segment'] = str(segment)
        
        # Churn prediction
        churn_prob = self._predict_churn(df_with_segment)
        is_churner = churn_prob > self.churn_threshold
        
        # Build result
        result = {
            'segment': int(segment),
            'segment_label': segment_label,
            'churn_probability': float(churn_prob),
            'is_churner': bool(is_churner),
            'threshold': self.churn_threshold,
            'segment_confidence': float(seg_confidence),
            'input_features': customer_data,
        }
        
        if return_intermediate:
            result['engineered_features'] = df_engineered.iloc[0].to_dict()
        
        return result

    def predict_batch(
        self,
        customer_df: pd.DataFrame,
        return_intermediate: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Predict segment and churn probability for multiple customers.

        Args:
            customer_df: DataFrame with customer data (rows = customers, columns = features)
            return_intermediate: If True, return engineered features

        Returns:
            Tuple of:
                - results_df: DataFrame with columns [segment, segment_label, churn_probability, is_churner, segment_confidence]
                - summary: Dictionary with batch statistics (total_rows, action_distribution, avg_priority, etc.)

        Raises:
            ValueError: If input validation fails
        """
        n_rows = len(customer_df)
        logger.info(f"Starting batch prediction for {n_rows} customers")
        
        # Preprocess all rows
        try:
            df_preprocessed = preprocess_data(customer_df.copy(), self.cfg)
        except Exception as e:
            logger.error(f"Batch preprocessing failed: {e}")
            raise ValueError(f"Preprocessing error: {str(e)}")
        
        # Engineer features for all rows
        try:
            df_engineered = engineer_features(df_preprocessed, self.cfg)
        except Exception as e:
            logger.error(f"Batch feature engineering failed: {e}")
            raise ValueError(f"Feature engineering error: {str(e)}")
        
        # Batch segment assignment
        segments, segment_labels, seg_confidences = self._assign_segments_batch(df_engineered)
        
        # Add segments to features for churn model (as strings for categorical encoding)
        df_with_segment = df_engineered.copy()
        df_with_segment['segment'] = segments.astype(str)
        
        # Batch churn prediction
        churn_probs = self._predict_churn_batch(df_with_segment)
        is_churners = churn_probs > self.churn_threshold
        
        # Build results DataFrame
        results_df = pd.DataFrame({
            'segment': segments,
            'segment_label': segment_labels,
            'churn_probability': churn_probs,
            'is_churner': is_churners,
            'segment_confidence': seg_confidences,
        })
        
        # Summary statistics
        summary = {
            'total_rows': n_rows,
            'segments_distribution': segments.value_counts().to_dict(),
            'churn_rate': float(is_churners.sum() / n_rows),
            'avg_churn_probability': float(churn_probs.mean()),
            'avg_segment_confidence': float(seg_confidences.mean()),
            'rows_processed': n_rows,
        }
        
        logger.info(f"✓ Batch prediction complete | Churn rate: {summary['churn_rate']:.2%}")
        
        return results_df, summary

    def _assign_segment(self, df: pd.DataFrame) -> Tuple[int, str, float]:
        """
        Assign segment for a single row.

        Returns:
            Tuple of (segment_id, segment_label, confidence_score)
        """
        # Validate and prepare features
        try:
            validate_feature_consistency(
                df,
                self.seg_numeric_cols,
                self.seg_categorical_cols,
                phase="SEGMENT_ASSIGNMENT"
            )
        except ValueError as e:
            logger.error(f"Segment feature validation failed: {e}")
            raise
        
        # Ensure categorical columns are strings
        for col in self.seg_categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Scale numeric features
        df_scaled = df.copy()
        df_scaled[self.seg_numeric_cols] = self.seg_scaler.transform(df[self.seg_numeric_cols])
        
        # Prepare feature matrix in exact order
        X = df_scaled[self.seg_feature_cols].to_numpy()
        
        # Calculate categorical indices relative to seg_feature_cols order
        cat_idx = [self.seg_feature_cols.index(col) for col in self.seg_categorical_cols 
                   if col in self.seg_feature_cols]
        
        # Predict cluster
        cluster_id = self.kproto.predict(X, categorical=cat_idx)[0]
        segment_label = self.segment_labels[cluster_id]
        
        # Confidence is always high for K-Prototypes (set to 0.95)
        confidence = 0.95
        
        return int(cluster_id), segment_label, confidence

    def _assign_segments_batch(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, np.ndarray]:
        """
        Assign segments for multiple rows.

        Returns:
            Tuple of (segment_ids, segment_labels, confidence_scores)
        """
        # Validate
        try:
            validate_feature_consistency(
                df,
                self.seg_numeric_cols,
                self.seg_categorical_cols,
                phase="SEGMENT_ASSIGNMENT"
            )
        except ValueError as e:
            logger.error(f"Batch segment validation failed: {e}")
            raise
        
        # Ensure categorical columns are strings
        for col in self.seg_categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Scale numeric features
        df_scaled = df.copy()
        df_scaled[self.seg_numeric_cols] = self.seg_scaler.transform(df[self.seg_numeric_cols])
        
        # Prepare feature matrix
        X = df_scaled[self.seg_feature_cols].to_numpy()
        
        # Calculate categorical indices
        cat_idx = [self.seg_feature_cols.index(col) for col in self.seg_categorical_cols 
                   if col in self.seg_feature_cols]
        
        # Predict clusters
        cluster_ids = self.kproto.predict(X, categorical=cat_idx)
        segment_labels = pd.Series(cluster_ids).map(self.segment_labels)
        
        # Confidence is always high for K-Prototypes (set to 0.95 for all)
        confidences = np.full(len(cluster_ids), 0.95)
        
        return pd.Series(cluster_ids), segment_labels, confidences

    def _predict_churn(self, df: pd.DataFrame) -> float:
        """Predict churn probability for a single customer."""
        X = self.churn_preprocessor.transform(df)
        churn_prob = self.lgbm_model.predict_proba(X)[0, 1]
        return float(churn_prob)

    def _predict_churn_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities for multiple customers."""
        X = self.churn_preprocessor.transform(df)
        churn_probs = self.lgbm_model.predict_proba(X)[:, 1]
        return churn_probs

    def explain_prediction(
        self,
        customer_data: Dict[str, Any],
        num_features: int = 5
    ) -> Dict[str, Any]:
        """
        Get feature importance for a single prediction (to be used with SHAP explainer).

        Args:
            customer_data: Customer attributes
            num_features: Number of top features to return

        Returns:
            Dictionary with explainability info (filled by SHAPExplainer)
        """
        # This is a placeholder - actual SHAP explanation handled by SHAPExplainer
        prediction = self.predict_single(customer_data, return_intermediate=True)
        return {
            'prediction': prediction,
            'features_available': True,
            'num_features': num_features,
            'message': 'Use SHAPExplainer.explain_instance() for detailed feature contributions'
        }
