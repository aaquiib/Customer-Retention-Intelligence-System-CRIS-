"""SHAP-based explainability for churn predictions."""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from inference.pipeline import InferencePipeline
from src.config import get_config
from src.data.preprocess import preprocess_data
from src.features.engineering import engineer_features
from src.utils import load_csv

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explanations for LightGBM churn model.

    Provides:
    - Global feature importance (average SHAP values across background data)
    - Per-instance explanations (SHAP values for a single prediction)
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        background_sample_path: Optional[str] = None,
        explainer_type: str = "tree",
        n_background_samples: int = 200
    ):
        """
        Initialize SHAP explainer.

        Args:
            pipeline: InferencePipeline instance with loaded models
            background_sample_path: Path to background data for SHAP (CSV)
                                   If None, uses random samples from training data
            explainer_type: 'tree' (fast, for LightGBM) or 'kernel' (model-agnostic, slow)
            n_background_samples: Number of background samples to use for explanation
        """
        self.pipeline = pipeline
        self.explainer_type = explainer_type
        self.n_background_samples = min(n_background_samples, 500)  # Cap at 500
        self.explainer: Optional[shap.Explainer] = None
        self.background_X: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self._shap_values_global: Optional[np.ndarray] = None
        self._expected_value: Optional[float] = None
        
        self._initialize_explainer(background_sample_path)
        logger.info(f"✓ SHAPExplainer initialized (type: {explainer_type})")

    def _initialize_explainer(self, background_sample_path: Optional[str]) -> None:
        """Initialize SHAP explainer with background data."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Explainability will be limited.")
            self.explainer = None
            self._shap_values_global = None
            return
        
        try:
            # Load or create background samples
            if background_sample_path:
                logger.info(f"Loading background data from {background_sample_path}")
                df_background = load_csv(background_sample_path)
            else:
                logger.info("Loading background data from training data...")
                cfg = self.pipeline.cfg
                churn_features_path = cfg['data'].get('churn_features_path', 'data/processed/churn_features.csv')
                
                try:
                    df_background = load_csv(churn_features_path)
                    # Remove Churn target if present
                    if 'Churn' in df_background.columns:
                        df_background = df_background.drop('Churn', axis=1)
                    logger.info(f"Loaded {len(df_background)} samples from {churn_features_path}")
                except Exception as e:
                    logger.warning(f"Could not load from {churn_features_path}: {e}")
                    df_background = self._create_synthetic_background()
            
            # Shuffle and sample to n_background_samples
            if len(df_background) > self.n_background_samples:
                df_background = df_background.sample(n=self.n_background_samples, random_state=42)
            
            # Ensure 'segment' column exists before preprocessing (required by churn_preprocessor)
            if 'segment' not in df_background.columns:
                df_background['segment'] = '0'  # Default segment value
            
            # Transform through the preprocessor to get encoded values
            try:
                logger.info("Transforming background data with preprocessor...")
                df_background = self.pipeline.churn_preprocessor.transform(df_background)
            except Exception as e:
                # If it fails, the data might be in unexpected format
                logger.warning(f"Could not apply preprocessor: {e}. Using data as-is.")
            
            # Convert to numpy array for SHAP
            if isinstance(df_background, pd.DataFrame):
                df_background = df_background.to_numpy()
            
            self.background_X = df_background[:self.n_background_samples]
            self.feature_names = self._get_feature_names()
            
            # Create SHAP explainer
            if self.explainer_type == "tree":
                logger.info("Creating TreeExplainer (LightGBM)...")
                self.explainer = shap.TreeExplainer(self.pipeline.lgbm_model)
                self._expected_value = self.explainer.expected_value
            else:  # kernel
                logger.info(f"Creating KernelExplainer with {len(self.background_X)} background samples...")
                self.explainer = shap.KernelExplainer(
                    self.pipeline.lgbm_model.predict_proba,
                    shap.sample(self.background_X, min(100, len(self.background_X)))
                )
                self._expected_value = self.explainer.expected_value
            
            logger.info("✓ SHAP explainer initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}", exc_info=True)
            raise

    def _create_synthetic_background(self) -> pd.DataFrame:
        """Create synthetic background data for SHAP baseline from real training data."""
        # Try to load real background data from processed features, fall back to synthetic
        try:
            cfg = self.pipeline.cfg
            churn_features_path = cfg['data'].get('churn_features_path', 'data/processed/churn_features.csv')
            logger.info(f"Attempting to load background data from {churn_features_path}")
            df_background = load_csv(churn_features_path)
            
            # Shuffle and take top n_samples
            df_background = df_background.sample(min(len(df_background), self.n_background_samples), random_state=42)
            logger.info(f"Loaded {len(df_background)} real background samples from training data")
            return df_background
        except Exception as e:
            logger.warning(f"Could not load real background data: {e}. Creating synthetic background...")
        
        # Fallback: Create simple synthetic background with common values
        # Using only values that are likely in the preprocessor's training categories
        n_samples = self.n_background_samples
        
        background_data = {
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice(['yes', 'no'], n_samples),  # After preprocessing
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check'], n_samples),  # Only common values
            'segment': str(np.random.randint(0, 4))  # Single segment value
        }
        
        return pd.DataFrame(background_data)

    def _get_feature_names(self) -> List[str]:
        """Get feature names from preprocessor."""
        try:
            # Try to get feature names from ColumnTransformer
            preprocessor = self.pipeline.churn_preprocessor
            
            # Get feature names from all transformers
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                feature_names.extend(columns)
            
            return feature_names
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}. Using generic names.")
            return [f"feature_{i}" for i in range(self.background_X.shape[1])]

    def get_global_importance(
        self,
        top_n: int = 10,
        force_compute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute global feature importance across background data.

        Args:
            top_n: Number of top features to return
            force_compute: If True, recompute even if cached

        Returns:
            Dictionary with:
                - top_features: List of dicts with feature name and importance
                - explainer_type: Type of explainer used
                - sample_size: Number of background samples used
        """
        try:
            # Compute SHAP values for background data (if not cached)
            if self._shap_values_global is None or force_compute:
                logger.info("Computing global SHAP values...")
                
                if self.explainer_type == "tree":
                    self._shap_values_global = self.explainer.shap_values(self.background_X)
                    # For binary classification, take positive class SHAP values
                    if isinstance(self._shap_values_global, list):
                        self._shap_values_global = self._shap_values_global[1]
                else:
                    # For KernelExplainer
                    self._shap_values_global = self.explainer.shap_values(self.background_X)
                
                logger.info(f"✓ Global SHAP values computed | Shape: {self._shap_values_global.shape}")
            
            # Compute mean absolute SHAP values
            mean_abs_shap = np.abs(self._shap_values_global).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                importance = float(mean_abs_shap[idx])
                
                # Determine sign (positive or negative impact on churn)
                mean_shap = self._shap_values_global[:, idx].mean()
                sign = "positive" if mean_shap > 0 else "negative"
                
                top_features.append({
                    'feature': feature_name,
                    'importance': importance,
                    'sign': sign,
                    'mean_shap': float(mean_shap)
                })
            
            return {
                'top_features': top_features,
                'explainer_type': self.explainer_type,
                'sample_size': len(self.background_X)
            }
        
        except Exception as e:
            logger.error(f"Error computing global importance: {e}", exc_info=True)
            raise

    def explain_instance(
        self,
        customer_data: Dict[str, Any],
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            customer_data: Customer features dict
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with:
                - prediction: Predicted churn probability
                - top_features: List of features with SHAP contributions
                - base_value: Model baseline
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            cfg = self.pipeline.cfg
            
            logger.info(f"Starting instance explanation | Initial columns: {df.columns.tolist()}")
            
            # Apply preprocessing and feature engineering
            df = preprocess_data(df, cfg)
            df = engineer_features(df, cfg)
            
            # Add segment column (required by churn_preprocessor)
            if 'segment' not in df.columns:
                df['segment'] = '0'
            
            # Log the columns we have
            logger.info(f"Engineered feature columns: {df.columns.tolist()}")
            logger.info(f"DataFrame shape after engineering: {df.shape}")
            
            # Apply the categorical/numerical transformation
            try:
                X = self.pipeline.churn_preprocessor.transform(df)
                logger.info(f"Preprocessor output shape: {X.shape}")
            except Exception as e:
                logger.error(f"Preprocessor transform failed: {e}")
                # If preprocessor fails, we need to skip instance explanation
                raise ValueError(f"Could not preprocess customer data for SHAP: {e}")
            
            # Compute SHAP values
            if self.explainer_type == "tree":
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class for binary
            else:
                shap_values = self.explainer.shap_values(X)
            
            shap_values = shap_values[0]  # Single instance
            
            # Get prediction
            churn_prob = self.pipeline.lgbm_model.predict_proba(X)[0, 1]
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]
            
            top_features = []
            for idx in top_indices:
                feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                
                # Try to get original feature value
                try:
                    if hasattr(self.pipeline.churn_preprocessor, 'get_feature_names_out'):
                        feature_names_out = self.pipeline.churn_preprocessor.get_feature_names_out()
                        feature_name = feature_names_out[idx] if idx < len(feature_names_out) else feature_name
                except:
                    pass
                
                shap_val = float(shap_values[idx])
                
                top_features.append({
                    'feature': feature_name,
                    'shap_value': shap_val,
                    'contribution': 'increases churn' if shap_val > 0 else 'decreases churn'
                })
            
            logger.info(f"Instance explanation computed | Prediction: {churn_prob:.4f}")
            
            return {
                'prediction': float(churn_prob),
                'top_features': top_features,
                'base_value': float(self._expected_value) if self._expected_value is not None else 0.0,
                'explainer_type': self.explainer_type
            }
        
        except Exception as e:
            logger.error(f"Error computing instance explanation: {e}", exc_info=True)
            raise

    def plot_force_plot(self, customer_data: Dict[str, Any]) -> str:
        """
        Generate SHAP force plot visualization (as HTML string).

        Args:
            customer_data: Customer features dict

        Returns:
            HTML string of the force plot
        """
        try:
            # Preprocess
            df = pd.DataFrame([customer_data])
            X = self.pipeline.churn_preprocessor.transform(df)
            
            # Compute SHAP values
            if self.explainer_type == "tree":
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = self.explainer.shap_values(X)
            
            shap_values = shap_values[0]
            
            # Create force plot
            force_plot = shap.force_plot(
                self._expected_value,
                shap_values,
                pd.DataFrame(X, columns=self.feature_names).iloc[0],
                matplotlib=False
            )
            
            return force_plot._repr_html_()
        
        except Exception as e:
            logger.error(f"Error creating force plot: {e}")
            return f"<p>Error generating force plot: {str(e)}</p>"
