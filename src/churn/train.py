"""Churn prediction model training using LightGBM."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_json, save_model

logger = logging.getLogger(__name__)


def train_churn_model(
    df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[LGBMClassifier, ColumnTransformer, float, Dict[str, Any]]:
    """
    Train LightGBM churn prediction model with threshold optimization.

    Steps:
    1. Prepare features (X, y) - drop Churn and segment_label columns
    2. Split data: 70% train | 15% val | 15% test
    3. Preprocess: StandardScaler (numeric) + OneHotEncoder (categorical)
    4. Train LightGBM on combined train+val set
    5. Optimize decision threshold via F1-score on test set

    Args:
        df: DataFrame with engineered features and segment labels
        cfg: Configuration dictionary with churn_modeling config

    Returns:
        Tuple of (lgbm_model, preprocessor, optimal_threshold, metadata_dict)
    """
    df = df.copy()
    logger.info(f"Starting LightGBM training | Input shape: {df.shape}")

    churn_cfg = cfg['churn_modeling']

    # ─────────────────────────────────────────────────────────────────
    # PREPARE FEATURES
    # ─────────────────────────────────────────────────────────────────

    # Drop target and segment labels
    X = df.drop(columns=['Churn', 'segment_label'], errors='ignore')
    y = df['Churn']

    logger.info(f"Feature shape: {X.shape} | Target shape: {y.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # Identify numeric and categorical columns
    num_cols = churn_cfg.get('numeric_columns', [])
    num_cols = [col for col in num_cols if col in X.columns]
    cat_cols = churn_cfg.get('categorical_columns', [])
    cat_cols = [col for col in cat_cols if col in X.columns]

    logger.info(f"Numeric columns: {len(num_cols)} | Categorical columns: {len(cat_cols)}")

    # Keep only the configured features
    feature_cols = num_cols + cat_cols
    X = X[feature_cols]
    logger.info(f"Feature shape after filtering: {X.shape}")

    # Ensure categorical columns are strings
    for col in cat_cols:
        X[col] = X[col].astype(str)

    # ─────────────────────────────────────────────────────────────────
    # TRAIN / VALIDATION / TEST SPLIT
    # ─────────────────────────────────────────────────────────────────

    train_size = churn_cfg.get('train_size', 0.70)
    val_size = churn_cfg.get('val_size', 0.15)
    random_seed = churn_cfg.get('random_seed', 42)
    stratified = churn_cfg.get('stratified', True)

    # Split: temp (85%) | test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=1 - train_size,
        stratify=y if stratified else None,
        random_state=random_seed
    )

    # Split temp: train (70%) | val (15%) of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (train_size),  # Proportional split
        stratify=y_temp if stratified else None,
        random_state=random_seed
    )

    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # ─────────────────────────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────────────────────────

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ])

    # Fit ONLY on train split
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    logger.info(f"Encoded shapes: train {X_train_enc.shape} | val {X_val_enc.shape} | test {X_test_enc.shape}")

    # Combine train+val for final training
    X_trainval_enc = np.vstack([X_train_enc, X_val_enc])
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────
    # TRAIN LIGHTGBM
    # ─────────────────────────────────────────────────────────────────

    lgbm_params = churn_cfg.get('lgbm_hyperparams', {})
    logger.info(f"Training LightGBM with {lgbm_params.get('n_estimators', 100)} estimators")

    lgbm_model = LGBMClassifier(
        **lgbm_params,
        random_state=random_seed
    )

    lgbm_model.fit(X_trainval_enc, y_trainval)
    logger.info(f"LightGBM training complete | Trees: {lgbm_model.n_estimators_}")

    # ─────────────────────────────────────────────────────────────────
    # THRESHOLD OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────

    from sklearn.metrics import precision_recall_curve, f1_score

    y_proba = lgbm_model.predict_proba(X_test_enc)[:, 1]

    # Find optimal threshold via F1-score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")

    # ─────────────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────────────

    models_dir = cfg['models']['churn_dir']

    # Save model
    save_model(lgbm_model, f"{models_dir}lgbm_churn_model.pkl")

    # Save preprocessor
    save_model(preprocessor, f"{models_dir}preprocessor.pkl")

    # Save threshold metadata
    from sklearn.metrics import roc_auc_score
    test_roc_auc = roc_auc_score(y_test, y_proba)

    threshold_meta = {
        'best_threshold': float(optimal_threshold),
        'default_threshold': 0.5,
        'metric_optimized': 'f1',
        'test_roc_auc': float(test_roc_auc),
        'n_estimators': lgbm_model.n_estimators_,
        'random_seed': random_seed
    }
    save_json(threshold_meta, f"{models_dir}threshold_meta.json")

    logger.info(f"All artifacts saved to {models_dir}")

    return lgbm_model, preprocessor, optimal_threshold, threshold_meta


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import load_csv, setup_logging

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load data with segment labels
    df_with_segments = load_csv(cfg['data']['processed_csv_path'])

    # Train model
    lgbm_model, preprocessor, opt_threshold, metadata = train_churn_model(df_with_segments, cfg)

    print(f"\n✓ Churn model trained successfully")
    print(f"  - Test ROC-AUC: {metadata['test_roc_auc']:.4f}")
    print(f"  - Optimal threshold: {opt_threshold:.4f}")
    print(f"  - Model saved to {cfg['models']['churn_dir']}")
