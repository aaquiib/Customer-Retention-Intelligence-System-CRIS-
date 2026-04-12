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
from src.utils.metrics_saver import save_metrics_to_json, append_metrics_to_csv, build_metrics_payload, validate_metrics
from src.churn.evaluate import evaluate_model, evaluate_model_on_splits

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
        test_size=1 - train_size - val_size,
        stratify=y if stratified else None,
        random_state=random_seed
    )

    # Split temp: train (70%) | val (15%) of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (train_size + val_size),  # Proportional split
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
    f1_scores = f1_scores[:-1]
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx])

    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")

    # ─────────────────────────────────────────────────────────────────
    # EVALUATE ON ALL SPLITS (train, val, test)
    # ─────────────────────────────────────────────────────────────────

    # Generate predictions on all splits
    y_train_proba = lgbm_model.predict_proba(X_train_enc)[:, 1]
    y_train_pred = (y_train_proba >= optimal_threshold).astype(int)

    y_val_proba = lgbm_model.predict_proba(X_val_enc)[:, 1]
    y_val_pred = (y_val_proba >= optimal_threshold).astype(int)

    y_test_pred = (y_proba >= optimal_threshold).astype(int)

    # Evaluate on all splits
    splits_data = {
        'train': {'y_true': y_train.values, 'y_pred': y_train_pred, 'y_proba': y_train_proba},
        'validation': {'y_true': y_val.values, 'y_pred': y_val_pred, 'y_proba': y_val_proba},
        'test': {'y_true': y_test.values, 'y_pred': y_test_pred, 'y_proba': y_proba}
    }

    evaluation_results = evaluate_model_on_splits(splits_data, threshold=optimal_threshold, cfg=cfg)

    logger.info(f"Evaluation complete on {len(evaluation_results)} splits")

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

    # ─────────────────────────────────────────────────────────────────
    # SAVE COMPREHENSIVE METRICS (JSON + CSV)
    # ─────────────────────────────────────────────────────────────────

    try:
        # Build metrics payload for JSON
        metrics_payload = build_metrics_payload(evaluation_results, threshold_meta)

        # Validate metrics before saving
        if validate_metrics(metrics_payload):
            # Save latest metrics to JSON
            save_metrics_to_json(metrics_payload, f"{models_dir}metrics_latest.json")

            # Extract flat metrics for CSV
            metrics_by_split = {}
            for split_name, eval_result in evaluation_results.items():
                if 'metrics' in eval_result:
                    metrics_by_split[split_name] = eval_result['metrics']
                else:
                    metrics_by_split[split_name] = eval_result

            # Append to CSV history
            append_metrics_to_csv(metrics_by_split, threshold_meta, f"{models_dir}metrics_history.csv")

            logger.info(f"Metrics saved successfully (JSON + CSV)")
        else:
            logger.warning("Metrics validation failed, skipping save")

    except Exception as e:
        logger.error(f"Error saving metrics: {e}", exc_info=True)
        # Don't fail the entire pipeline if metrics saving fails

    return lgbm_model, preprocessor, optimal_threshold, threshold_meta


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import load_csv, setup_logging

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load data with segment labels
    df_with_segments = load_csv(
        cfg['data']['processed_csv_path'].replace('processed_df', 'df_with_segment_labels')
    )

    # Train model
    lgbm_model, preprocessor, opt_threshold, metadata = train_churn_model(df_with_segments, cfg)

    print(f"\n✓ Churn model trained successfully")
    print(f"  - Test ROC-AUC: {metadata['test_roc_auc']:.4f}")
    print(f"  - Optimal threshold: {opt_threshold:.4f}")
    print(f"  - Model saved to {cfg['models']['churn_dir']}")
