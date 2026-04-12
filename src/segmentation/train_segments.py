"""Segmentation model training using K-Prototypes clustering."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

from src.utils import save_json, save_model

logger = logging.getLogger(__name__)


def train_segmentation_model(
    df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[KPrototypes, StandardScaler, List[int], Dict[str, Any]]:
    """
    Train K-Prototypes clustering model on engineered features.

    Args:
        df: DataFrame with engineered features
        cfg: Configuration dictionary with segmentation config

    Returns:
        Tuple of (kprototypes_model, scaler, categorical_indices, feature_metadata)
    """
    df = df.copy()
    logger.info(f"Starting K-Prototypes training | Input shape: {df.shape}")

    seg_cfg = cfg['segmentation']

    # Extract segmentation features in EXACT order from config
    seg_feature_cols = seg_cfg.get('segmentation_features', [])
    seg_feature_cols = [col for col in seg_feature_cols if col in df.columns]
    
    # Also get numeric and categorical for processing
    num_cols = seg_cfg.get('numeric_features', [])
    cat_cols = seg_cfg.get('categorical_features', [])
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]

    logger.info(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    # ─────────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # ─────────────────────────────────────────────────────────────────

    # Convert numeric columns to float and categorical columns to string
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # Scale numeric columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled {len(num_cols)} numeric features")

    # Prepare feature matrix with features in EXACT ORDER
    X = df[seg_feature_cols].to_numpy()
    logger.info(f"Feature matrix shape: {X.shape}")

    # Get categorical column indices (RELATIVE to seg_feature_cols order)
    cat_idx = [seg_feature_cols.index(col) for col in cat_cols if col in seg_feature_cols]
    logger.info(f"Categorical indices: {cat_idx}")

    # ─────────────────────────────────────────────────────────────────
    # K-PROTOTYPES TRAINING
    # ─────────────────────────────────────────────────────────────────

    n_clusters = seg_cfg.get('n_clusters', 4)
    init_method = seg_cfg.get('init_method', 'Cao')
    n_init = seg_cfg.get('n_init', 10)
    random_seed = seg_cfg.get('random_seed', 42)

    logger.info(f"Training K-Prototypes (k={n_clusters}, init={init_method}, n_init={n_init})")

    kproto = KPrototypes(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        verbose=0,
        random_state=random_seed
    )

    cluster_labels = kproto.fit_predict(X, categorical=cat_idx)
    logger.info(f"Training complete | Cluster cost: {kproto.cost_:.4f}")
    logger.info(f"Cluster distribution: {np.bincount(cluster_labels)}")

    # ─────────────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────────────

    models_dir = cfg['models']['segmentation_dir']

    # Save model
    save_model(kproto, f"{models_dir}kproto.pkl")

    # Save scaler
    save_model(scaler, f"{models_dir}scaler.pkl")

    # Save categorical indices
    save_json(cat_idx, f"{models_dir}catidx.json")

    # Save feature metadata (column names and order for reproducibility)
    feature_metadata = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "segmentation_features": seg_feature_cols,  # EXACT order for prediction
        "n_clusters": n_clusters,
        "random_seed": random_seed
    }
    save_json(feature_metadata, f"{models_dir}feature_metadata.json")

    # Save segment labels mapping
    segment_labels = seg_cfg.get('segment_labels', {})
    save_json({str(k): v for k, v in segment_labels.items()},
              f"{models_dir}segment_labels.json")

    logger.info(f"All artifacts saved to {models_dir}")

    return kproto, scaler, cat_idx, feature_metadata


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import load_csv, setup_logging

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load engineered features
    df_features = load_csv(cfg['data']['segmentation_features_path'])

    # Train model
    kproto, scaler, cat_idx, metadata = train_segmentation_model(df_features, cfg)

    print(f"\n✓ Segmentation model trained successfully")
    print(f"  - Model saved to {cfg['models']['segmentation_dir']}kproto.pkl")
