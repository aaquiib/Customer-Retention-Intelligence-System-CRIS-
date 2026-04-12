"""Segment assignment using trained K-Prototypes model."""

import logging
from typing import Any, Dict

import pandas as pd

from src.utils import load_json, load_model, save_csv, validate_feature_consistency

logger = logging.getLogger(__name__)


def assign_segments(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Assign customer segments using previously trained K-Prototypes model.

    Loads saved model, scaler, and metadata. Validates feature consistency
    and assigns segment labels to each customer.

    Args:
        df: DataFrame with engineered features (must match training features)
        cfg: Configuration dictionary

    Returns:
        DataFrame with only 'segment' and 'segment_label' columns (7032 × 2)
    """
    df = df.copy()
    logger.info(f"Starting segment assignment | Input shape: {df.shape}")

    models_dir = cfg['models']['segmentation_dir']

    # ─────────────────────────────────────────────────────────────────
    # LOAD ARTIFACTS
    # ─────────────────────────────────────────────────────────────────

    kproto = load_model(f"{models_dir}kproto.pkl")
    scaler = load_model(f"{models_dir}scaler.pkl")
    cat_idx = load_json(f"{models_dir}catidx.json")
    feature_metadata = load_json(f"{models_dir}feature_metadata.json")
    segment_labels_raw = load_json(f"{models_dir}segment_labels.json")

    # Convert segment labels keys back to int
    segment_labels = {int(k): v for k, v in segment_labels_raw.items()}

    logger.info(f"Loaded model artifacts from {models_dir}")

    # ─────────────────────────────────────────────────────────────────
    # FEATURE VALIDATION
    # ─────────────────────────────────────────────────────────────────

    expected_num_cols = feature_metadata.get('numeric_columns', [])
    expected_cat_cols = feature_metadata.get('categorical_columns', [])
    seg_feature_cols = feature_metadata.get('segmentation_features', [])

    try:
        validate_feature_consistency(
            df,
            expected_num_cols,
            expected_cat_cols,
            phase="SEGMENT_ASSIGNMENT"
        )
        logger.info("✓ Feature consistency validation passed")
    except ValueError as e:
        logger.error(f"Feature validation failed: {str(e)}")
        raise

    # ─────────────────────────────────────────────────────────────────
    # DATA PREPARATION
    # ─────────────────────────────────────────────────────────────────

    # Ensure categorical columns are strings
    for col in expected_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Scale numeric columns using saved scaler
    df[expected_num_cols] = scaler.transform(df[expected_num_cols])
    logger.info(f"Scaled {len(expected_num_cols)} numeric features")

    # ─────────────────────────────────────────────────────────────────
    # PREDICT SEGMENTS
    # ─────────────────────────────────────────────────────────────────

    # Create feature matrix with features in EXACT order
    X = df[seg_feature_cols].to_numpy()
    logger.info(f"Prediction data shape: {X.shape}")

    # Calculate categorical indices RELATIVE to seg_feature_cols order
    cat_idx = [seg_feature_cols.index(col) for col in expected_cat_cols if col in seg_feature_cols]

    cluster_labels = kproto.predict(X, categorical=cat_idx)
    logger.info(f"Segments assigned | Distribution: {pd.Series(cluster_labels).value_counts().to_dict()}")

    # ─────────────────────────────────────────────────────────────────
    # CREATE SEGMENT RESULTS
    # ─────────────────────────────────────────────────────────────────

    # Return only segment columns (not full engineered features)
    segment_map = pd.Series(cluster_labels).map(segment_labels)
    segments_df = pd.DataFrame({
        'segment': cluster_labels,
        'segment_label': segment_map.values
    })

    logger.info(f"Segment assignment complete | Output shape: {segments_df.shape}")

    return segments_df


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import load_csv, setup_logging

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load engineered features for segmentation
    df_features = load_csv(cfg['data']['segmentation_features_path'])

    # Assign segments (returns only 'segment' and 'segment_label' columns)
    segments = assign_segments(df_features, cfg)

    # Display results
    print(f"\n✓ Segment assignment complete")
    print(f"  Output shape: {segments.shape}")
    print(f"\nSegment distribution:")
    print(segments['segment_label'].value_counts())
