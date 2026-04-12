"""Feature building pipeline - orchestrates preprocessing and engineering."""

import logging
from typing import Any, Dict

from src.config import get_config
from src.data import load_raw_data, preprocess_data
from src.features.engineering import engineer_features
from src.utils import save_csv, setup_logging

logger = logging.getLogger(__name__)


def build_features(cfg: Dict[str, Any]) -> None:
    """
    Build features: load raw → preprocess → engineer → save.

    Args:
        cfg: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("FEATURE BUILDING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load raw data
    df_raw = load_raw_data(cfg['data']['raw_csv_path'])

    # Step 2: Preprocess
    df_preprocessed = preprocess_data(df_raw, cfg)
    save_csv(df_preprocessed, cfg['data']['processed_csv_path'])

    # Step 3: Engineer features
    df_engineered = engineer_features(df_preprocessed, cfg)
    save_csv(df_engineered, cfg['data']['segmentation_features_path'])

    # Also save a copy for churn modeling (without segmentation labels)
    save_csv(df_engineered, cfg['data']['churn_features_path'])

    logger.info("=" * 80)
    logger.info("FEATURE BUILDING COMPLETE")
    logger.info(f"  - Processed: {cfg['data']['processed_csv_path']}")
    logger.info(f"  - Segmentation features: {cfg['data']['segmentation_features_path']}")
    logger.info(f"  - Churn features: {cfg['data']['churn_features_path']}")
    logger.info("=" * 80)


if __name__ == "__main__":
    cfg = get_config()
    setup_logging(cfg['logging'])
    build_features(cfg)
