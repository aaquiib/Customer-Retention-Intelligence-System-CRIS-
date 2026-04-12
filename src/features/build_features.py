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
    
    # Step 4: Select only segmentation features (25 columns) in EXACT order
    seg_feature_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'tenure_band', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend',
        'charge_gap', 'is_high_value', 'PhoneService', 'MultipleLines',
        'InternetService', 'streaming_count', 'security_count', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'payment_electronic_check',
        'month_to_month_paperless', 'no_support_services', 'is_isolated',
        'fiber_no_security', 'no_internet_services'
    ]
    
    df_segmentation = df_engineered[seg_feature_cols]
    
    save_csv(df_segmentation, cfg['data']['segmentation_features_path'])
    logger.info(f"Selected {len(seg_feature_cols)} segmentation features | Shape: {df_segmentation.shape}")

    # Also save full engineered for churn modeling (includes all 32 features)
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
