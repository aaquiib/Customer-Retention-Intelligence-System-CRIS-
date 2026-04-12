"""Main pipeline orchestration - runs complete ML workflow."""

import logging

from src.churn.train import train_churn_model
from src.config import load_config
from src.data import load_raw_data, preprocess_data
from src.features import engineer_features
from src.features.build_features import build_features
from src.segmentation import assign_segments, train_segmentation_model
from src.utils import load_csv, load_json, save_csv, setup_logging

logger = logging.getLogger(__name__)


def _log_metrics_summary(cfg: dict) -> None:
    """
    Log a summary of saved evaluation metrics.

    Args:
        cfg: Configuration dictionary with paths
    """
    try:
        metrics_json_path = f"{cfg['models']['churn_dir']}metrics_latest.json"
        metrics = load_json(metrics_json_path)

        logger.info("\n" + "=" * 80)
        logger.info("CHURN MODEL EVALUATION METRICS SUMMARY")
        logger.info("=" * 80)

        # Print metrics by split
        for split_name, split_metrics in metrics.get('split_metrics', {}).items():
            logger.info(f"\n{split_name.upper()} SET:")
            logger.info(f"  Precision: {split_metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall:    {split_metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score:  {split_metrics.get('f1', 0):.4f}")
            logger.info(f"  Accuracy:  {split_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  ROC-AUC:   {split_metrics.get('roc_auc', 0):.4f}")

            if 'confusion_matrix' in split_metrics:
                cm = split_metrics['confusion_matrix']
                logger.info(f"  Confusion Matrix: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")

        logger.info(f"\nModel Configuration:")
        model_cfg = metrics.get('model_config', {})
        logger.info(f"  Best Threshold:   {model_cfg.get('best_threshold', 0):.4f}")
        logger.info(f"  Metric Optimized: {model_cfg.get('metric_optimized', 'N/A')}")
        logger.info(f"  N Estimators:     {model_cfg.get('n_estimators', 0)}")
        logger.info(f"  Random Seed:      {model_cfg.get('random_seed', 0)}")

        logger.info(f"\nMetrics saved to:")
        logger.info(f"  - JSON: {metrics_json_path}")
        logger.info(f"  - CSV:  {cfg['models']['churn_dir']}metrics_history.csv")
        logger.info("=" * 80 + "\n")

    except Exception as e:
        logger.warning(f"Could not load metrics summary: {e}")


def run_full_pipeline() -> None:
    """
    Run complete pipeline: data → features → segmentation → churn.

    Steps:
    1. Load configuration
    2. Load and preprocess raw data
    3. Engineer features
    4. Train segmentation model
    5. Assign segments
    6. Train churn model
    """
    # Setup
    cfg = load_config()
    setup_logging(cfg['logging'])

    logger.info("=" * 80)
    logger.info("CHURN-SEGMENTATION MLOPS PIPELINE")
    logger.info("=" * 80)

    # Phase 1: Data Loading & Preprocessing

    logger.info("\n[PHASE 1] DATA PREPROCESSING")
    logger.info("-" * 80)
    df_raw = load_raw_data(cfg['data']['raw_csv_path'])
    df_preprocessed = preprocess_data(df_raw, cfg)
    save_csv(df_preprocessed, cfg['data']['processed_csv_path'])
    logger.info(f"✓ Preprocessed data saved")

    # Phase 2: Feature Engineering

    logger.info("\n[PHASE 2] FEATURE ENGINEERING")
    logger.info("-" * 80)
    df_engineered = engineer_features(df_preprocessed, cfg)
    
    # Select segmentation features in EXACT order
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
    logger.info(f"✓ Engineered features saved | Shape: {df_segmentation.shape}")

    # Phase 3: Segmentation Training

    logger.info("\n[PHASE 3] SEGMENTATION MODEL TRAINING")
    logger.info("-" * 80)
    kproto, scaler, cat_idx, metadata = train_segmentation_model(df_segmentation, cfg)
    logger.info(f"✓ Segmentation model trained and saved")

    # Phase 4: Segment Assignment
    logger.info("\n[PHASE 4] SEGMENT ASSIGNMENT")
    logger.info("-" * 80)
    segments = assign_segments(df_segmentation, cfg)
    
    # Add segment columns to processed data for churn modeling
    df_processed = load_csv(cfg['data']['processed_csv_path'])
    df_processed['segment'] = segments['segment']
    df_processed['segment_label'] = segments['segment_label']
    
    # Save segmentation results
    save_csv(
        df_processed,
        cfg['data']['processed_csv_path'].replace('processed_df', 'df_with_segment_labels')
    )
    logger.info(f"✓ Segments assigned and saved")
    logger.info(f"\nSegment distribution:")
    for label, count in segments['segment_label'].value_counts().items():
        logger.info(f"  {label}: {count} customers")

    # Phase 5: Churn Model Training
    logger.info("\n[PHASE 5] CHURN MODEL TRAINING")
    logger.info("-" * 80)
    df_with_segment_labels = load_csv(
        cfg['data']['processed_csv_path'].replace('processed_df', 'df_with_segment_labels')
    )
    lgbm_model, preprocessor, opt_threshold, threshold_meta = train_churn_model(
        df_with_segment_labels,
        cfg
    )
    logger.info(f"✓ Churn model trained and saved")
    logger.info(f"  Test ROC-AUC: {threshold_meta['test_roc_auc']:.4f}")
    logger.info(f"  Optimal threshold: {opt_threshold:.4f}")

    # Log metrics summary
    _log_metrics_summary(cfg)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nArtifacts saved to:")
    logger.info(f"  - Data: {cfg['data']['processed_csv_path']}")
    logger.info(f"  - Segmentation model: {cfg['models']['segmentation_dir']}")
    logger.info(f"  - Churn model: {cfg['models']['churn_dir']}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_pipeline()
