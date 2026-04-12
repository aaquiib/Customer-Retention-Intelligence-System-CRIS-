"""Feature engineering module for creating business-relevant features."""

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Engineer features for customer segmentation and churn modeling.

    Features created:
    - Billing/Value: avg_monthly_spend, charge_gap, is_high_value
    - Tenure: tenure_band (categorical)
    - Services: streaming_count, security_count
    - Risk flags: payment_electronic_check, month_to_month_paperless,
                  no_support_services, is_isolated, fiber_no_security,
                  no_internet_services

    Args:
        df: Preprocessed DataFrame (output of preprocess_data)
        cfg: Configuration dictionary with feature_engineering config

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    logger.info(f"Starting feature engineering | Input shape: {df.shape}")

    fe_cfg = cfg['feature_engineering']

    # ─────────────────────────────────────────────────────────────────
    # BILLING / VALUE FEATURES
    # ─────────────────────────────────────────────────────────────────
    if all(col in df.columns for col in ['TotalCharges', 'tenure', 'MonthlyCharges']):
        # Average monthly spend over tenure
        df['avg_monthly_spend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

        # Gap between current and historical average
        df['charge_gap'] = df['MonthlyCharges'] - df['avg_monthly_spend']

        # High-value customer flag (compared to median)
        monthly_median = df['MonthlyCharges'].median()
        df['is_high_value'] = (df['MonthlyCharges'] > monthly_median).astype(int)
        logger.info(f"Created billing features (median threshold: ${monthly_median:.2f})")

    # ─────────────────────────────────────────────────────────────────
    # TENURE BANDS (CATEGORICAL)
    # ─────────────────────────────────────────────────────────────────
    if 'tenure' in df.columns:
        tenure_bins = fe_cfg.get('tenure_bins', [0, 12, 36, 72])
        tenure_labels = fe_cfg.get('tenure_labels', ['0-12', '12-36', '36+'])

        df['tenure_band'] = pd.cut(
            df['tenure'],
            bins=tenure_bins,
            labels=tenure_labels,
            right=True
        )
        df['tenure_band'] = df['tenure_band'].astype(str)
        logger.info(f"Created tenure bands: {tenure_labels}")

    # ─────────────────────────────────────────────────────────────────
    # SERVICE USAGE COUNTS
    # ─────────────────────────────────────────────────────────────────
    streaming_services = fe_cfg.get('streaming_services', [])
    if streaming_services and all(col in df.columns for col in streaming_services):
        df['streaming_count'] = (
            (df[streaming_services] == 'Yes') | (df[streaming_services] == 'yes')
        ).sum(axis=1)
        logger.info(f"Created streaming_count from {streaming_services}")

    security_services = fe_cfg.get('security_services', [])
    if security_services and all(col in df.columns for col in security_services):
        df['security_count'] = (
            (df[security_services] == 'Yes') | (df[security_services] == 'yes')
        ).sum(axis=1)
        logger.info(f"Created security_count from {security_services}")

    # ─────────────────────────────────────────────────────────────────
    # RISK / VULNERABILITY FLAGS
    # ─────────────────────────────────────────────────────────────────

    # Payment method risk
    if 'PaymentMethod' in df.columns:
        df['payment_electronic_check'] = (
            df['PaymentMethod'] == 'Electronic check'
        ).astype(int)
        logger.info("Created payment_electronic_check flag")

    # Contract + Billing risk combo
    if 'Contract' in df.columns and 'PaperlessBilling' in df.columns:
        df['month_to_month_paperless'] = (
            (df['Contract'] == 'Month-to-month') & (df['PaperlessBilling'] == 'Yes')
        ).astype(int)
        logger.info("Created month_to_month_paperless flag")

    # Support vulnerability
    if 'TechSupport' in df.columns and 'OnlineSecurity' in df.columns:
        df['no_support_services'] = (
            (df['TechSupport'] == 'No') & (df['OnlineSecurity'] == 'No')
        ).astype(int)
        logger.info("Created no_support_services flag")

    # Social isolation
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['is_isolated'] = (
            (df['Partner'] == 'No') & (df['Dependents'] == 'No')
        ).astype(int)
        logger.info("Created is_isolated flag")

    # Internet service vulnerabilities
    if 'InternetService' in df.columns and 'OnlineSecurity' in df.columns:
        df['fiber_no_security'] = (
            (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No')
        ).astype(int)
        logger.info("Created fiber_no_security flag")

    if 'InternetService' in df.columns:
        df['no_internet_services'] = (df['InternetService'] == 'No').astype(int)
        logger.info("Created no_internet_services flag")

    logger.info(f"Feature engineering complete | Output shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import setup_logging, load_csv, save_csv

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load preprocessed data
    df_preprocessed = load_csv(cfg['data']['processed_csv_path'])
    df_engineered = engineer_features(df_preprocessed, cfg)

    # Save engineered features
    save_csv(df_engineered, cfg['data']['segmentation_features_path'])
    print(f"\n✓ Feature engineering complete | Saved to {cfg['data']['segmentation_features_path']}")
