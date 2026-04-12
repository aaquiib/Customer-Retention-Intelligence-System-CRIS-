"""Data preprocessing module for cleaning and transforming raw data."""

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess raw customer churn data.

    Cleaning steps:
    - Drop columns
    - Convert TotalCharges to numeric, handle NaN
    - Drop rows with tenure == 0
    - Convert SeniorCitizen 0/1 to 'yes'/'no'
    - Convert Churn 'Yes'/'No' to 1/0

    Args:
        df: Raw DataFrame from load_raw_data()
        cfg: Configuration dictionary with preprocessing config

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    logger.info(f"Starting preprocessing | Input shape: {df.shape}")

    # Step 1: Drop columns
    drop_cols = cfg['preprocessing'].get('drop_columns', [])
    if drop_cols:
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        logger.info(f"Dropped columns: {drop_cols} | Shape: {df.shape}")

    # Step 2: Convert TotalCharges to numeric and handle NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_count = df['TotalCharges'].isna().sum()
        if missing_count > 0:
            logger.warning(f"TotalCharges has {missing_count} NaN values")
            # Drop rows with missing TotalCharges
            df = df.dropna(subset=['TotalCharges'])
            logger.info(f"Dropped {missing_count} rows with missing TotalCharges | Shape: {df.shape}")

    # Step 3: Drop rows where tenure == 0 (typically new customers with missing data)
    if 'tenure' in df.columns:
        tenure_zero_count = (df['tenure'] == 0).sum()
        if tenure_zero_count > 0:
            df = df[df['tenure'] != 0]
            logger.info(f"Dropped {tenure_zero_count} rows with tenure==0 | Shape: {df.shape}")

    # Step 4: Convert SeniorCitizen from 0/1 to categorical
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'yes', 0: 'no'})
        logger.info("Converted SeniorCitizen to 'yes'/'no'")

    # Step 5: Convert Churn from 'Yes'/'No' to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        logger.info("Converted Churn to 1/0 (1=Churned, 0=Retained)")

    logger.info(f"Preprocessing complete | Output shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Entry point for testing
    from src.config import load_config
    from src.utils import setup_logging
    from src.data.ingest import load_raw_data

    cfg = load_config()
    setup_logging(cfg['logging'])

    # Load and preprocess
    df_raw = load_raw_data(cfg['data']['raw_csv_path'])
    df_cleaned = preprocess_data(df_raw, cfg)

    # Save preprocessed data
    from src.utils import save_csv
    save_csv(df_cleaned, cfg['data']['processed_csv_path'])
    print(f"\n✓ Preprocessing complete | Saved to {cfg['data']['processed_csv_path']}")
