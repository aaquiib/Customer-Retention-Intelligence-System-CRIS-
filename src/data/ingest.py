"""Data ingestion module for loading raw datasets."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw customer churn dataset from CSV.

    Args:
        filepath: Path to raw CSV file

    Returns:
        DataFrame with raw data

    Raises:
        FileNotFoundError: If CSV file not found
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded raw data from {filepath} | Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Raw data file not found: {filepath}") from e
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise


if __name__ == "__main__":
    # Entry point for testing
    import sys
    from src.config import load_config
    from src.utils import setup_logging

    cfg = load_config()
    setup_logging(cfg['logging'])

    df = load_raw_data(cfg['data']['raw_csv_path'])
    print(f"\n✓ Ingestion successful | Shape: {df.shape}")
