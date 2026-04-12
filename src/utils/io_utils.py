"""I/O utilities for loading/saving data and models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded CSV: {filepath} | Shape: {df.shape}")
    return df


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Save DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved CSV: {filepath} | Shape: {df.shape}")


def load_model(filepath: str) -> Any:
    """Load joblib-serialized model."""
    model = joblib.load(filepath)
    logger.info(f"Loaded model: {filepath}")
    return model


def save_model(model: Any, filepath: str) -> None:
    """Save model using joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Saved model: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON: {filepath}")
    return data


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON: {filepath}")
