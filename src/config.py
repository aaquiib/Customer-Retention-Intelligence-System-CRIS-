"""Configuration loader for churn-segmentation MLOps pipeline."""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")

    # Validate required keys
    required_keys = ['data', 'preprocessing', 'feature_engineering',
                     'segmentation', 'churn_modeling', 'logging']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Create directories if they don't exist
    os.makedirs(config['data']['processed_csv_path'].rsplit('/', 1)[0], exist_ok=True)
    os.makedirs(config['models']['segmentation_dir'], exist_ok=True)
    os.makedirs(config['models']['churn_dir'], exist_ok=True)

    return config


_config_cache = None


def get_config() -> Dict[str, Any]:
    """Get global config (cached)."""
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache
