"""Utilities for saving and persisting evaluation metrics to JSON and CSV."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def save_metrics_to_json(
    metrics: Dict[str, Any],
    filepath: str,
    overwrite: bool = True
) -> None:
    """
    Save complete evaluation metrics to JSON file.

    Structure:
    {
        "timestamp": "ISO 8601 string",
        "split_metrics": {
            "train": {"precision": X, "recall": X, "f1": X, "accuracy": X, "roc_auc": X, "confusion_matrix": {...}},
            "validation": {...},
            "test": {...}
        },
        "model_config": {...}
    }

    Args:
        metrics: Dictionary with keys 'timestamp', 'split_metrics', 'model_config'
        filepath: Path to save JSON file
        overwrite: If True, overwrite existing file; if False, merge (not implemented)

    Returns:
        None
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved metrics to JSON: {filepath}")


def append_metrics_to_csv(
    metrics_by_split: Dict[str, Dict[str, float]],
    model_config: Dict[str, Any],
    filepath: str
) -> None:
    """
    Append evaluation metrics for all splits to CSV history file.

    Each split (train, val, test) becomes one row in the CSV.

    Columns:
        timestamp, split, precision, recall, f1, accuracy, roc_auc, best_threshold,
        metric_optimized, n_estimators, random_seed

    Args:
        metrics_by_split: Dict mapping split names to metric dicts
            e.g., {"train": {"precision": 0.8, "recall": 0.75, ...}, "val": {...}, "test": {...}}
        model_config: Dict with keys: best_threshold, metric_optimized, n_estimators, random_seed
        filepath: Path to CSV file (created if doesn't exist)

    Returns:
        None
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    file_exists = Path(filepath).exists()

    with open(filepath, 'a', newline='') as f:
        fieldnames = [
            'timestamp', 'split', 'precision', 'recall', 'f1', 'accuracy', 'roc_auc',
            'best_threshold', 'metric_optimized', 'n_estimators', 'random_seed'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is new
        if not file_exists:
            writer.writeheader()

        # Write one row per split
        for split_name, metrics in metrics_by_split.items():
            row = {
                'timestamp': timestamp,
                'split': split_name,
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1': metrics.get('f1'),
                'accuracy': metrics.get('accuracy'),
                'roc_auc': metrics.get('roc_auc'),
                'best_threshold': model_config.get('best_threshold'),
                'metric_optimized': model_config.get('metric_optimized'),
                'n_estimators': model_config.get('n_estimators'),
                'random_seed': model_config.get('random_seed')
            }
            writer.writerow(row)

    logger.info(f"Appended metrics to CSV: {filepath} | Splits: {', '.join(metrics_by_split.keys())}")


def build_metrics_payload(
    metrics_by_split: Dict[str, Dict[str, Any]],
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build complete metrics payload for JSON file.

    Args:
        metrics_by_split: Dict mapping split names to evaluation results
            e.g., {"train": {eval_result}, "val": {...}, "test": {...}}
            where eval_result is from evaluate_model()
        model_config: Dict with model configuration (threshold, estimators, seed, etc.)

    Returns:
        Dictionary ready for JSON serialization
    """
    # Extract flat metrics from nested eval results
    split_metrics = {}
    for split_name, eval_result in metrics_by_split.items():
        # eval_result structure: {"threshold": X, "metrics": {...}, "confusion_matrix": {...}}
        if 'metrics' in eval_result:
            split_metrics[split_name] = eval_result['metrics'].copy()
            if 'confusion_matrix' in eval_result:
                split_metrics[split_name]['confusion_matrix'] = eval_result['confusion_matrix']
        else:
            # Fallback for flat dict structure
            split_metrics[split_name] = eval_result

    payload = {
        'timestamp': datetime.now().isoformat(),
        'split_metrics': split_metrics,
        'model_config': model_config
    }

    return payload


def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validate that metrics dictionary has required structure and reasonable values.

    Args:
        metrics: Metrics dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    # Check required keys
    if 'split_metrics' not in metrics or 'model_config' not in metrics:
        logger.warning("Metrics missing required keys: split_metrics or model_config")
        return False

    # Check at least one split
    if not metrics['split_metrics']:
        logger.warning("No splits found in metrics")
        return False

    # Validate metric ranges
    for split_name, split_data in metrics['split_metrics'].items():
        for metric_name in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
            if metric_name in split_data:
                val = split_data[metric_name]
                if not isinstance(val, (int, float)) or not 0 <= val <= 1:
                    logger.warning(
                        f"{split_name}.{metric_name}={val} is outside [0, 1] range"
                    )
                    return False

    logger.info("Metrics validation passed")
    return True


if __name__ == "__main__":
    print("Metrics saving utilities")
    print("Use: from src.utils.metrics_saver import save_metrics_to_json, append_metrics_to_csv")
