"""Churn model evaluation utilities."""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    threshold: float = 0.5,
    dataset_name: str = "test",
    cfg: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluate binary classification model on test set.

    Computes: ROC-AUC, Precision, Recall, F1, Confusion Matrix, Classification Report

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_proba: Predicted probabilities (for ROC-AUC) - optional
        threshold: Decision threshold (default 0.5)
        dataset_name: Name of dataset being evaluated ("train", "val", "test") - optional
        cfg: Configuration dictionary - optional

    Returns:
        Dictionary with all metrics
    """
    results = {
        'threshold': threshold,
        'metrics': {}
    }

    # Compute metrics
    results['metrics']['accuracy'] = (y_pred == y_true).mean()
    results['metrics']['precision'] = precision_score(y_true, y_pred, zero_division=0)
    results['metrics']['recall'] = recall_score(y_true, y_pred, zero_division=0)
    results['metrics']['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        results['metrics']['roc_auc'] = roc_auc_score(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }

    # Classification report (as dict)
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    results['classification_report'] = clf_report

    logger.info(f"Evaluation on {dataset_name} set at threshold {threshold:.4f}:")
    logger.info(f"  Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"  Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"  F1: {results['metrics']['f1']:.4f}")
    if 'roc_auc' in results['metrics']:
        logger.info(f"  ROC-AUC: {results['metrics']['roc_auc']:.4f}")

    return results


def evaluate_model_on_splits(
    splits: Dict[str, Dict[str, np.ndarray]],
    threshold: float = 0.5,
    cfg: Dict[str, Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model on multiple dataset splits (train, val, test).

    Args:
        splits: Dictionary mapping split names to {"y_true": array, "y_pred": array, "y_proba": array}
            e.g., {"train": {"y_true": [...], "y_pred": [...], "y_proba": [...]}, "val": {...}, "test": {...}}
        threshold: Decision threshold (applied to all splits)
        cfg: Configuration dictionary - optional

    Returns:
        Dictionary mapping split names to evaluation results:
        {"train": {eval_result}, "val": {eval_result}, "test": {eval_result}}
    """
    results = {}

    for split_name, split_data in splits.items():
        y_true = split_data.get('y_true')
        y_pred = split_data.get('y_pred')
        y_proba = split_data.get('y_proba')

        if y_true is None or y_pred is None:
            logger.warning(f"Skipping split '{split_name}': missing y_true or y_pred")
            continue

        results[split_name] = evaluate_model(
            y_true,
            y_pred,
            y_proba=y_proba,
            threshold=threshold,
            dataset_name=split_name,
            cfg=cfg
        )

    logger.info(f"Evaluation complete on {len(results)} splits")
    return results


def compare_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list = None,
    cfg: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compare metrics across multiple decision thresholds.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to compare (default: [0.3, 0.4, 0.5, 0.6, 0.7])
        cfg: Configuration dictionary - optional

    Returns:
        Dictionary mapping thresholds to evaluation results
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    comparison = {}

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        comparison[threshold] = evaluate_model(y_true, y_pred, y_proba, threshold, cfg)

    logger.info(f"Threshold comparison complete for {len(thresholds)} thresholds")

    return comparison


if __name__ == "__main__":
    # Example usage
    print("Churn model evaluation utilities")
    print("Use: from src.churn.evaluate import evaluate_model, compare_thresholds")
