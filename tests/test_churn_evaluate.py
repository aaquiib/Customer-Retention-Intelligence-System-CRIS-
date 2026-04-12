"""Test suite for churn evaluation module."""

import pytest
import numpy as np

from src.churn.evaluate import compare_thresholds, evaluate_model


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    y_true = np.random.choice([0, 1], 100)
    y_proba = np.random.uniform(0, 1, 100)
    y_pred = (y_proba >= 0.5).astype(int)
    
    return y_true, y_pred, y_proba


def test_evaluate_returns_dict(sample_predictions):
    """Test that evaluate_model returns a dictionary."""
    y_true, y_pred, y_proba = sample_predictions
    result = evaluate_model(y_true, y_pred, y_proba)
    
    assert isinstance(result, dict)
    assert 'metrics' in result
    assert 'confusion_matrix' in result


def test_evaluate_metrics_in_range(sample_predictions):
    """Test that metrics are in valid range."""
    y_true, y_pred, y_proba = sample_predictions
    result = evaluate_model(y_true, y_pred, y_proba)
    
    metrics = result['metrics']
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_compare_thresholds_compares_multiple(sample_predictions):
    """Test that compare_thresholds evaluates multiple thresholds."""
    y_true, y_pred, y_proba = sample_predictions
    thresholds = [0.3, 0.5, 0.7]
    result = compare_thresholds(y_true, y_proba, thresholds)
    
    assert len(result) == len(thresholds)
    for threshold in thresholds:
        assert threshold in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
