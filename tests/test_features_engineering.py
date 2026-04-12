"""Test suite for feature engineering module."""

import pytest
import pandas as pd

from src.config import load_config
from src.features import engineer_features


@pytest.fixture
def sample_preprocessed_data():
    """Create sample preprocessed data for testing."""
    return pd.DataFrame({
        'tenure': [1, 12, 24, 60],
        'MonthlyCharges': [29.85, 65.0, 52.0, 45.0],
        'TotalCharges': [29.85, 780.0, 1248.0, 2700.0],
        'StreamingTV': ['No', 'Yes', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'Yes', 'Yes'],
        'OnlineSecurity': ['No', 'Yes', 'No', 'Yes'],
        'OnlineBackup': ['No', 'No', 'Yes', 'Yes'],
        'DeviceProtection': ['No', 'No', 'No', 'Yes'],
        'TechSupport': ['No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'],
        'Partner': ['No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic'],
    })


def test_engineer_creates_billing_features(sample_preprocessed_data):
    """Test that billing features are created."""
    cfg = load_config()
    result = engineer_features(sample_preprocessed_data, cfg)
    
    assert 'avg_monthly_spend' in result.columns
    assert 'charge_gap' in result.columns
    assert 'is_high_value' in result.columns


def test_engineer_creates_tenure_band(sample_preprocessed_data):
    """Test that tenure band is created."""
    cfg = load_config()
    result = engineer_features(sample_preprocessed_data, cfg)
    
    assert 'tenure_band' in result.columns
    assert len(result['tenure_band'].unique()) > 1


def test_engineer_creates_service_counts(sample_preprocessed_data):
    """Test that service count features are created."""
    cfg = load_config()
    result = engineer_features(sample_preprocessed_data, cfg)
    
    assert 'streaming_count' in result.columns
    assert 'security_count' in result.columns


def test_engineer_creates_risk_flags(sample_preprocessed_data):
    """Test that risk flags are created."""
    cfg = load_config()
    result = engineer_features(sample_preprocessed_data, cfg)
    
    assert 'payment_electronic_check' in result.columns
    assert 'is_isolated' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
