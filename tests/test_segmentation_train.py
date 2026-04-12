"""Test suite for segmentation module."""

import pytest
import pandas as pd
import numpy as np

from src.config import load_config
from src.segmentation.train_segments import train_segmentation_model


@pytest.fixture
def sample_engineered_data():
    """Create sample engineered data for testing."""
    np.random.seed(42)
    n_rows = 100
    
    return pd.DataFrame({
        'tenure': np.random.randint(1, 72, n_rows),
        'MonthlyCharges': np.random.uniform(20, 100, n_rows),
        'TotalCharges': np.random.uniform(20, 5000, n_rows),
        'avg_monthly_spend': np.random.uniform(20, 100, n_rows),
        'charge_gap': np.random.uniform(-50, 50, n_rows),
        'streaming_count': np.random.randint(0, 3, n_rows),
        'security_count': np.random.randint(0, 5, n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'SeniorCitizen': np.random.choice(['yes', 'no'], n_rows),
        'Partner': np.random.choice(['Yes', 'No'], n_rows),
        'Dependents': np.random.choice(['Yes', 'No'], n_rows),
        'tenure_band': np.random.choice(['0-12', '12-36', '36+'], n_rows),
        'is_high_value': np.random.choice([0, 1], n_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], n_rows),
        'MultipleLines': np.random.choice(['Yes', 'No'], n_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_rows),
        'PaymentMethod': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_rows),
        'payment_electronic_check': np.random.choice([0, 1], n_rows),
        'month_to_month_paperless': np.random.choice([0, 1], n_rows),
        'no_support_services': np.random.choice([0, 1], n_rows),
        'is_isolated': np.random.choice([0, 1], n_rows),
        'fiber_no_security': np.random.choice([0, 1], n_rows),
        'no_internet_services': np.random.choice([0, 1], n_rows),
    })


def test_train_segmentation_returns_model(sample_engineered_data):
    """Test that training returns a KPrototypes model."""
    cfg = load_config()
    kproto, scaler, cat_idx, metadata = train_segmentation_model(sample_engineered_data, cfg)
    
    assert kproto is not None
    assert scaler is not None
    assert len(cat_idx) > 0
    assert 'numeric_columns' in metadata


def test_train_segmentation_saves_artifacts(sample_engineered_data, tmp_path):
    """Test that artifacts are saved correctly."""
    cfg = load_config()
    # Override model directories for testing
    cfg['models']['segmentation_dir'] = str(tmp_path) + '/'
    
    kproto, scaler, cat_idx, metadata = train_segmentation_model(sample_engineered_data, cfg)
    
    # Check files exist
    assert (tmp_path / 'kproto.pkl').exists()
    assert (tmp_path / 'scaler.pkl').exists()
    assert (tmp_path / 'catidx.json').exists()
    assert (tmp_path / 'feature_metadata.json').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
