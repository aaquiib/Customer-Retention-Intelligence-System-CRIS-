"""Test suite for data preprocessing module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data import preprocess_data


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003', 'C004'],
        'SeniorCitizen': [0, 1, 0, 0],
        'tenure': [1, 12, 24, 0],  # One row with tenure=0
        'MonthlyCharges': [29.85, 65.0, 52.0, 45.0],
        'TotalCharges': ['nan', '780.0', '1248.0', '0'],  # One NaN
        'Churn': ['No', 'Yes', 'No', 'Yes']
    })


def test_preprocess_drops_customerid(sample_raw_data):
    """Test that customerID column is dropped."""
    cfg = load_config()
    result = preprocess_data(sample_raw_data, cfg)
    assert 'customerID' not in result.columns


def test_preprocess_converts_seniorcitizen(sample_raw_data):
    """Test that SeniorCitizen is converted to yes/no."""
    cfg = load_config()
    result = preprocess_data(sample_raw_data, cfg)
    assert set(result['SeniorCitizen'].unique()).issubset({'yes', 'no'})


def test_preprocess_converts_churn(sample_raw_data):
    """Test that Churn is converted to 0/1."""
    cfg = load_config()
    result = preprocess_data(sample_raw_data, cfg)
    assert set(result['Churn'].unique()).issubset({0, 1})


def test_preprocess_drops_tenure_zero(sample_raw_data):
    """Test that rows with tenure=0 are dropped."""
    cfg = load_config()
    result = preprocess_data(sample_raw_data, cfg)
    assert (result['tenure'] == 0).sum() == 0


def test_preprocess_handles_nan_totalcharges(sample_raw_data):
    """Test that NaN TotalCharges are handled."""
    cfg = load_config()
    result = preprocess_data(sample_raw_data, cfg)
    assert result['TotalCharges'].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
