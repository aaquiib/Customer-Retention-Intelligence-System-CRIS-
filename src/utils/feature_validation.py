"""Feature validation utilities (extracted from segmentation.ipynb)."""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def validate_feature_consistency(
    df: pd.DataFrame,
    expected_num_cols: List[str],
    expected_cat_cols: List[str],
    phase: str = ""
) -> Dict[str, Any]:
    """
    Validate that a dataframe has expected features with correct properties.

    Args:
        df: DataFrame to validate
        expected_num_cols: list of expected numeric feature names
        expected_cat_cols: list of expected categorical feature names
        phase: str, phase name for error messages

    Returns:
        dict with validation results

    Raises:
        ValueError: if features don't match expectations
    """
    results = {
        'phase': phase,
        'all_pass': True,
        'checks': {}
    }

    # Check 1: All expected numeric columns present
    missing_num = [c for c in expected_num_cols if c not in df.columns]
    if missing_num:
        results['all_pass'] = False
        results['checks']['missing_numeric'] = missing_num
        raise ValueError(f"[{phase}] Missing numeric columns: {missing_num}")
    results['checks']['numeric_present'] = '✓'

    # Check 2: All expected categorical columns present
    missing_cat = [c for c in expected_cat_cols if c not in df.columns]
    if missing_cat:
        results['all_pass'] = False
        results['checks']['missing_categorical'] = missing_cat
        raise ValueError(f"[{phase}] Missing categorical columns: {missing_cat}")
    results['checks']['categorical_present'] = '✓'

    # Check 3: Verify numeric columns are numeric
    for col in expected_num_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                results['all_pass'] = False
                raise ValueError(f"[{phase}] Numeric column '{col}' has dtype {df[col].dtype}")
    results['checks']['numeric_dtypes'] = '✓'

    # Check 4: Verify categorical columns are string type
    non_string_cats = []
    for col in expected_cat_cols:
        if col in df.columns:
            if df[col].dtype not in ['object', 'string']:
                non_string_cats.append(f"{col} ({df[col].dtype})")

    if non_string_cats:
        results['checks']['categorical_dtype_warning'] = f"⚠ {non_string_cats}"
    else:
        results['checks']['categorical_dtypes'] = '✓'

    results['checks']['numeric_count'] = len(expected_num_cols)
    results['checks']['categorical_count'] = len(expected_cat_cols)

    return results


def print_feature_validation_report(
    results_training: dict,
    results_prediction: dict = None
) -> None:
    """Print formatted validation report."""
    print("\n" + "="*80)
    print("FEATURE CONSISTENCY VALIDATION REPORT")
    print("="*80)

    print(f"\n[TRAINING PHASE]")
    print(f"  Numeric features: {results_training['checks']['numeric_count']}")
    print(f"  Categorical features: {results_training['checks']['categorical_count']}")

    if results_prediction:
        print(f"\n[PREDICTION PHASE]")
        print(f"  Numeric features: {results_prediction['checks']['numeric_count']}")
        print(f"  Categorical features: {results_prediction['checks']['categorical_count']}")

        num_match = (results_training['checks']['numeric_count'] ==
                     results_prediction['checks']['numeric_count'])
        cat_match = (results_training['checks']['categorical_count'] ==
                     results_prediction['checks']['categorical_count'])

        print(f"\n[COMPARISON] Numeric match: {'✓' if num_match else '✗'} | "
              f"Categorical match: {'✓' if cat_match else '✗'}")

    print("="*80 + "\n")
