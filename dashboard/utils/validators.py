"""Input validation utilities."""

import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

from config import CUSTOMER_FIELDS, CATEGORICAL_VALUES, NUMERIC_RANGES

logger = logging.getLogger(__name__)


def convert_numpy_to_python(value: Any) -> Any:
    """Convert numpy/pandas types to native Python types for JSON serialization.
    
    Args:
        value: Value that might be numpy type (int64, float64, etc.)
        
    Returns:
        Native Python type (int, float, str, bool)
    """
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (pd.Series, pd.Index)):
        return value.item() if len(value) == 1 else value.tolist()
    elif pd.isna(value):
        return None
    return value


def validate_customer_fields(
    customer: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Validate customer input data.
    
    Args:
        customer: Customer feature dictionary
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check all required fields present
    for field in CUSTOMER_FIELDS:
        if field not in customer:
            errors.append(f"Missing required field: {field}")
    
    # Validate categorical fields
    for field, valid_values in CATEGORICAL_VALUES.items():
        if field in customer and customer[field] not in valid_values:
            errors.append(
                f"Invalid value for {field}: {customer[field]}. "
                f"Valid options: {valid_values}"
            )
    
    # Validate numeric ranges
    for field, (min_val, max_val) in NUMERIC_RANGES.items():
        if field in customer:
            try:
                val = float(customer[field])
                if not (min_val <= val <= max_val):
                    errors.append(
                        f"{field} out of range [{min_val}, {max_val}]: {val}"
                    )
            except (ValueError, TypeError):
                errors.append(f"{field} must be numeric: {customer[field]}")
    
    return len(errors) == 0, errors


def validate_csv_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate CSV has required columns.
    
    Args:
        df: Pandas DataFrame from CSV
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check column count
    if len(df.columns) != 19:
        errors.append(f"CSV must have exactly 19 columns, found {len(df.columns)}")
    
    # Check all required fields present (case-insensitive)
    df_cols_lower = [c.lower() for c in df.columns]
    customer_fields_lower = [f.lower() for f in CUSTOMER_FIELDS]
    
    for field in customer_fields_lower:
        if field not in df_cols_lower:
            errors.append(f"Missing required column: {field}")
    
    return len(errors) == 0, errors


def parse_csv_file(uploaded_file) -> Tuple[bool, pd.DataFrame, str]:
    """Parse uploaded CSV file with validation.
    
    Args:
        uploaded_file: Streamlit file upload object
        
    Returns:
        Tuple of (success, dataframe, error_message)
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate schema
        is_valid, errors = validate_csv_schema(df)
        if not is_valid:
            error_msg = "CSV validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            return False, df, error_msg
        
        # Rename columns to match expected names (case-insensitive mapping)
        col_mapping = {}
        for orig_col in df.columns:
            for expected_field in CUSTOMER_FIELDS:
                if orig_col.lower() == expected_field.lower():
                    col_mapping[orig_col] = expected_field
                    break
        
        df = df.rename(columns=col_mapping)
        
        logger.info(f"Parsed CSV with {len(df)} rows, {len(df.columns)} columns")
        return True, df, ""
    
    except Exception as e:
        error_msg = f"CSV parsing error: {str(e)}"
        logger.error(error_msg)
        return False, pd.DataFrame(), error_msg
