"""Test validators module for CSV and customer field validation."""
import sys
sys.path.insert(0, '.')

import pandas as pd
from utils.validators import validate_csv_schema, validate_customer_fields

# Test 1: Valid CSV validation
print("\n=== TEST 1: Valid CSV schema ===")
valid_csv = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [24],
    'MonthlyCharges': [65.50],
    'TotalCharges': [1572.00],
    'PhoneService': ['Yes'],
    'MultipleLines': ['Yes'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['Yes'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['Yes'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Two year'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
})

is_valid, errors = validate_csv_schema(valid_csv)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert is_valid, f"Valid CSV should pass validation. Errors: {errors}"
print("✅ Valid CSV passes validation")

# Test 2: Missing required columns
print("\n=== TEST 2: CSV with missing required columns ===")
missing_cols = valid_csv[['gender', 'tenure']].copy()  # Missing many columns
is_valid, errors = validate_csv_schema(missing_cols)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert not is_valid, "CSV with missing columns should fail"
assert len(errors) > 0, "Should have error messages"
print("✅ Missing columns detected correctly")

# Test 3: Wrong column count
print("\n=== TEST 3: CSV with wrong column count ===")
wrong_cols = valid_csv.drop(columns=['MonthlyCharges'])
is_valid, errors = validate_csv_schema(wrong_cols)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert not is_valid, "CSV with wrong column count should fail"
print("✅ Wrong column count detected")

# Test 4: Customer field validation - valid
print("\n=== TEST 4: Valid customer fields ===")
customer_input = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'MonthlyCharges': 65.50,
    'TotalCharges': 1572.00,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'Yes',
    'StreamingTV': 'No',
    'StreamingMovies': 'Yes',
    'Contract': 'Two year',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
}

is_valid, errors = validate_customer_fields(customer_input)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert is_valid, f"Valid customer input should pass. Errors: {errors}"
print("✅ Valid customer input passes")

# Test 5: Customer input validation - missing fields
print("\n=== TEST 5: Customer input with missing fields ===")
incomplete_input = {k: v for k, v in customer_input.items() if k != 'gender'}
is_valid, errors = validate_customer_fields(incomplete_input)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert not is_valid, "Missing fields should fail"
print("✅ Missing fields detected")

# Test 6: Customer input validation - invalid values
print("\n=== TEST 6: Customer input with invalid values ===")
invalid_input = customer_input.copy()
invalid_input['tenure'] = 100  # Out of range
is_valid, errors = validate_customer_fields(invalid_input)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert not is_valid, "Invalid values should fail"
print("✅ Invalid values detected in customer input")

# Test 7: Customer input validation - invalid categorical
print("\n=== TEST 7: Customer input with invalid categorical value ===")
invalid_cat = customer_input.copy()
invalid_cat['gender'] = 'InvalidGender'
is_valid, errors = validate_customer_fields(invalid_cat)
print(f"  Valid: {is_valid}")
print(f"  Errors: {errors}")
assert not is_valid, "Invalid categorical should fail"
print("✅ Invalid categorical values detected")

print("\n" + "="*60)
print("🎉 ALL VALIDATION TESTS PASSED!")
print("="*60)
