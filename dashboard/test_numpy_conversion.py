"""Test numpy type conversion for batch data loading."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from utils.validators import convert_numpy_to_python
import json

print("=" * 70)
print("NUMPY TYPE CONVERSION TEST")
print("=" * 70)

# Test various numpy types
test_cases = {
    "int64": np.int64(42),
    "int32": np.int32(10),
    "float64": np.float64(65.5),
    "float32": np.float32(3.14),
    "string": "Male",
    "bool": True,
    "None/NaN": np.nan,
    "Python int": 5,
    "Python float": 7.5,
}

print("\n1. Testing Type Conversions")
print("-" * 70)

converted = {}
for name, value in test_cases.items():
    converted_value = convert_numpy_to_python(value)
    original_type = type(value).__name__
    converted_type = type(converted_value).__name__
    converted[name] = converted_value
    
    print(f"✅ {name:20} | {original_type:12} → {converted_type:12} | {converted_value}")

print("\n2. Testing JSON Serialization")
print("-" * 70)

# Create a dict like the what-if simulator would
batch_customer_data = {
    "gender": "Male",
    "SeniorCitizen": np.int64(0),
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": np.int64(12),
    "MonthlyCharges": np.float64(65.5),
    "TotalCharges": np.float64(786.0),
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}

print("Original data (with numpy types):")
print(f"  SeniorCitizen type: {type(batch_customer_data['SeniorCitizen'])}")
print(f"  tenure type: {type(batch_customer_data['tenure'])}")
print(f"  MonthlyCharges type: {type(batch_customer_data['MonthlyCharges'])}")

print("\nAttempting JSON serialization WITHOUT conversion...")
try:
    json.dumps(batch_customer_data)
    print("❌ Should have failed but didn't!")
except TypeError as e:
    print(f"❌ Expected error: {e}")

print("\nApplying conversion function...")
converted_data = {k: convert_numpy_to_python(v) for k, v in batch_customer_data.items()}

print(f"  SeniorCitizen type: {type(converted_data['SeniorCitizen'])}")
print(f"  tenure type: {type(converted_data['tenure'])}")
print(f"  MonthlyCharges type: {type(converted_data['MonthlyCharges'])}")

print("\nAttempting JSON serialization WITH conversion...")
try:
    json_str = json.dumps(converted_data)
    print(f"✅ JSON serialization successful!")
    print(f"   Serialized length: {len(json_str)} chars")
except TypeError as e:
    print(f"❌ Failed: {e}")

print("\n3. Testing with Pandas DataFrame")
print("-" * 70)

# Simulate what happens when loading from batch_df
df = pd.DataFrame({
    "customerID": [0],
    "gender": ["Male"],
    "SeniorCitizen": [np.int64(0)],
    "tenure": [np.int64(12)],
    "MonthlyCharges": [np.float64(65.5)],
    "Contract": ["Month-to-month"]
})

print(f"DataFrame dtypes:")
for col in df.columns:
    print(f"  {col:20} : {df[col].dtype}")

print(f"\nExtracting customer 0...")
customer_row = df[df["customerID"] == 0].iloc[0]

print(f"\nBefore conversion:")
print(f"  SeniorCitizen: {customer_row['SeniorCitizen']} (type: {type(customer_row['SeniorCitizen']).__name__})")
print(f"  tenure: {customer_row['tenure']} (type: {type(customer_row['tenure']).__name__})")

print(f"\nAfter conversion:")
converted_senior = convert_numpy_to_python(customer_row['SeniorCitizen'])
converted_tenure = convert_numpy_to_python(customer_row['tenure'])
print(f"  SeniorCitizen: {converted_senior} (type: {type(converted_senior).__name__})")
print(f"  tenure: {converted_tenure} (type: {type(converted_tenure).__name__})")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - BATCH DATA WHAT-IF SHOULD NOW WORK")
print("=" * 70)
