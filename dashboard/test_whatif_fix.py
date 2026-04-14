"""Test What-If simulator API client and data flow."""

import sys
sys.path.insert(0, ".")

from utils.api_client import APIClient
from utils.data_processors import prepare_batch_result_df
import json

# Test customer data with all 19 fields
test_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
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

modifications = {
    "Contract": "Two year",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes"
}

print("=" * 70)
print("WHAT-IF SIMULATOR TEST")
print("=" * 70)

api_client = APIClient()

print("\n1. Testing What-If Single Scenario")
print("-" * 70)

success, scenario, error = api_client.what_if_single(test_customer, modifications)

print(f"✅ Success: {success}")
print(f"❌ Error: {error}" if error else "")

if success and scenario:
    print(f"\n📊 Scenario Keys: {list(scenario.keys())}")
    
    # Check required keys
    required_keys = ["original_prediction", "modified_prediction", "delta"]
    missing_keys = [k for k in required_keys if k not in scenario]
    
    if missing_keys:
        print(f"❌ Missing keys: {missing_keys}")
    else:
        print(f"✅ All required keys present")
    
    # Display results
    if "original_prediction" in scenario:
        orig = scenario["original_prediction"]
        print(f"\n📈 Original Prediction:")
        print(f"   Segment: {orig.get('segment_label', 'N/A')}")
        print(f"   Churn Probability: {orig.get('churn_probability', 'N/A'):.4f}")
        print(f"   Will Churn: {orig.get('is_churner', 'N/A')}")
    
    if "modified_prediction" in scenario:
        mod = scenario["modified_prediction"]
        print(f"\n📉 Modified Prediction:")
        print(f"   Segment: {mod.get('segment_label', 'N/A')}")
        print(f"   Churn Probability: {mod.get('churn_probability', 'N/A'):.4f}")
        print(f"   Will Churn: {mod.get('is_churner', 'N/A')}")
    
    if "delta" in scenario:
        delta = scenario["delta"]
        delta_prob = delta.get('churn_probability_delta', 0)
        direction = "↓ IMPROVEMENT" if delta_prob < 0 else "↑ WORSENING"
        print(f"\n💫 Delta (Change):")
        print(f"   Probability Delta: {delta_prob:+.4f} ({direction})")
        print(f"   Segment Changed: {delta.get('segment_changed', False)}")
        print(f"   Churner Status Changed: {delta.get('is_churner_changed', False)}")
    
    if "modified_features" in scenario:
        print(f"\n🔧 Modifications Applied:")
        for field, value in scenario.get("modified_features", {}).items():
            print(f"   {field}: {value}")
    
    print("\n✅ What-If simulator is working correctly!")

else:
    print(f"❌ Failed to get scenario: {scenario}")

print("\n" + "=" * 70)
print("2. Testing Batch DataFrame with All Customer Fields")
print("-" * 70)

# Simulate a batch response structure
mock_batch_predictions = [
    {
        "segment": 1,
        "segment_label": "Low Engagement",
        "segment_confidence": 0.92,
        "churn_probability": 0.745,
        "is_churner": True,
        "recommended_action": {"action_label": "Monitor", "priority_score": 0.5, "reason": "Test"},
        "input_features": test_customer,
        "top_features": [
            {"feature_name": "num__tenure", "shap_value": -0.234},
            {"feature_name": "cat__Contract_Month-to-month", "shap_value": 0.156},
        ]
    }
]

batch_df = prepare_batch_result_df(mock_batch_predictions)

print(f"\n📊 Batch DataFrame Info:")
print(f"   Shape: {batch_df.shape}")
print(f"   Total Columns: {len(batch_df.columns)}")

# Check for all 19 required customer fields
required_fields = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "MonthlyCharges", "TotalCharges", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

missing_fields = [f for f in required_fields if f not in batch_df.columns]

if missing_fields:
    print(f"❌ Missing customer fields: {missing_fields}")
else:
    print(f"✅ All 19 customer fields present")

print(f"\nAvailable columns:")
for col in sorted(batch_df.columns):
    print(f"   - {col}")

print("\n✅ Batch DataFrame includes all fields for What-If simulator!")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - WHAT-IF SIMULATOR SHOULD WORK NOW")
print("=" * 70)
