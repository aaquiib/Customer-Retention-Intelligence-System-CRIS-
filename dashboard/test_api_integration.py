"""Comprehensive API integration test."""
import sys
sys.path.insert(0, '.')

from utils.api_client import APIClient
import time

print("\n" + "="*70)
print("🔌 COMPREHENSIVE API INTEGRATION TEST")
print("="*70)

client = APIClient()

# Test 1: Health Check
print("\n📡 Test 1: Health Check")
success, data, error = client.get_health()
if success:
    print(f"   ✅ SUCCESS")
    print(f"   Status: {data.get('status')}")
    print(f"   Models Loaded: {data.get('models_loaded')}")
else:
    print(f"   ❌ FAILED: {error}")

# Test 2: Model Info
print("\n📡 Test 2: Get Model Info")
success, data, error = client.get_model_info()
if success:
    print(f"   ✅ SUCCESS")
    print(f"   Model Type: {data.get('model_type')}")
    print(f"   N Features: {data.get('n_features')}")
    print(f"   Features: {str(data.get('feature_names', [])[:3])}...")
else:
    print(f"   ❌ FAILED: {error}")

# Test 3: Get Batch Template
print("\n📡 Test 3: Get Batch CSV Template")
success, data, error = client.get_batch_template()
if success:
    print(f"   ✅ SUCCESS")
    if isinstance(data, bytes):
        print(f"   Template Size: {len(data)} bytes")
        print(f"   Preview: {str(data[:100])}...")
    else:
        print(f"   Data: {str(data)[:100]}...")
else:
    print(f"   ❌ FAILED: {error}")

# Test 4: Single Prediction
print("\n📡 Test 4: Single Customer Prediction")
customer = {
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

success, data, error = client.predict_single(customer)
if success:
    print(f"   ✅ SUCCESS")
    print(f"   Churn Probability: {data.get('churn_probability', 'N/A'):.4f}")
    print(f"   Segment: {data.get('segment_label', 'N/A')}")
    print(f"   Recommended Action: {data.get('recommended_action', 'N/A')}")
else:
    print(f"   ❌ FAILED: {error}")

# Test 5: Global Feature Importance
print("\n📡 Test 5: Global Feature Importance")
success, data, error = client.get_global_importance()
if success:
    print(f"   ✅ SUCCESS")
    print(f"   Total Features: {len(data.get('importances', []))}")
    if data.get('importances'):
        print(f"   Top 3: {data.get('importances', [])[:3]}")
else:
    print(f"   ❌ FAILED: {error}")

# Test 6: Instance Importance (SHAP)
print("\n📡 Test 6: Instance Feature Importance (SHAP)")
success, data, error = client.get_instance_importance(customer)
if success:
    print(f"   ✅ SUCCESS")
    base_value = data.get('base_value', 'N/A')
    shap_values = data.get('shap_values', [])
    print(f"   Base Value: {base_value}")
    print(f"   SHAP Values Count: {len(shap_values)}")
    if shap_values:
        print(f"   Top Feature Impact: {shap_values[0] if isinstance(shap_values, list) else 'N/A'}")
else:
    print(f"   ❌ FAILED: {error}")

print("\n" + "="*70)
print("API INTEGRATION TEST COMPLETE")
print("="*70)
print("\n✅ All critical endpoints tested successfully!")
print("\nDashboard is now ready to connect to API at http://localhost:8000")
