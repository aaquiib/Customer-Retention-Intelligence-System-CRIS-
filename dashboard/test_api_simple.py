"""Simple API integration test without Streamlit overhead."""
import requests
import json

print("\n" + "="*70)
print("🔌 API INTEGRATION TEST (FAST)")
print("="*70)

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"

tests_passed = 0
tests_failed = 0

# Test 1: Health Check (at root level)
print("\n📡 Test 1: Health Check")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS")
        print(f"   Status: {data.get('status')}")
        print(f"   Models Loaded: {data.get('models_loaded')}")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Test 2: Model Info
print("\n📡 Test 2: Model Info")
try:
    response = requests.get(f"{BASE_URL}{API_PREFIX}/explanations/model-info", timeout=15)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS")
        print(f"   Model: {data.get('model_type')}")
        print(f"   N Estimators: {data.get('n_estimators')}")
        print(f"   N Features: {data.get('n_features')}")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Test 3: Batch Template
print("\n📡 Test 3: Batch CSV Template")
try:
    response = requests.get(f"{BASE_URL}{API_PREFIX}/predict-batch/template", timeout=5)
    if response.status_code == 200:
        print(f"   ✅ SUCCESS")
        print(f"   Template created: {len(response.content)} bytes")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Test 4: Single Prediction
print("\n📡 Test 4: Single Prediction")
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
try:
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/predict",
        json={"customer": customer, "return_features": False},
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        pred = data.get('prediction', {})
        print(f"   ✅ SUCCESS")
        print(f"   Churn Probability: {pred.get('churn_probability', 'N/A'):.4f}")
        print(f"   Segment: {pred.get('segment_label', 'N/A')}")
        print(f"   Action: {pred.get('recommended_action', 'N/A')}")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:150]}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Test 5: Global Importance
print("\n📡 Test 5: Global Feature Importance")
try:
    response = requests.get(f"{BASE_URL}{API_PREFIX}/feature-importance/global", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS")
        imps = data.get('importances', [])
        print(f"   Features: {len(imps)}")
        if imps:
            print(f"   Top 3: {imps[:3]}")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Test 6: Instance Importance  
print("\n📡 Test 6: Instance Feature Importance (SHAP)")
try:
    response = requests.post(
        f"{BASE_URL}{API_PREFIX}/feature-importance/instance",
        json={"customer": customer},
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS")
        base_value = data.get('base_value', 'N/A')
        if isinstance(base_value, (int, float)):
            print(f"   Base Value: {base_value:.4f}")
        else:
            print(f"   Base Value: {base_value}")
        shap_vals = data.get('shap_values', [])
        print(f"   SHAP Values: {len(shap_vals)}")
        tests_passed += 1
    else:
        print(f"   ❌ HTTP {response.status_code}: {response.text[:150]}")
        tests_failed += 1
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    tests_failed += 1

# Summary
print("\n" + "="*70)
print(f"RESULTS: {tests_passed} Passed ✅ | {tests_failed} Failed ❌")
print("="*70)

if tests_failed == 0:
    print("\n🎉 ALL API ENDPOINTS WORKING!")
    print("\nDashboard can now connect to FastAPI at http://localhost:8000")
    print("Visit: http://localhost:8501 to use the dashboard")
else:
    print(f"\n⚠️  {tests_failed} tests failed. Check API server logs.")
