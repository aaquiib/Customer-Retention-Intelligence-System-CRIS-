"""Final comprehensive API test with proper port."""

import requests
import json
import sys

BASE_URL = "http://localhost:8001"

print("\n" + "=" * 80)
print("FINAL CRIS API COMPREHENSIVE TEST")
print("=" * 80)

# Sample customer
customer_high_churn_risk = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 6, 'MonthlyCharges': 85.0, 'TotalCharges': 510.0,
    'PhoneService': 'No', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check'
}

customer_low_churn_risk = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
    'tenure': 60, 'MonthlyCharges': 45.0, 'TotalCharges': 2700.0,
    'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL',
    'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes',
    'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Two year', 'PaperlessBilling': 'No', 'PaymentMethod': 'Credit card (automatic)'
}

tests_passed = 0
tests_failed = 0

# Test 1: Health
print("\n[1] HEALTH CHECK")
try:
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()['models_loaded'] == True
    print("✓ API is healthy and models are loaded")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 2: Single prediction - high churn risk
print("\n[2] SINGLE PREDICTION (HIGH CHURN RISK CUSTOMER)")
try:
    r = requests.post(f"{BASE_URL}/api/predict",
        json={'customer': customer_high_churn_risk})
    assert r.status_code == 200
    data = r.json()
    assert data['success'] == True
    pred = data['prediction']
    print(f"✓ Segment: {pred['segment']} ({pred['segment_label']})")
    print(f"✓ Churn prob: {pred['churn_probability']:.2%}")
    print(f"✓ Action: {data['recommended_action']} (Priority: {data['priority_score']:.1%})")
    assert pred['is_churner'] == True  # Should be high churn risk
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 3: Single prediction - low churn risk
print("\n[3] SINGLE PREDICTION (LOW CHURN RISK CUSTOMER)")
try:
    r = requests.post(f"{BASE_URL}/api/predict",
        json={'customer': customer_low_churn_risk})
    assert r.status_code == 200
    data = r.json()
    assert data['success'] == True
    pred = data['prediction']
    print(f"✓ Segment: {pred['segment']} ({pred['segment_label']})")
    print(f"✓ Churn prob: {pred['churn_probability']:.2%}")
    print(f"✓ Action: {data['recommended_action']} (Priority: {data['priority_score']:.1%})")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 4: What-If simulation
print("\n[4] WHAT-IF SIMULATION (Tenure Extension)")
try:
    r = requests.post(f"{BASE_URL}/api/what-if",
        json={
            'customer': customer_high_churn_risk,
            'modifications': {'tenure': 36, 'Contract': 'Two year'}
        })
    assert r.status_code == 200
    data = r.json()
    assert data['success'] == True
    orig_churn = data['original_prediction']['churn_probability']
    new_churn = data['modified_prediction']['churn_probability']
    delta = data['delta']['churn_probability_delta']
    print(f"✓ Original churn prob: {orig_churn:.2%}")
    print(f"✓ Modified churn prob: {new_churn:.2%}")
    print(f"✓ Delta: {delta:.2%} (improvement: {abs(delta):.2%})")
    assert delta < 0  # Should improve (reduce) churn risk
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 5: Policy change scenarios
print("\n[5] POLICY CHANGE SCENARIOS")
try:
    r = requests.get(f"{BASE_URL}/api/what-if/policy-changes")
    assert r.status_code == 200
    data = r.json()
    scenarios = data['scenarios']
    print(f"✓ {len(scenarios)} pre-defined scenarios available:")
    for s in scenarios[:3]:
        print(f"  - {s['name']}")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 6: Model info
print("\n[6] MODEL INFORMATION")
try:
    r = requests.get(f"{BASE_URL}/api/explanations/model-info")
    assert r.status_code == 200
    data = r.json()
    metrics = data['training_metrics']
    print(f"✓ Model: {data['model_type']}")
    print(f"✓ Estimators: {data['n_estimators']}")
    print(f"✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ Test Precision: {metrics['precision']:.4f}")
    print(f"✓ Test Recall: {metrics['recall']:.4f}")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 7: Batch template
print("\n[7] BATCH PREDICTION TEMPLATE")
try:
    r = requests.get(f"{BASE_URL}/api/predict-batch/template")
    assert r.status_code == 200
    data = r.json()
    print(f"✓ Template columns: {len(data['columns'])}")
    print(f"✓ Max batch size: {data['max_rows']:,}")
    print(f"✓ Format: {data['expected_format']}")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Test 8: Documentation
print("\n[8] DOCUMENTATION ENDPOINTS")
try:
    r_swagger = requests.get(f"{BASE_URL}/docs")
    r_redoc = requests.get(f"{BASE_URL}/redoc")
    assert r_swagger.status_code == 200
    assert r_redoc.status_code == 200
    print(f"✓ Swagger docs available at /docs")
    print(f"✓ ReDoc docs available at /redoc")
    tests_passed += 1
except Exception as e:
    print(f"✗ Failed: {e}")
    tests_failed += 1

# Summary
print("\n" + "=" * 80)
print(f"TEST RESULTS: {tests_passed} passed, {tests_failed} failed")
print("=" * 80)

if tests_failed == 0:
    print("\n✅ ALL TESTS PASSED - API IS PRODUCTION READY")
    print("\nAPI is accessible at: http://localhost:8001")
    print("Interactive API docs: http://localhost:8001/docs")
    print("\nKey endpoints:")
    print("  • POST /api/predict - Single customer prediction")
    print("  • POST /api/predict-batch -Batch prediction from CSV")
    print("  • POST /api/what-if - Feature perturbation analysis")
    print("  • GET /api/feature-importance/global - Global feature importance")
    print("  • GET /api/what-if/policy-changes - Pre-defined scenarios")
else:
    print(f"\n⚠️  {tests_failed} tests failed")
    sys.exit(1)
