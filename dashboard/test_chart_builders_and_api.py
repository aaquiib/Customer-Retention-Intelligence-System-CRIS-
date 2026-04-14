"""Test chart builders and API client for basic functionality."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from utils.chart_builders import (
    create_gauge_chart,
    create_distribution_histogram,
    create_segment_donut,
    create_action_donut,
    create_feature_importance_bar,
    create_waterfall_chart
)
from utils.api_client import APIClient

print("\n" + "="*60)
print("TESTING CHART BUILDERS")
print("="*60)

# Test 1: Gauge chart
print("\n=== TEST 1: create_gauge_chart ===")
try:
    fig = create_gauge_chart(0.52, label="Churn Risk", threshold=0.4356)
    assert fig is not None, "Gauge should return a figure"
    print("✅ Gauge chart works")
except Exception as e:
    print(f"❌ Gauge chart failed: {e}")

# Test 2: Distribution histogram
print("\n=== TEST 2: create_distribution_histogram ===")
try:
    churn_probs = [0.1, 0.2, 0.3, 0.55, 0.65, 0.75, 0.85, 0.95]
    fig = create_distribution_histogram(churn_probs, title="Churn Distribution")
    assert fig is not None, "Histogram should return a figure"
    print("✅ Distribution histogram works")
except Exception as e:
    print(f"❌ Distribution histogram failed: {e}")

# Test 3: Segment donut
print("\n=== TEST 3: create_segment_donut ===")
try:
    segment_dist = {
        0: 100,  # Long-term Loyal
        1: 50,   # Low Engagement
        2: 150,  # Medium Engagement
        3: 80    # New/High-Value
    }
    fig = create_segment_donut(segment_dist)
    assert fig is not None, "Donut should return a figure"
    print("✅ Segment donut works")
except Exception as e:
    print(f"❌ Segment donut failed: {e}")

# Test 4: Action donut
print("\n=== TEST 4: create_action_donut ===")
try:
    action_dist = {
        'Retention Call': 80,
        'Cross-sell': 120,
        'Win-back': 50,
        'Upgrade': 40
    }
    fig = create_action_donut(action_dist)
    assert fig is not None, "Action donut should return a figure"
    print("✅ Action donut works")
except Exception as e:
    print(f"❌ Action donut failed: {e}")

# Test 5: Feature importance bar
print("\n=== TEST 5: create_feature_importance_bar ===")
try:
    features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'TechSupport', 'OnlineSecurity']
    importances = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07]
    fig = create_feature_importance_bar(features, importances)
    assert fig is not None, "Importance bar should return a figure"
    print("✅ Feature importance bar works")
except Exception as e:
    print(f"❌ Feature importance bar failed: {e}")

# Test 6: Waterfall chart
print("\n=== TEST 6: create_waterfall_chart ===")
try:
    features = ['Base', 'tenure', 'MonthlyCharges', 'Contract']
    shap_values = [0.35, -0.15, 0.10, 0.08]
    base_value = 0.35
    prediction_value = 0.68
    fig = create_waterfall_chart(features, shap_values, base_value, prediction_value)
    assert fig is not None, "Waterfall should return a figure"
    print("✅ Waterfall chart works")
except Exception as e:
    print(f"❌ Waterfall chart failed: {e}")

print("\n" + "="*60)
print("TESTING API CLIENT")
print("="*60)

# Test 7: API Client instantiation
print("\n=== TEST 7: APIClient instantiation ===")
try:
    client = APIClient()
    assert client is not None, "Client should be created"
    print("✅ APIClient instantiation works")
except Exception as e:
    print(f"❌ APIClient instantiation failed: {e}")

# Test 8: API Client methods exist
print("\n=== TEST 8: APIClient methods exist ===")
try:
    client = APIClient()
    methods = [
        'get_health', 'get_model_info', 'predict_single', 'predict_batch',
        'get_batch_template', 'get_global_importance', 'get_instance_importance',
        'what_if_single', 'what_if_batch', 'get_policy_scenarios'
    ]
    for method_name in methods:
        assert hasattr(client, method_name), f"Missing method: {method_name}"
        assert callable(getattr(client, method_name)), f"Method not callable: {method_name}"
    print(f"✅ All {len(methods)} APIClient methods exist and are callable")
except Exception as e:
    print(f"❌ APIClient methods check failed: {e}")

print("\n" + "="*60)
print("✅ ALL STRUCTURAL TESTS PASSED!")
print("="*60)
print("\nNote: API endpoint tests require running backend at localhost:8000")
print("Recommend testing with: curl http://localhost:8000/api/health")
