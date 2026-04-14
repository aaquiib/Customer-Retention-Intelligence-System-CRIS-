#!/usr/bin/env python
"""Test data processors logic."""
import sys
sys.path.insert(0, '.')
import pandas as pd
from utils.data_processors import (
    aggregate_batch_summary,
    prepare_batch_result_df,
    build_segment_stats
)

print("Testing data_processors functions...\n")

# Test with sample data
sample_predictions = [
    {
        'segment': 0,
        'segment_label': 'Long-term Loyal',
        'segment_confidence': 0.95,
        'churn_probability': 0.1,
        'is_churner': False,
        'recommended_action': 'Retention Call',
        'input_features': {
            'MonthlyCharges': 65.50,
            'tenure': 24,
            'Contract': 'Month-to-month'
        },
        'top_features': [
            {'feature_name': 'tenure', 'shap_value': -0.5},
            {'feature_name': 'MonthlyCharges', 'shap_value': 0.2}
        ],
        'error': None
    },
    {
        'segment': 1,
        'segment_label': 'Low Engagement',
        'segment_confidence': 0.87,
        'churn_probability': 0.75,
        'is_churner': True,
        'recommended_action': 'Service Upgrade',
        'input_features': {
            'MonthlyCharges': 45.25,
            'tenure': 3,
            'Contract': 'Month-to-month'
        },
        'top_features': [
            {'feature_name': 'tenure', 'shap_value': 0.6},
            {'feature_name': 'Contract', 'shap_value': 0.3}
        ],
        'error': None
    }
]

try:
    summary = aggregate_batch_summary(sample_predictions)
    print(f"✅ aggregate_batch_summary works")
    print(f"   Total rows: {summary['total_rows']}, Churn rate: {summary['churn_rate']:.2%}")
except Exception as e:
    print(f"❌ aggregate_batch_summary: {str(e)[:100]}")

try:
    df = prepare_batch_result_df(sample_predictions)
    print(f"✅ prepare_batch_result_df works")
    print(f"   Created {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"❌ prepare_batch_result_df: {str(e)[:100]}")

try:
    # Create test dataframe for build_segment_stats
    test_df = pd.DataFrame({
        'segment': [0, 0, 1, 1],
        'is_churner': [False, False, True, True],
        'tenure': [24, 36, 3, 5],
        'MonthlyCharges': [65.5, 80.0, 45.25, 50.0],
        'segment_confidence': [0.95, 0.92, 0.87, 0.85],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year'],
        'InternetService': ['Fiber optic', 'DSL', 'Fiber optic', 'DSL']
    })
    
    stats = build_segment_stats(test_df, 0)
    print(f"✅ build_segment_stats works")
    print(f"   Segment 0: size={stats['size']}, churn_rate={stats['churn_rate']:.2%}")
except Exception as e:
    print(f"❌ build_segment_stats: {str(e)[:100]}")

print("\nData processor tests completed!")
