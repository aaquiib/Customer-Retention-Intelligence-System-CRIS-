"""Comprehensive test of data processors with edge cases."""
import sys
sys.path.insert(0, '.')

import pandas as pd
from utils.data_processors import (
    aggregate_batch_summary,
    prepare_batch_result_df,
    build_segment_stats,
    get_top_customers_by_risk,
    segment_filter,
    risk_band_filter,
    calculate_revenue_at_risk
)

# Test 1: build_segment_stats with full data
print("\n=== TEST 1: build_segment_stats with real data ===")
test_df = pd.DataFrame({
    'customerID': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
    'segment': [0, 0, 1, 1],
    'segment_label': ['Long-term Loyal', 'Long-term Loyal', 'Low Engagement', 'Low Engagement'],
    'segment_confidence': [0.95, 0.88, 0.92, 0.85],
    'churn_probability': [0.1, 0.15, 0.6, 0.65],
    'is_churner': [False, False, True, True],
    'recommended_action': ['Retention Call', 'Cross-sell', 'Win-back', 'Win-back'],
    'tenure': [24, 36, 6, 8],
    'MonthlyCharges': [65.50, 75.00, 85.00, 90.00],
    'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year'],
    'InternetService': ['Fiber optic', 'DSL', 'Fiber optic', 'DSL'],
})

for seg in [0, 1, 2, 3]:  # Test all segments including empty ones
    stats = build_segment_stats(test_df, seg)
    print(f"  Segment {seg}: {stats}")
    assert isinstance(stats, dict), f"Expected dict, got {type(stats)}"
    assert 'size' in stats, "Missing 'size' key"
    assert 'churn_rate' in stats, "Missing 'churn_rate' key"

print("✅ build_segment_stats works with real data and empty segments")

# Test 2: Edge case - empty DataFrame
print("\n=== TEST 2: build_segment_stats with empty DataFrame ===")
empty_df = pd.DataFrame()
stats = build_segment_stats(empty_df, 0)
print(f"  Empty DF result: {stats}")
assert stats['size'] == 0, "Empty DF should have size 0"
print("✅ Handles empty DataFrame correctly")

# Test 3: Edge case - DataFrame without segment column
print("\n=== TEST 3: build_segment_stats with missing segment column ===")
df_no_segment = pd.DataFrame({
    'customerID': ['CUST001'],
    'tenure': [24],
})
stats = build_segment_stats(df_no_segment, 0)
print(f"  Missing segment column result: {stats}")
assert stats['size'] == 0, "Should return 0 size when no segment column"
print("✅ Handles missing segment column correctly")

# Test 4: aggregate_batch_summary
print("\n=== TEST 4: aggregate_batch_summary ===")
predictions = [
    {'segment': 0, 'churn_probability': 0.1, 'is_churner': False, 'recommended_action': 'Retention Call'},
    {'segment': 0, 'churn_probability': 0.6, 'is_churner': True, 'recommended_action': 'Win-back'},
    {'segment': 1, 'churn_probability': 0.5, 'is_churner': True, 'recommended_action': 'Win-back'},
]
summary = aggregate_batch_summary(predictions)
print(f"  Summary: {summary}")
assert summary['total_rows'] == 3, f"Expected 3 rows, got {summary['total_rows']}"
assert abs(summary['churn_rate'] - 0.6667) < 0.01, "Churn rate should be ~67%"
print("✅ aggregate_batch_summary calculates correctly")

# Test 5: prepare_batch_result_df
print("\n=== TEST 5: prepare_batch_result_df ===")
result_df = prepare_batch_result_df(predictions)
print(f"  Created {len(result_df)} rows × {len(result_df.columns)} columns")
assert len(result_df) == 3, f"Expected 3 rows, got {len(result_df)}"
assert 'churn_probability' in result_df.columns, "Missing churn_probability column"
print("✅ prepare_batch_result_df creates correct structure")

# Test 6: get_top_customers_by_risk
print("\n=== TEST 6: get_top_customers_by_risk ===")
test_df['is_churner'] = [False, False, True, True]
test_df['churn_probability'] = [0.1, 0.15, 0.6, 0.65]
top_customers = get_top_customers_by_risk(test_df, n=2)
print(f"  Top 2 customers: {top_customers.to_dict('records')}")
assert len(top_customers) == 2, f"Expected 2 rows, got {len(top_customers)}"
assert top_customers.iloc[0]['churn_probability'] >= top_customers.iloc[1]['churn_probability'], "Not sorted by risk"
print("✅ get_top_customers_by_risk works correctly")

# Test 7: segment_filter
print("\n=== TEST 7: segment_filter ===")
filtered = segment_filter(test_df, segment_ids=[0])
print(f"  Filtered to segment 0: {len(filtered)} rows")
assert len(filtered) == 2, f"Expected 2 rows for segment 0, got {len(filtered)}"
assert all(filtered['segment'] == 0), "Not all rows are segment 0"
print("✅ segment_filter works correctly")

# Test 8: risk_band_filter
print("\n=== TEST 8: risk_band_filter ===")
filtered = risk_band_filter(test_df, risk_bands=['High'])  # > 0.65
print(f"  Filtered to High risk: {len(filtered)} rows")
assert len(filtered) == 1, f"Expected 1 high risk row, got {len(filtered)}"
print("✅ risk_band_filter works correctly")

# Test 9: calculate_revenue_at_risk
print("\n=== TEST 9: calculate_revenue_at_risk ===")
risk_value = calculate_revenue_at_risk(test_df)
print(f"  Revenue at risk: ${risk_value:.2f}")
assert risk_value > 0, "Revenue at risk should be positive for churners"
print("✅ calculate_revenue_at_risk works correctly")

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED!")
print("="*60)
