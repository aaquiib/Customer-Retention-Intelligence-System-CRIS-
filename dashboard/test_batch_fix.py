"""Test the fixed aggregate_batch_summary function."""
import sys
sys.path.insert(0, '.')

from utils.data_processors import aggregate_batch_summary
import pandas as pd

print("Testing aggregate_batch_summary with dict recommended_action...")

# Simulate predictions with dict-based recommended_action (as API returns)
predictions = [
    {
        'segment': 0,
        'segment_label': 'Long-term Loyal',
        'churn_probability': 0.1,
        'is_churner': False,
        'recommended_action': {'action_label': 'Monitor', 'priority_score': 0.27, 'reason': 'Low risk'}
    },
    {
        'segment': 1,
        'segment_label': 'Low Engagement',
        'churn_probability': 0.6,
        'is_churner': True,
        'recommended_action': {'action_label': 'Win-back', 'priority_score': 0.85, 'reason': 'High churn risk'}
    },
    {
        'segment': 0,
        'segment_label': 'Long-term Loyal',
        'churn_probability': 0.15,
        'is_churner': False,
        'recommended_action': {'action_label': 'Cross-sell', 'priority_score': 0.40, 'reason': 'Upsell opportunity'}
    },
]

try:
    summary = aggregate_batch_summary(predictions)
    print("✅ SUCCESS!")
    print(f"\nBatch Summary:")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Processed: {summary['rows_processed']}")
    print(f"  Churn rate: {summary['churn_rate']:.2%}")
    print(f"  Avg churn prob: {summary['avg_churn_probability']:.4f}")
    print(f"  Segment distribution: {summary['segment_distribution']}")
    print(f"  Action distribution: {summary['action_distribution']}")
    
    if 'Win-back' in summary['action_distribution']:
        print(f"\n✅ Action labels extracted correctly!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
