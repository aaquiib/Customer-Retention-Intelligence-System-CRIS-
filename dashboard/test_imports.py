#!/usr/bin/env python
"""Test all imports."""
import sys
sys.path.insert(0, '.')

pages_to_test = [
    'page_01_overview',
    'page_02_single_prediction',
    'page_03_batch_scoring',
    'page_04_segment_intelligence',
    'page_05_churn_risk',
    'page_06_action_planning',
    'page_07_what_if_simulator',
    'page_08_explainability',
    'page_09_model_health'
]

print("Testing page imports...\n")

for page_name in pages_to_test:
    try:
        module = __import__(f'pages.{page_name}', fromlist=[page_name])
        print(f'✅ {page_name} loads')
    except Exception as e:
        print(f'❌ {page_name}: {str(e)[:100]}')

print("\nAll tests completed!")
