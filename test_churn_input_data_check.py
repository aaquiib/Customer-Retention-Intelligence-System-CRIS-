"""Quick check of data structure before training."""
import pandas as pd
from src.config import load_config

cfg = load_config()

# Check if df_with_segment_labels exists and has correct columns
try:
    df = pd.read_csv(
        cfg['data']['processed_csv_path'].replace('processed_df', 'df_with_segment_labels')
    )
    
    print("✓ df_with_segment_labels.csv loaded successfully")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in sorted(df.columns):
        print(f"  - {col}")
    
    # Check required columns
    churn_cfg = cfg['churn_modeling']
    required_cols = churn_cfg['numeric_columns'] + churn_cfg['categorical_columns'] + ['Churn', 'segment_label']
    
    print(f"\n\nConfiguration expects (20+2):")
    print(f"  Numeric: {churn_cfg['numeric_columns']}")
    print(f"  Categorical: {churn_cfg['categorical_columns']}")
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"\n✗ MISSING COLUMNS: {missing}")
    else:
        print(f"\n✓ All required columns present!")
        
except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
