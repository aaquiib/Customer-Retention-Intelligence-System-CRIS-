"""
Extract and save segmentation model artifacts.
Reproduces the KPrototypes model from notebooks/segmentation.ipynb and saves all artifacts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("Loading processed data...")
df = pd.read_csv("data/processed/processed_df.csv")
print(f"Loaded data with shape: {df.shape}")

# ============================================================================
# 2. FEATURE ENGINEERING (replicate from notebook)
# ============================================================================

print("\nPerforming feature engineering...")

# Billing / Value features
df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))
df['charge_gap'] = df['MonthlyCharges'] - df['avg_monthly_spend']
df['is_high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

# Tenure band
df['tenure_band'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 36, 72],
    labels=['0-12', '12-36', '36+'],
    right=True
)
df['tenure_band'] = df['tenure_band'].astype(str)

# Service usage counts
streaming_services = ['StreamingTV', 'StreamingMovies']
df['streaming_count'] = ((df[streaming_services] == 'Yes') | (df[streaming_services] == 'yes')).sum(axis=1)

security_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
df['security_count'] = ((df[security_services] == 'Yes') | (df[security_services] == 'yes')).sum(axis=1)

# Contract / payment patterns
df['month_to_month_paperless'] = (
    (df['Contract'] == 'Month-to-month') &
    (df['PaperlessBilling'] == 'Yes')
).astype(int)

df['payment_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

# Vulnerability flags
df['no_support_services'] = ((df['TechSupport'] == 'No') & (df['OnlineSecurity'] == 'No')).astype(int)
df['is_isolated'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
df['fiber_no_security'] = ((df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No')).astype(int)
df['no_internet_services'] = (df['InternetService'] == 'No').astype(int)

print("Feature engineering complete.")

# ============================================================================
# 3. SELECT SEGMENTATION FEATURES
# ============================================================================

print("\nPreparing segmentation features...")

segmentation_features = [
    # Demographics (categorical)
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    # Lifecycle (numeric + categorical)
    'tenure', 'tenure_band',
    # Value (numeric)
    'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'charge_gap', 'is_high_value',
    # Services (mixed)
    'PhoneService', 'MultipleLines', 'InternetService',
    'streaming_count', 'security_count',
    # Contract / payment (mixed)
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'payment_electronic_check', 'month_to_month_paperless',
    # Risk flags (numeric)
    'no_support_services', 'is_isolated', 'fiber_no_security',
    'no_internet_services'
]

data_seg = df[segmentation_features].copy()
print(f"Segmentation features prepared. Shape: {data_seg.shape}")
print(f"Features: {len(segmentation_features)}")

# ============================================================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================================

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_gap", 
            "streaming_count", "security_count", "is_high_value", "payment_electronic_check", 
            "month_to_month_paperless", "no_support_services", "is_isolated", 
            "fiber_no_security", "no_internet_services", "SeniorCitizen"]

cat_cols = [c for c in data_seg.columns if c not in num_cols]

print(f"\nNumeric columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ============================================================================
# 5. SCALE NUMERIC FEATURES
# ============================================================================

print("\nScaling numeric features...")
scaler = StandardScaler()
data_seg_scaled = data_seg.copy()
data_seg_scaled[num_cols] = scaler.fit_transform(data_seg[num_cols])

# Convert categorical to string
for c in cat_cols:
    data_seg_scaled[c] = data_seg_scaled[c].astype(str)

print("Scaling complete.")

# ============================================================================
# 6. GET CATEGORICAL INDICES FOR KPROTOTYPES
# ============================================================================

cat_idx = [data_seg_scaled.columns.get_loc(c) for c in cat_cols]
print(f"\nCategorical indices: {cat_idx}")

# ============================================================================
# 7. ELBOW METHOD (to capture elbow curve data)
# ============================================================================

print("\nPerforming elbow analysis...")
X = data_seg_scaled.to_numpy()
ks = range(2, 11)
costs = []
elbow_data = {}

for k in ks:
    kproto = KPrototypes(n_clusters=k, init="Cao", n_init=5, verbose=0, random_state=42)
    kproto.fit_predict(X, categorical=cat_idx)
    cost = kproto.cost_
    costs.append(cost)
    elbow_data[str(k)] = float(cost)
    print(f"  k={k}: cost={cost:.4f}")

print("Elbow analysis complete.")

# ============================================================================
# 8. TRAIN FINAL KPROTOTYPES MODEL
# ============================================================================

print("\nTraining final KPrototypes model (k=4)...")
final_k = 4

kproto_final = KPrototypes(
    n_clusters=final_k,
    init="Cao",
    n_init=10,
    verbose=0,
    random_state=42
)

cluster_labels = kproto_final.fit_predict(X, categorical=cat_idx)
print(f"Model trained. Cluster distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

# ============================================================================
# 9. COMPUTE CLUSTER STATISTICS
# ============================================================================

print("\nComputing cluster statistics...")
df_with_clusters = df.copy()
df_with_clusters['segment'] = cluster_labels

cluster_stats = {}
for cluster_id in sorted(df_with_clusters['segment'].unique()):
    cluster_data = df_with_clusters[df_with_clusters['segment'] == cluster_id]
    stats = {
        'cluster_id': int(cluster_id),
        'size': int(cluster_data.shape[0]),
        'size_pct': float(cluster_data.shape[0] / len(df_with_clusters) * 100),
        'churn_rate': float((cluster_data['Churn'] == 1).mean()),
        'tenure_median': float(cluster_data['tenure'].median()),
        'monthly_charges_median': float(cluster_data['MonthlyCharges'].median()),
        'total_charges_median': float(cluster_data['TotalCharges'].median()),
        'avg_monthly_spend_median': float(cluster_data['avg_monthly_spend'].median()),
    }
    cluster_stats[str(cluster_id)] = stats
    print(f"  Cluster {cluster_id}: {stats['size']} customers, {stats['churn_rate']:.1%} churn")

# ============================================================================
# 10. DEFINE SEGMENT LABELS AND DESCRIPTIONS
# ============================================================================

segment_labels_map = {
    "3": {
        "label": "At risk High-value 🚨",
        "cluster_id": 3,
        "description": "High-value customers showing churn risk signals. Require immediate retention focus."
    },
    "0": {
        "label": "Loyal High-Value 💎",
        "cluster_id": 0,
        "description": "Premium customers with strong loyalty and high spending. Nurture for expansion."
    },
    "2": {
        "label": "Stable Mid-Value 👍",
        "cluster_id": 2,
        "description": "Stable mid-tier customers with consistent behavior. Good cross-sell opportunity."
    },
    "1": {
        "label": "Low Engagement ⚠️",
        "cluster_id": 1,
        "description": "Customers with limited service adoption and engagement. Need activation strategy."
    }
}

print("\nSegment labels:")
for cluster_id, label_info in segment_labels_map.items():
    print(f"  Cluster {cluster_id}: {label_info['label']}")

# ============================================================================
# 11. SAVE ARTIFACTS
# ============================================================================

artifact_dir = "models/v1.0.0-segmentation-20260411_224248"
print(f"\nSaving artifacts to {artifact_dir}...")

# 11a. Save KPrototypes model
model_path = os.path.join(artifact_dir, "kprototypes", "model.pkl")
joblib.dump(kproto_final, model_path)
print(f"  ✓ Saved KPrototypes model to {model_path}")

# 11b. Save StandardScaler
scaler_path = os.path.join(artifact_dir, "preprocessing", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"  ✓ Saved StandardScaler to {scaler_path}")

# 11c. Save feature configuration
feature_config = {
    "segmentation_features": segmentation_features,
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "categorical_indices": cat_idx,
    "total_features": len(segmentation_features),
    "numeric_count": len(num_cols),
    "categorical_count": len(cat_cols),
    "scaler_mean": {f: float(m) for f, m in zip(num_cols, scaler.mean_)},
    "scaler_std": {f: float(s) for f, s in zip(num_cols, scaler.scale_)},
}

feature_config_path = os.path.join(artifact_dir, "preprocessing", "feature_config.json")
with open(feature_config_path, 'w') as f:
    json.dump(feature_config, f, indent=2)
print(f"  ✓ Saved feature configuration to {feature_config_path}")

# 11d. Save metadata
metadata = {
    "model_type": "KPrototypes",
    "model_name": "Customer Segmentation (KPrototypes)",
    "version": "v1.0.0-segmentation-20260411_224248",
    "training_date": datetime.now().isoformat(),
    "hyperparameters": {
        "n_clusters": final_k,
        "init": "Cao",
        "n_init": 10,
        "random_state": 42,
        "verbose": 0,
    },
    "training_data": {
        "total_samples": int(len(df_with_clusters)),
        "features": len(segmentation_features),
        "numeric_features": len(num_cols),
        "categorical_features": len(cat_cols),
    },
    "elbow_curve": elbow_data,
    "selected_k": final_k,
    "cluster_statistics": cluster_stats,
    "segment_labels": segment_labels_map,
}

metadata_path = os.path.join(artifact_dir, "metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Saved metadata to {metadata_path}")

# 11e. Save categorical indices separately for quick reference
cat_indices_path = os.path.join(artifact_dir, "preprocessing", "categorical_indices.json")
with open(cat_indices_path, 'w') as f:
    json.dump({
        "categorical_indices": cat_idx,
        "categorical_columns": cat_cols,
        "total_feature_count": len(segmentation_features),
    }, f, indent=2)
print(f"  ✓ Saved categorical indices to {cat_indices_path}")

# 11f. Save segment labels separately for easy lookup
labels_path = os.path.join(artifact_dir, "metadata", "segment_labels.json")
with open(labels_path, 'w') as f:
    json.dump(segment_labels_map, f, indent=2)
print(f"  ✓ Saved segment labels to {labels_path}")

print(f"\n✓ All artifacts saved successfully to {artifact_dir}")
print(f"  - KPrototypes model: kprototypes/model.pkl")
print(f"  - StandardScaler: preprocessing/scaler.pkl")
print(f"  - Feature config: preprocessing/feature_config.json")
print(f"  - Categorical indices: preprocessing/categorical_indices.json")
print(f"  - Metadata: metadata.json")
print(f"  - Segment labels: metadata/segment_labels.json")

# ============================================================================
# 12. SAVE LABELED DATA FOR REFERENCE
# ============================================================================

print("\nSaving labeled data for reference...")
df_labeled = df_with_clusters.copy()
df_labeled['segment_label'] = df_labeled['segment'].map(
    {int(k): v['label'] for k, v in segment_labels_map.items()}
)

labeled_data_path = "data/processed/df_with_segment_labels.csv"
df_labeled.to_csv(labeled_data_path, index=False)
print(f"  ✓ Saved labeled data to {labeled_data_path}")

print("\n" + "="*70)
print("SEGMENTATION MODEL EXTRACTION COMPLETE")
print("="*70)
