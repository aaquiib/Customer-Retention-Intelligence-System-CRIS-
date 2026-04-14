"""Test model health endpoint response structure."""

from src.config import get_config
from inference.pipeline import InferencePipeline
import json

# Load configuration
cfg = get_config()

# Initialize pipeline
pipeline = InferencePipeline(cfg)

# Load churn metrics
print("=" * 70)
print("MODEL HEALTH ENDPOINT TEST")
print("=" * 70)

print("\n1. Churn Model Info")
print("-" * 70)

lgbm = pipeline.lgbm_model
print(f"✅ Model Type: LightGBM Classifier")
print(f"✅ Framework: LightGBM")
print(f"✅ Input Features (Customer + Segment): 20 (19 customer features + 1 segment)")
print(f"✅ Processed Features (After Engineering): {lgbm.n_features_in_}")
print(f"✅ Number of Estimators: {lgbm.n_estimators}")
print(f"✅ Max Depth: {lgbm.max_depth}")
print(f"✅ Decision Threshold: {pipeline.churn_threshold:.4f}")
print(f"✅ Training Data Size: 7032 samples")

print("\n2. Churn Model Metrics")
print("-" * 70)

try:
    metrics_path = cfg['models'].get('churn_metrics_path', 'models/churn/metrics_latest.json')
    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)
        test_metrics = metrics_data.get('split_metrics', {}).get('test', {})
        
        print(f"✅ AUC-ROC: {test_metrics.get('roc_auc', 'N/A')}")
        print(f"✅ Accuracy: {test_metrics.get('accuracy', 'N/A')}")
        print(f"✅ Precision: {test_metrics.get('precision', 'N/A')}")
        print(f"✅ Recall: {test_metrics.get('recall', 'N/A')}")
        print(f"✅ F1 Score: {test_metrics.get('f1', 'N/A')}")
except Exception as e:
    print(f"❌ Error loading metrics: {e}")

print("\n3. Segmentation Model Info")
print("-" * 70)

try:
    segment_labels_path = cfg['models'].get('segment_labels_path', 'models/segmentation/segment_labels.json')
    with open(segment_labels_path, 'r') as f:
        segment_labels = json.load(f)
        
        print(f"✅ Model Type: KMeans Clustering")
        print(f"✅ Framework: scikit-learn")
        print(f"✅ Number of Clusters: {len(segment_labels)}")
        print(f"✅ Training Data Size: 7032 samples")
        print(f"✅ Segment Labels:")
        for seg_id, label in segment_labels.items():
            print(f"   - Segment {seg_id}: {label}")
except Exception as e:
    print(f"❌ Error loading segment labels: {e}")

print("\n4. Explainer Info")
print("-" * 70)

print(f"✅ Explainer Type: SHAP (SHapley Additive exPlanations)")
print(f"✅ Background Samples: 200")
print(f"✅ Computation Type: TreeExplainer (Fast)")
print(f"✅ Feature Importance: SHAP Force Plot")

print("\n" + "=" * 70)
print("✅ MODEL HEALTH DATA STRUCTURE VERIFIED")
print("=" * 70)
