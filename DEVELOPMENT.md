# Development Guide

This document explains the project structure and how to extend the codebase.

## Architecture Overview

```
Data Ingestion
    ↓
Data Preprocessing (src/data/)
    ↓
Feature Engineering (src/features/)
    ↓
├─→ Segmentation Module (src/segmentation/)
│   ├─ train_segments.py (fit K-Prototypes)
│   └─ assign_segments.py (predict clusters)
│
└─→ Churn Module (src/churn/)
    ├─ train.py (fit LightGBM + threshold)
    └─ evaluate.py (metrics)

Configuration & Utils (src/config.py, src/utils/)
```

## Module Responsibilities

### `src/data/`
**Responsibility**: Raw data → Cleaned, typed DataFrame

- `ingest.py`: Load raw CSV, validate file exists
- `preprocess.py`: Handle missing values, type conversions, drop anomalies

**Key Functions**:
```python
from src.data import load_raw_data, preprocess_data
df = load_raw_data(config['data']['raw_csv_path'])
df_clean = preprocess_data(df, config)
```

### `src/features/`
**Responsibility**: Cleaned data → Engineered features (20+ new columns)

- `engineering.py`: Create business-relevant features (billing, tenure, risk flags)
- `build_features.py`: Orchestrate preprocessing + engineering pipeline

**Key Functions**:
```python
from src.features import engineer_features
df_engineered = engineer_features(df_clean, config)
```

### `src/segmentation/`
**Responsibility**: Train and assign customer segments

- `train_segments.py`: Fit K-Prototypes (k=4) on engineered features
  - Saves: `kproto.pkl`, `scaler.pkl`, `catidx.json`, `feature_metadata.json`
  - Validates feature consistency

- `assign_segments.py`: Use trained model to assign clusters
  - Loads artifacts, validates features, predicts clusters, maps to persona names

**Key Functions**:
```python
from src.segmentation import train_segmentation_model, assign_segments
kproto, scaler, cat_idx, meta = train_segmentation_model(df_engineered, config)
df_with_segments = assign_segments(df_engineered, config)
```

### `src/churn/`
**Responsibility**: Train and evaluate churn prediction model

- `train.py`: Fit LightGBM on engineered + segmented data
  - Splits: 70% train | 15% val | 15% test (stratified)
  - Preprocessing: StandardScaler (numeric) + OneHotEncoder (categorical)
  - Threshold optimization: Find best F1 threshold
  - Saves: `lgbm_churn_model.pkl`, `preprocessor.pkl`, `threshold_meta.json`

- `evaluate.py`: Compute metrics (ROC-AUC, Precision, Recall, F1, Confusion Matrix)

**Key Functions**:
```python
from src.churn import train_churn_model, evaluate_model
lgbm, preprocessor, threshold, meta = train_churn_model(df_with_segments, config)
metrics = evaluate_model(y_true, y_pred, y_proba)
```

### `src/utils/`
**Responsibility**: Cross-module utilities

- `io_utils.py`: CSV, JSON, model I/O (centralizes all file operations)
- `logging_config.py`: Setup logging across all modules
- `feature_validation.py`: Feature consistency checks (from segmentation notebook)

## Adding New Features

### Example: Add a New Business Feature

1. **Edit** `src/features/engineering.py`:
```python
def engineer_features(df, cfg):
    # ... existing code ...
    
    # NEW FEATURE: Churn_likelihood_score (custom metric)
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['churn_likelihood_score'] = (
            (df['MonthlyCharges'] / df['MonthlyCharges'].max()) * 
            (1 / (df['tenure'] + 1))  # Newer customers = higher score
        )
        logger.info("Created churn_likelihood_score feature")
```

2. **Add to config** `config/config.yaml`:
```yaml
feature_engineering:
  # ... existing ...
  churn_likelihood_scale: true  # Enable/disable feature
```

3. **Add to segmentation config** if used in clustering:
```yaml
segmentation:
  numeric_features:
    - ... existing ...
    - 'churn_likelihood_score'  # Add here
```

4. **Test** in `tests/test_features_engineering.py`:
```python
def test_engineer_creates_churn_likelihood_score(sample_data):
    result = engineer_features(sample_data, cfg)
    assert 'churn_likelihood_score' in result.columns
    assert result['churn_likelihood_score'].between(0, 1).all()
```

5. **Run**:
```bash
python -m src.features.build_features
pytest tests/test_features_engineering.py -v
```

### Example: Add a New Model

1. **Create** `src/churn/train_xgboost.py` (alternative to LightGBM):
```python
from xgboost import XGBClassifier

def train_xgboost_model(df, cfg):
    """Train XGBoost alternative."""
    # ... setup code ...
    xgb = XGBClassifier(**cfg['churn_modeling']['xgb_hyperparams'])
    xgb.fit(X_train_enc, y_train)
    # ... save model ...
    return xgb, preprocessor, threshold, meta
```

2. **Add to** `src/churn/__init__.py`:
```python
from src.churn.train_xgboost import train_xgboost_model
__all__ = [..., 'train_xgboost_model']
```

3. **Update** config with XGBoost hyperparams:
```yaml
churn_modeling:
  xgb_hyperparams:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 7
```

## Common Workflows

### Workflow 1: Tune Segmentation Hyperparameters

```python
from src.config import load_config
from src.utils import load_csv, setup_logging, save_json
from src.segmentation import train_segmentation_model

cfg = load_config()
setup_logging(cfg['logging'])
df = load_csv(cfg['data']['segmentation_features_path'])

# Test different k values
results = {}
for k in [3, 4, 5, 6]:
    cfg['segmentation']['n_clusters'] = k
    kproto, scaler, cat_idx, meta = train_segmentation_model(df, cfg)
    results[k] = kproto.cost_

# Save results
save_json(results, 'kprototypes_elbow.json')
```

### Workflow 2: Compare Churn Model Thresholds

```python
from src.churn import evaluate_model, compare_thresholds
from sklearn.metrics import roc_auc_score

# After training, on test set
y_proba = lgbm.predict_proba(X_test_enc)[:, 1]
y_true = y_test

# Compare multiple thresholds
comparison = compare_thresholds(y_true, y_proba, [0.3, 0.4, 0.5, 0.6, 0.7])

for threshold, results in comparison.items():
    print(f"Threshold {threshold:.1f}: F1={results['metrics']['f1']:.4f}")
```

### Workflow 3: Evaluate on Specific Segment

```python
from src.utils import load_csv
from src.churn import evaluate_model

# Load data with segments
df = load_csv(cfg['data']['processed_csv_path'].replace('processed', 'df_with_segment_labels'))

# Get specific segment
segment_data = df[df['segment_label'] == 'At-risk High-value']

# Evaluate
metrics = evaluate_model(
    segment_data['Churn'].values,
    lgbm.predict(preprocessor.transform(segment_data.drop(['Churn', 'segment_label']))),
    y_proba=lgbm.predict_proba(...)[:, 1]
)
```

## Testing Strategy

### Unit Tests (in `tests/`)
- Test individual functions with mock data
- Fast, isolated, deterministic
- Examples: `test_preprocess_drops_customerid()`, `test_engineer_creates_streaming_count()`

### Integration Tests (future)
- Test full pipelines end-to-end
- Use real data (subset)
- Verify outputs match expected schema

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_data_preprocess.py -v

# Specific test
pytest tests/test_data_preprocess.py::test_preprocess_drops_customerid -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Debugging Checklist

**Issue**: "Missing columns in X"
- [ ] Check `config.yaml` has correct column names
- [ ] Verify preprocessing output matches feature_metadata.json
- [ ] Ensure categorical columns are strings

**Issue**: "Feature validation failed"
- [ ] Run feature validation check: `python -c "from src.utils import validate_feature_consistency; ..."`
- [ ] Compare actual vs expected columns
- [ ] Check for extra columns in data

**Issue**: "Model pickle load fails"
- [ ] Check file path exists: `ls models/segmentation/kproto.pkl`
- [ ] Verify joblib version consistency (save/load compatibility)
- [ ] Pickle is NOT version-safe; retrain if packages changed

**Issue**: "Threshold tuning produces NaN"
- [ ] Check for NaN in y_proba: `np.isnan(y_proba).any()`
- [ ] Verify y_test has both classes: `np.unique(y_test)`
- [ ] Check precision_recall_curve output

## Code Standards

Every module must:

1. **Have docstring** (module level)
   ```python
   """Short module description."""
   ```

2. **Every function typed & documented**
   ```python
   def process_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
       """Clean and transform data according to config."""
   ```

3. **Use logging** (no print)
   ```python
   logger = logging.getLogger(__name__)
   logger.info("Processing started")
   ```

4. **Support `if __name__ == "__main__"`**
   ```python
   if __name__ == "__main__":
       cfg = load_config()
       # ... example usage ...
   ```

5. **Re-export in `__init__.py`**
   ```python
   from src.module.submodule import function
   __all__ = ['function']
   ```

## File Organization Rules

- **Never hardcode paths**: Use `config['data']['path']` instead of `'data/raw/...'`
- **Never hardcode hyperparams**: Use `config['segmentation']['n_clusters']` instead of `4`
- **All I/O via utils**: Use `save_csv()`, `load_model()` from `src.utils.io_utils`
- **Dependencies flow down**: utils → data → features → segmentation/churn (never upward)

## Useful Commands

```bash
# Config validation
python -c "from src.config import load_config; cfg = load_config(); print(cfg.keys())"

# Check imports
python -c "from src.segmentation import train_segmentation_model; print(train_segmentation_model.__doc__)"

# Quick test
python -m src.data.preprocess
python -m src.features.build_features
python -m src.segmentation.train_segments

# Full pipeline
python run_pipeline.py

# Tests
pytest tests/ -v --tb=short
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-new-feature

# Make changes, test
pytest tests/ -v

# Commit with meaningful message
git add .
git commit -m "feat: add new churn risk feature to engineering module"

# Push and create PR
git push origin feature/add-new-feature
```

## Documentation

- **Code**: Docstrings in `"""..."""` format
- **Config**: Comments in `config/config.yaml`
- **Project**: This file (DEVELOPMENT.md)
- **Usage**: README.md
- **Tests**: Test files show expected behavior

---

**Happy developing!** 🚀

For questions, refer to existing code patterns in the project. When in doubt, check the notebook originals in `notebooks/` for the business logic.
