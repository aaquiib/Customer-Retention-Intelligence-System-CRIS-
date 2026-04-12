# Quick Reference: Running the Pipeline

## 🚀 One-Liner to Run Everything

```bash
cd "e:\ML PROJECTS\churn-segmentation-decision_system" && python run_pipeline.py
```

---

## 📋 What This Does (in order)

1. **Data Preprocessing** (src/data/preprocess.py)
   - Loads raw CSV from `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
   - Drops customerID, handles missing values
   - Converts data types (SeniorCitizen 0/1 → yes/no, Churn Yes/No → 1/0)
   - Saves to `data/processed/processed_df.csv`

2. **Feature Engineering** (src/features/engineering.py)
   - Creates 20+ business features:
     - Billing: avg_monthly_spend, charge_gap, is_high_value
     - Tenure: tenure_band (4 groups)
     - Services: streaming_count, security_count (0-4)
     - Risk: 6 vulnerability flags
   - Saves to `data/processed/segment_modelling_features.csv`

3. **Segmentation Training** (src/segmentation/train_segments.py)
   - Trains K-Prototypes (k=4) on mixed numeric/categorical features
   - Creates 4 customer personas:
     - 0: Loyal High-Value (high tenure, sticky)
     - 1: Low Engagement (low service adoption)
     - 2: Stable Mid-Value (moderate spend)
     - 3: At-risk High-value (new, high spend, likely churn)
   - Saves models to `models/segmentation/`

4. **Segment Assignment** (src/segmentation/assign_segments.py)
   - Assigns every customer to a segment
   - Validates feature consistency
   - Saves to `data/processed/df_with_segment_labels.csv`

5. **Churn Model Training** (src/churn/train.py)
   - Trains LightGBM (650 estimators, learning_rate=0.0085)
   - Splits: 70% train | 15% val | 15% test
   - Preprocessing: StandardScaler + OneHotEncoder
   - Optimizes decision threshold via F1-score
   - Saves models to `models/churn/`

**Total Runtime**: ~1-2 minutes (depending on hardware)

---

## 📂 Output Files Created

```
data/processed/
├── processed_df.csv                      # Cleaned data
├── segment_modelling_features.csv        # Engineered features
├── churn_features.csv                    # Same as above (copy)
└── df_with_segment_labels.csv            # Data + segment assignments

models/segmentation/
├── kproto.pkl                            # Trained K-Prototypes model
├── scaler.pkl                            # StandardScaler for numerics
├── catidx.json                           # Categorical column indices
├── feature_metadata.json                 # Feature names + order
└── segment_labels.json                   # Segment name mappings

models/churn/
├── lgbm_churn_model.pkl                  # Trained LightGBM model
├── preprocessor.pkl                      # StandardScaler + OneHotEncoder
└── threshold_meta.json                   # Optimal threshold + metrics
```

---

## 🧪 Test the Implementation

```bash
# Run all tests (14 test cases)
pytest tests/ -v

# Run specific test file
pytest tests/test_data_preprocess.py -v

# Test individual function
pytest tests/test_data_preprocess.py::test_preprocess_drops_customerid -v
```

---

## ✅ Verify Structure

```bash
# Check all files exist + imports work
python verify_implementation.py

# Test configuration loads
python -c "from src.config import load_config; cfg = load_config(); print('✓ Config OK')"

# Test individual modules
python -m src.data.preprocess
python -m src.features.build_features
python -m src.segmentation.train_segments
python -m src.churn.train
```

---

## 🔧 Change Configuration

Edit `config/config.yaml` (no code changes needed):

```yaml
# Change number of segments
segmentation:
  n_clusters: 5  # was 4

# Change LightGBM learning rate
churn_modeling:
  lgbm_hyperparams:
    learning_rate: 0.01  # was 0.008535...

# Change data split
churn_modeling:
  train_size: 0.75  # was 0.70
  test_size: 0.25   # was 0.15
```

Then re-run pipeline:
```bash
python run_pipeline.py
```

---

## 📊 Use Trained Models

```python
from src.config import load_config
from src.utils import load_model, load_csv, load_json

cfg = load_config()

# Load segmentation model
kproto = load_model(f"{cfg['models']['segmentation_dir']}kproto.pkl")
scaler = load_model(f"{cfg['models']['segmentation_dir']}scaler.pkl")
catidx = load_json(f"{cfg['models']['segmentation_dir']}catidx.json")

# Load churn model
lgbm = load_model(f"{cfg['models']['churn_dir']}lgbm_churn_model.pkl")
preprocessor = load_model(f"{cfg['models']['churn_dir']}preprocessor.pkl")
threshold_meta = load_json(f"{cfg['models']['churn_dir']}threshold_meta.json")

# Make predictions on new data
# (assuming new customers are preprocessed + engineered)
X_new_enc = preprocessor.transform(X_new)
churn_proba = lgbm.predict_proba(X_new_enc)[:, 1]
optimal_threshold = threshold_meta['best_threshold']
churn_pred = (churn_proba >= optimal_threshold).astype(int)
```

---

## 📖 Documentation

- **README.md** — Complete user guide (start here)
- **DEVELOPMENT.md** — Developer guide, architecture, extending code
- **IMPLEMENTATION_SUMMARY.md** — What was built, verification checklist
- **config/config.yaml** — All configurable values

---

## 🆘 Troubleshooting

**Error: "ModuleNotFoundError: No module named 'yaml'"**
```bash
pip install pyyaml lightgbm kmodes scikit-learn pandas numpy joblib
```

**Error: "File not found: data/raw/..."**
- Ensure raw CSV is in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Check path in `config/config.yaml`

**Error: "Feature validation failed"**
- Check that engineered features match expected columns
- Run `python verify_implementation.py` to diagnose

**Models won't load**
- Ensure models saved in correct directory (check config.yaml)
- Pickle files are version-specific; may need retraining if packages updated

---

## 📊 Expected Results

After running `python run_pipeline.py`:

- **Segmentation**: 4 clusters with ~4,100-1,700 customers each
- **Churn Model**: ~77-82% ROC-AUC on test set
- **Optimal Threshold**: Usually 0.35-0.45 (better than default 0.5)
- **Artifacts**: 8 model files + metadata (~5-20 MB total)

---

## 🎯 Common Tasks

### Retrain Just Churn Model
```bash
python -c "
from src.config import load_config
from src.utils import load_csv, setup_logging
from src.churn.train import train_churn_model
cfg = load_config()
setup_logging(cfg['logging'])
df = load_csv(cfg['data']['processed_csv_path'].replace('processed', 'df_with_segment_labels'))
train_churn_model(df, cfg)
"
```

### Evaluate on Specific Segment
```python
from src.churn import evaluate_model
segment_data = df[df['segment_label'] == 'At-risk High-value']
metrics = evaluate_model(
    segment_data['Churn'].values,
    lgbm.predict(X_segment_enc)
)
```

### Compare Thresholds
```python
from src.churn import compare_thresholds
comparison = compare_thresholds(y_test, y_proba, [0.3, 0.4, 0.5, 0.6, 0.7])
for threshold, results in comparison.items():
    print(f"Threshold {threshold}: F1={results['metrics']['f1']:.4f}")
```

---

## ⏱️ Timeline

- **Phase 1** (Foundation): config, logging, utils ~ 10 min
- **Phase 2** (Data): preprocess ~ 5 min  
- **Phase 3** (Features): engineering, 20+ features ~ 10 min
- **Phase 4** (Segmentation): K-Prototypes training ~ 5 min
- **Phase 5** (Churn): LightGBM + threshold tuning ~ 10 min
- **Phase 6-7** (Tests + Docs): test suites, README, guides ~ 20 min

**Total Implementation Time**: ~1 hour

---

**Ready to go!** 🚀

Next step: `python run_pipeline.py`
