# Churn-Segmentation MLOps Pipeline

A production-grade machine learning pipeline for predicting customer churn and segmenting customers using K-Prototypes clustering and LightGBM classification.

## 📋 Project Structure

```
churn-mlops/
├── config/
│   └── config.yaml                  # All configuration (paths, hyperparams, thresholds)
├── src/
│   ├── config.py                    # Configuration loader
│   ├── data/
│   │   ├── ingest.py               # Load raw CSV
│   │   └── preprocess.py           # Clean & transform data
│   ├── features/
│   │   ├── engineering.py          # Feature creation (20+ features)
│   │   └── build_features.py       # Orchestrate feature building
│   ├── segmentation/
│   │   ├── train_segments.py       # Train K-Prototypes (k=4)
│   │   └── assign_segments.py      # Assign clusters to customers
│   ├── churn/
│   │   ├── train.py                # Train LightGBM + threshold tuning
│   │   └── evaluate.py             # Model evaluation metrics
│   └── utils/
│       ├── io_utils.py             # CSV, JSON, model I/O
│       ├── logging_config.py        # Logging setup
│       └── feature_validation.py    # Feature consistency checks
├── models/
│   ├── segmentation/               # K-Prototypes artifacts
│   │   ├── kproto.pkl
│   │   ├── scaler.pkl
│   │   ├── catidx.json
│   │   ├── feature_metadata.json
│   │   └── segment_labels.json
│   └── churn/                      # LightGBM artifacts
│       ├── lgbm_churn_model.pkl
│       ├── preprocessor.pkl
│       └── threshold_meta.json
├── data/
│   ├── raw/                        # Original CSV
│   └── processed/                  # Preprocessed & engineered CSVs
├── tests/                          # pytest test suites
├── notebooks/                      # Original exploration (reference)
├── run_pipeline.py                 # Main entry point
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_pipeline.py
```

This will:
1. Load raw customer data → `data/processed/processed_df.csv`
2. Engineer 20+ features → `data/processed/segment_modelling_features.csv`
3. Train K-Prototypes model → `models/segmentation/`
4. Assign customer segments
5. Train LightGBM churn model → `models/churn/`

### 3. Run Individual Modules
```bash
# Data preprocessing
python -m src.data.preprocess

# Feature engineering
python -m src.features.build_features

# Segment training
python -m src.segmentation.train_segments

# Churn model training
python -m src.churn.train
```

## 📊 Configuration

All hardcoded values are in `config/config.yaml`:

- **Data paths**: Input/output file locations
- **Feature engineering**: Tenure bins, service lists, thresholds
- **Segmentation**: K-Prototypes hyperparams (n_clusters=4, init='Cao', n_init=10)
- **Churn modeling**: Train/val/test split (70/15/15), LightGBM hyperparams (650 trees, learning_rate=0.0085)
- **Thresholds**: Default decision threshold (0.5), optimization metric (F1)

Modify `config.yaml` to change any setting without touching code.

## 🔧 Key Features

### Data Preprocessing
- Drop unnecessary columns (customerID)
- Convert TotalCharges to numeric, handle NaN
- Remove inconsistent records (tenure=0)
- Encode categorical variables (SeniorCitizen, Churn)

### Feature Engineering
**Billing/Value:**
- `avg_monthly_spend`: Average monthly spend over tenure
- `charge_gap`: Current vs historical average gap
- `is_high_value`: Flag for high-spenders (above median)

**Lifecycle:**
- `tenure_band`: 4 tenure groups (0-12, 12-36, 36+)

**Services:**
- `streaming_count`: Number of streaming services (0-2)
- `security_count`: Number of security services (0-4)

**Risk Flags:**
- `payment_electronic_check`: Using risky payment method
- `month_to_month_paperless`: Month-to-month contract + paperless billing
- `no_support_services`: No tech support or security services
- `is_isolated`: No partner or dependents
- `fiber_no_security`: Fiber optic internet without security
- `no_internet_services`: No internet service

### Segmentation (K-Prototypes)
Splits customers into 4 personas:
- **Segment 0**: Loyal High-Value (high tenure, high spend, sticky)
- **Segment 1**: Low Engagement (low service adoption)
- **Segment 2**: Stable Mid-Value (moderate spend, steady)
- **Segment 3**: At-Risk High-Value (new, high spend, likely to churn)

### Churn Modeling (LightGBM)
Optimized for F1-score with tuned decision threshold:
- **650 estimators**, max_depth=13, learning_rate=0.0085
- **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical)
- **Threshold tuning**: Finds optimal threshold on test set (default 0.5 → optimized)
- **Evaluation**: ROC-AUC, Precision, Recall, F1, Confusion Matrix

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_data_preprocess.py::test_preprocess_drops_customerid -v
```

Test coverage:
- `test_data_preprocess.py`: Data cleaning & type conversions
- `test_features_engineering.py`: Feature creation
- `test_segmentation_train.py`: Model training & artifact saving
- `test_churn_evaluate.py`: Evaluation metrics

## 📝 Usage Examples

### Load and Use Models

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

# Predict segments on new customers
import pandas as pd
from src.features import engineer_features
from src.segmentation import assign_segments

new_customers = pd.read_csv('new_customers.csv')  # Requires same preprocessing+engineering
new_engineered = engineer_features(new_customers, cfg)
new_with_segments = assign_segments(new_engineered, cfg)

# Predict churn
X_new = new_with_segments.drop(['Churn', 'segment_label'], errors='ignore')
X_new_enc = preprocessor.transform(X_new)
churn_proba = lgbm.predict_proba(X_new_enc)[:, 1]
optimal_threshold = threshold_meta['best_threshold']
churn_pred = (churn_proba >= optimal_threshold).astype(int)
```

## 🔄 Pipeline Flow

```
Raw CSV Data
    ↓
[preprocess_data] → processed_df.csv
    ↓
[engineer_features] → segment_modelling_features.csv
    ↓
┌─→ [train_segmentation_model] → kproto.pkl + artifacts
│       ↓
│   [assign_segments] → df_with_segment_labels.csv
│
└─→ [train_churn_model] → lgbm_churn_model.pkl + preprocessing.pkl
        ↓
    [evaluate_model] → metrics (ROC-AUC, F1, etc.)
```

## 📦 Dependencies

- **Data**: pandas
- **ML**: scikit-learn, lightgbm, kmodes
- **Configuration**: PyYAML
- **Testing**: pytest
- **Utilities**: joblib

See `requirements.txt` for specific versions.

## 🎯 Coding Standards

All code follows these rules:

✅ **Executable & Importable**: Every module has `if __name__ == "__main__"` guard  
✅ **Type Hints**: All functions annotated with input/output types  
✅ **Logging**: Uses `logging.getLogger(__name__)` (no print statements)  
✅ **Configuration**: All parameters in `config.yaml` or function arguments  
✅ **I/O Centralized**: All file operations via `src/utils/io_utils.py`  
✅ **No Circular Imports**: Dependency flow: utils → data → features → segmentation → churn  
✅ **Module Exports**: Public functions re-exported in `__init__.py`  
✅ **Validated Inputs**: Feature consistency checks in segmentation & churn modules  

## 📚 Reference

**Original Notebooks** (kept for reference, not imported):
- `notebooks/EDA.ipynb` → Exploratory data analysis + visualization
- `notebooks/segmentation.ipynb` → Segmentation model development
- `notebooks/best_churn_model.ipynb` → Churn model hyperparameter tuning

**Key Results from Notebooks:**
- Customer churn rate: 26.6%
- Fiber optic users: high churn (strong signal)
- Month-to-month + paperless billing: high risk combo
- No tech support: major churn driver
- New customers (low tenure): vulnerable

## 🚦 Next Steps

- [ ] Add DVC for data versioning
- [ ] Implement MLflow experiment tracking
- [ ] Create FastAPI inference service
- [ ] Add per-segment churn models (optional)
- [ ] Implement model retraining pipeline
- [ ] Add integration tests
- [ ] Create monitoring dashboards

## 📧 Questions?

Refer to `config/config.yaml` for all configurable options.
Check module docstrings and test files for usage examples.

---

**Version**: 1.0  
**Last Updated**: April 2026
