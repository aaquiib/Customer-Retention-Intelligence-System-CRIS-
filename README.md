# CRIS: Customer Retention Intelligence System

A production-ready machine learning system for **predicting customer churn**, **segmenting customers**, and **recommending retention actions**. CRIS combines advanced segmentation (K-Prototypes clustering), interpretable churn prediction (LightGBM), and explainability (SHAP) in a unified platform with dual interfaces: interactive Streamlit dashboard and production FastAPI service.

**Status:** ✅ Production Ready | **Version:** 1.0 (April 2026) | **License:** [Your License Here]

---

## Business Problem & Solution

### The Challenge
The telecom industry faces a critical challenge: **customer churn directly erodes revenue**. Companies lose 26-30% of annual revenue to customer attrition. Traditional retention strategies cast a wide net, wasting resources on low-value customers while missing at-risk high-value clients.

**Key Pain Points:**
- Reactive churn (no early warning) vs. Proactive retention
- Non-segmented campaigns (same message to all customers)
- Unclear ROI on retention spend (intervene on wrong customers)
- Lack of explainability (why will this customer churn?)

### The Solution: CRIS
CRIS predicts which customers will churn **3-6 months in advance**, segments them into **4 strategic tiers** (loyalty × spending), and recommends **targeted retention actions** with clear financial impact.

**Business Outcomes:**
- **78.6% recall** — Catches 78.6% of customers likely to churn
- **84% AUC** — Excellent ranking of risk vs. safety
- **Segment-based actions** — VIP customers get premium retention (concierge support), Low-engagement get upsell offers
- **Explainable decisions** — Know exactly why each prediction was made (SHAP)

### Customer Segments
1. **Loyal High-Value** (0) — Protect: High tenure, sticky, high spend → Premium support
2. **Low Engagement** (1) — Upsell: Low service adoption, mid/low spend → Service education
3. **Stable Mid-Value** (2) — Maintain: Moderate tenure, stable spend → Standard support
4. **At-Risk High-Value** (3) — Urgent Action: New or unstable, high spend, churn-prone → Proactive outreach

---

## Model Performance & Hyperparameter Tuning

### Metrics Summary

| Metric | Value | Business Interpretation |
|--------|-------|------------------------|
| **ROC-AUC (Test)** | **0.8398** (84%) | Excellent discrimination between churners & stayers |
| **Recall (Test)** | **0.7857** (78.6%) | Catches 78.6% of actual churners before they leave |
| **Decision Threshold** | 0.4356 | Optimized for F1-score (vs default 0.5) |
| **Model** | LightGBM (650 trees) | 33 engineered features, 70/15/15 train/val/test split |

### Hyperparameter Tuning Journey: 58% → 80% Recall

The path to 78.6% recall involved a **two-phase hyperparameter optimization strategy**:

#### **Phase 1: RandomSearch — Narrow the Search Space**
- **Baseline Recall:** 58% (Logistic + default LightGBM params)
- **Approach:** GridSearch over 100 random hyperparameter combinations
- **Focus:** Identify promising regions for `max_depth`, `learning_rate`, `num_leaves`, `scale_pos_weight`
- **Result:** Recall improved to ~72%, identified optimal ranges
- **Key Finding:** Higher `learning_rate` (0.008-0.012) and `scale_pos_weight` (1.5-1.8) improved minority class detection

#### **Phase 2: Optuna — Fine-Tune for Production**
- **Starting Point:** RandomSearch best params + identified ranges
- **Approach:** Bayesian optimization (Optuna) with 500+ trials, F1-score objective
- **Tuned Parameters:** `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- **Result:** Recall boosted to **~80%**, generalization maintained across train/val/test
- **Final Params:** See [config/config.yaml](config/config.yaml#L156-L175)

#### **Threshold Optimization**
- After training, further optimized decision threshold from default 0.5 → 0.4356 (F1-optimized)
- Lower threshold increases recall (catch more churners) with acceptable precision tradeoff

**Summary:** Two-phase approach delivered **22% relative recall improvement** (58% → 78.6%) while maintaining 84% AUC and model stability.

---

## Key Features

- 🎯 **78.6% Recall Churn Detection** — Proactively identify customers at risk before they leave
- 🔍 **4-Segment Customer Intelligence** — Identify customer personas (Loyal High-Value, Low Engagement, Stable Mid-Value, At-Risk High-Value)
- 💰 **Revenue Impact Quantification** — Segment-based financial prioritization for retention spend
- 🎮 **What-If Simulation** — Test retention strategies by simulating feature changes (e.g., contract upgrades)
- 📈 **SHAP-Based Explainability** — Global feature importance + instance-level explanations (why each prediction?)
- 📊 **Batch Scoring** — Upload CSV files to score up to 50,000 customers simultaneously
- 🔧 **Production-Ready API** — FastAPI service with CORS, validation, async support
- ⚙️ **Config-Driven Design** — Customize thresholds, business rules, features via YAML (no code changes)
- 🏗️ **DVC MLOps Pipeline** — Reproducible ML pipeline with version control for data, features, models

---

## System Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CRIS CHURN PREDICTION WORKFLOW                        │
└──────────────────────────────────────────────────────────────────────────┘

  Customer Data (Raw)
        │
        ↓
  ┌─────────────────────────────────────────┐
  │  DATA PREPROCESSING                      │
  │ • Clean missing values, drop customerID  │
  │ • Convert data types                     │
  └─────────────────────────────────────────┘
        │
        ↓
  ┌─────────────────────────────────────────┐
  │  FEATURE ENGINEERING                     │
  │ • Create 33 business features            │
  │ • Billing, tenure, services, risk flags  │
  └─────────────────────────────────────────┘
        │
        ├─────────────────────────┬──────────────────────────┐
        │                         │                          │
        ↓                         ↓                          ↓
   ┌─────────────┐          ┌─────────────┐         ┌──────────────┐
   │ K-Prototypes│          │   LightGBM  │         │ SHAP Explainer
   │ Clustering  │          │  (650 trees)│         │  (Background
   │  (k=4)      │          │             │         │   Samples)
   │             │          │ Churn Prob: │         │
   │ SEGMENT     │          │ 0.0 - 1.0   │         │ Feature
   │ Assignment  │          │             │         │ Importance
   └─────────────┘          └─────────────┘         └──────────────┘
        │                         │                          │
        └─────────────┬───────────┴──────────────┬───────────┘
                      │                          │
                      ↓                          ↓
            ┌──────────────────────────────────────────────┐
            │  BUSINESS RULES ENGINE                       │
            │ • Match segment + churn_prob + value        │
            │ • Recommend retention action                │
            │ • Calculate priority score                  │
            └──────────────────────────────────────────────┘
                      │
                      ↓
            ┌──────────────────────────────────────────────┐
            │  OUTPUT: PREDICTION + EXPLANATION            │
            │ • Segment (0-3, label, confidence)          │
            │ • Churn probability                         │
            │ • Recommended action (VIP/Upsell/Support)   │
            │ • Top 5 contributing features (SHAP)        │
            └──────────────────────────────────────────────┘
```

**Supported Inputs:**
- Single customer (API, Dashboard form)
- Batch CSV (Dashboard, up to 50K rows)
- Manual what-if scenario testing (Dashboard)

---

## Quick Start

### Prerequisites

- **Python:** 3.9 or later
- **System:** Windows, macOS, or Linux (WSL supported)
- **Disk Space:** ~500 MB (models + data)

### Installation

```bash
# Clone or navigate to project directory
cd "path/to/churn-segmentation-decision_system"

# Create virtual environment (optional but recommended)
python -m venv churn_env
churn_env\Scripts\activate  # Windows
# source churn_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md).

---

## Running the System

### **Option 1: Interactive Dashboard** (Recommended for analysts)

```bash
cd dashboard1
streamlit run app.py
```

**Opens:** [http://localhost:8501](http://localhost:8501) in your browser  
**Features:** 9-page interactive app with customer analysis, batch scoring, what-if simulation, and model explainability  
📖 Full guide: [DASHBOARD_FEATURES_REPORT.md](DASHBOARD_FEATURES_REPORT.md)

### **Option 2: Production API** (Recommended for integration)

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /api/predict` — Single/batch predictions
- `GET /api/feature-importance/global` — Global feature importance
- `POST /api/what-if` — Feature perturbation simulation

**Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)  
📖 Full guide: [API_REPORT.md](API_REPORT.md)

### **Option 3: ML Pipeline** (For data scientists)

```bash
python run_pipeline.py
```

Trains segmentation and churn models end-to-end (~1-2 minutes)  
📖 Full guide: [QUICKSTART.md](QUICKSTART.md) | DVC setup: [DVC_IMPLEMENTATION_SUMMARY.md](DVC_IMPLEMENTATION_SUMMARY.md)

---

## Architecture

### High-Level System Design

```
Data Pipeline (DVC-Managed, 5 Stages)
├── Stage 1: Preprocess — Raw CSV → Cleaned data (drop nulls, format conversion)
├── Stage 2: Engineer — 33 business features (billing, tenure, services, risk)
├── Stage 3: Segmentation Train — K-Prototypes clustering (k=4, mixed features)
├── Stage 4: Segment Assign — Assign every customer to a segment
└── Stage 5: Churn Train — LightGBM with threshold optimization

Inference Engine (Runtime)
├── K-Prototypes Model — Segment assignment (outputs 0-3 + confidence)
├── LightGBM Model — Churn probability (0.0-1.0) + threshold decision
├── SHAP Explainer — Per-instance feature importance + global rankings
└── Business Rules Engine — Action recommendations (VIP, Upsell, Support)

Interfaces
├── Streamlit Dashboard (dashboard1/) — 9 pages, interactive, real-time visualization
└── FastAPI Server (api/) — RESTful endpoints, async, production-grade with CORS
```

### Models in Detail

#### **Segmentation: K-Prototypes Clustering**
- **Algorithm:** K-Prototypes (handles mixed numeric + categorical features)
- **Clusters:** 4 segments (Loyal High-Value, Low Engagement, Stable Mid-Value, At-Risk High-Value)
- **Features:** 25 total (7 numeric, 18 categorical)
  - Numeric: tenure, MonthlyCharges, TotalCharges, avg_monthly_spend, charge_gap, streaming_count, security_count
  - Categorical: gender, contract type, internet service, service adoption flags
- **Training:** K-Means++ initialization, 10 restarts, converges in ~30 seconds
- **Output:** Segment ID (0-3) + confidence score (distance-based)

#### **Churn Prediction: LightGBM Classifier**
- **Algorithm:** LightGBM (gradient boosting, fast, handles imbalanced data)
- **Architecture:** 650 trees, max_depth=13, optimized via Optuna
- **Features:** 33 total (16 numeric + engineered, 17 categorical)
  - **Raw:** tenure, MonthlyCharges, TotalCharges, gender, contract, internet service, payment method
  - **Engineered:** avg_monthly_spend, tenure_band, streaming_count, security_count, payment_electronic_check, month_to_month_paperless, risk flags
  - **Contextual:** segment (from K-Prototypes output)
- **Target Class Imbalance:** Addressed via `scale_pos_weight=1.71` (upweight minority class)
- **Training Split:** 70% train (4,922 samples) | 15% val (1,055) | 15% test (1,055)
- **Output:** Churn probability (0.0-1.0) → Apply threshold 0.4356 → Binary decision (churner or stayer)

#### **Explainability: SHAP TreeExplainer**
- **Method:** TreeExplainer on LightGBM model (fast, exact Shapley values for trees)
- **Global:** Pre-computed on 200 background samples from training data (cached at startup)
- **Local:** Computed per-instance on API request (SHAP values + feature contributions)
- **Output:** Top 5-10 features ranked by |SHAP value| (most impactful for each customer)

---

## Project Structure

```
api/                      ← FastAPI application (production service)
├── app.py               ← Entry point + model cache (singleton pattern)
├── schemas.py           ← Pydantic request/response validation
└── endpoints/           ← /predict, /feature-importance, /what-if routes

dashboard1/              ← Primary Streamlit dashboard (interactive UI)
├── app.py               ← 9-page application with sidebar navigation
├── config.py            ← Dashboard configuration & styling
├── pages/               ← Individual page implementations
└── utils/               ← API client, validators, chart builders

inference/               ← ML inference & explainability
├── pipeline.py          ← Single/batch prediction, model loading
├── business_rules.py    ← Retention action recommendation engine
├── shap_explainer.py    ← SHAP global/local explanation computation
└── shap_utils.py        ← Feature importance extraction & ranking

src/                     ← Model training source code
├── data/preprocess.py   ← Data cleaning & formatting
├── features/            ← Feature engineering (33 features)
├── churn/               ← Churn model training & evaluation
│   ├── train.py         ← LightGBM training + threshold optimization
│   └── evaluate.py      ← Metrics computation (AUC, recall, F1, etc.)
├── segmentation/        ← Segmentation training & assignment
│   ├── train_segments.py ← K-Prototypes training (k=4)
│   └── assign_segments.py ← Segment assignment for all customers
└── utils/               ← Logging, model I/O, config loading

config/                  ← Configuration files
├── config.yaml          ← 📊 **Feature definitions, hyperparameters (see Phase 2 tuning results)**
└── business_rules.json  ← Retention actions, thresholds, segment priorities

models/                  ← Trained model artifacts
├── segmentation/
│   ├── kproto.pkl       ← K-Prototypes model (k=4)
│   ├── scaler.pkl       ← StandardScaler for numeric features
│   ├── catidx.json      ← Categorical feature indices
│   └── segment_labels.json ← Segment name mappings
└── churn/
    ├── lgbm_churn_model.pkl ← **LightGBM (650 trees, AUC 0.8398, Recall 0.7857)**
    ├── preprocessor.pkl  ← StandardScaler + OneHotEncoder pipeline
    ├── threshold_meta.json ← **Optimal threshold (0.4356) + metrics**
    └── metrics_latest.json ← Latest train/val/test split metrics

data/                    ← Datasets (raw & processed)
├── raw/                 ← Original Telco Customer Churn CSV (7,043 customers)
└── processed/
    ├── processed_df.csv ← Cleaned data
    ├── segment_modelling_features.csv ← Features for segmentation
    ├── churn_features.csv ← Features for churn model
    └── df_with_segment_labels.csv ← Data + segment assignments + predictions

tests/                   ← Unit tests (14 test cases)
└── test_*.py            ← Tests for preprocessing, features, segmentation, churn

notebooks/               ← Jupyter notebooks & experiment tracking
├── experiment_log.csv   ← 📊 **Experiment history: baseline recall 58% → final 78.6%**
├── EDA.ipynb            ← Exploratory data analysis
└── experiment*.ipynb    ← Hyperparameter tuning experiments
```

**Key Artifacts:**
- `models/churn/metrics_latest.json` — Latest model evaluation (train/val/test metrics)
- `config/config.yaml` — Hyperparameters tuned via RandomSearch + Optuna (see "Hyperparameter Tuning Journey")
- `notebooks/experiment_log.csv` — Complete experiment history tracking recall improvements
- `models/churn/threshold_meta.json` — Optimal threshold (0.4356) for F1-optimized predictions

---

## Documentation & Resources

| Resource | Purpose | Key Info |
|----------|---------|----------|
| [QUICKSTART.md](QUICKSTART.md) | One-liner setup, pipeline execution, output files | Run full pipeline in 1-2 minutes |
| [API_REPORT.md](API_REPORT.md) | Complete API specification, examples, integration | Endpoint docs: `/predict`, `/feature-importance/global`, `/what-if` |
| [DASHBOARD_FEATURES_REPORT.md](DASHBOARD_FEATURES_REPORT.md) | Dashboard walkthrough, all 9 pages, workflows | Single customer, batch scoring, what-if simulation |
| [DVC_IMPLEMENTATION_SUMMARY.md](DVC_IMPLEMENTATION_SUMMARY.md) | MLOps pipeline, DVC setup, reproducibility | 5-stage pipeline with metrics tracking |
| [dvc_workflow.md](dvc_workflow.md) | DVC commands, metrics tracking, git integration | `dvc repro`, `dvc metrics show`, `dvc metrics diff` |
| [config/config.yaml](config/config.yaml) | **Feature definitions, Optuna-tuned hyperparameters** | **See lines 156-175 for final LightGBM params** |
| [config/business_rules.json](config/business_rules.json) | Retention actions, priority scoring, thresholds | Segment-based rules engine configuration |
| [notebooks/experiment_log.csv](notebooks/experiment_log.csv) | **Experiment history: baseline 58% → final 78.6% recall** | **See Phase 1 & Phase 2 results** |
| [models/churn/metrics_latest.json](models/churn/metrics_latest.json) | **Latest metrics: AUC 0.8398, Recall 0.7857 on test set** | Train/Val/Test split metrics, threshold optimization |

---

## Quick Facts

- **Recall Improvement:** 58% baseline (Logistic) → 78.6% production (LightGBM with Optuna tuning)
- **Hyperparameter Tuning Timeline:** RandomSearch (100 trials) → Optuna (500+ trials) → Threshold optimization
- **Key Hyperparameters Tuned:** `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `scale_pos_weight` (see [config.yaml](config/config.yaml#L156-L175))
- **Training Data:** 7,032 customers (Telco churn dataset), 70/15/15 train/val/test split
- **Feature Count:** 33 total (16 numeric + engineered, 17 categorical)
- **Model Inference:** <100ms per customer (single) or <5s per 1000 (batch)

---

## Testing

Run all tests to validate preprocessing, features, segmentation, and churn model:

```bash
pytest tests/ -v
```

**Coverage:** 14 test cases across data pipeline, feature engineering, and model training

---

## Support & Troubleshooting

- **Setup Issues?** → See [QUICKSTART.md](QUICKSTART.md#troubleshooting)
- **API Errors?** → Check [API_REPORT.md](API_REPORT.md#error-handling)
- **Dashboard Questions?** → See [DASHBOARD_FEATURES_REPORT.md](DASHBOARD_FEATURES_REPORT.md)
- **Model Performance?** → Review config/config.yaml for hyperparameters and thresholds

---

**Built with:** Python 3.9+ | FastAPI | Streamlit | LightGBM | scikit-learn | SHAP | DVC