# Model Card: Churn Segmentation Decision System

## Overview

This document describes the LightGBM and CatBoost models trained for customer churn prediction in the Churn Segmentation Decision System. These models were developed through a three-phase hyperparameter optimization pipeline and are production-ready for deployment.

---

## Model Details

### Model 1: LightGBM (Primary Model)

| Attribute | Value |
|-----------|-------|
| **Framework** | LightGBM |
| **Task** | Binary Classification (Churn Prediction) |
| **Test AUC** | 0.8406 |
| **Test Precision** | 0.5600 |
| **Test Recall** | 0.7000 |
| **Test F1-Score** | 0.6222 |
| **Format** | LightGBM native (model.txt) |
| **Optimal Threshold** | 0.3735 (tuned for F1 maximization) |

#### Hyperparameters (Phase 2 Optimized)

```yaml
n_estimators: 650
max_depth: 13
learning_rate: 0.008535633844517027
num_leaves: 13
min_child_samples: 21
subsample: 0.7580026562631494
colsample_bytree: 0.6066081804708111
reg_alpha: 0.15703551247124883
reg_lambda: 2.884795719882881
scale_pos_weight: 1.710339753731322
min_split_gain: 0.43601254903789377
subsample_freq: 4
```

**Training Details:**
- Trained on combined train+validation split (85% of data)
- Evaluated on held-out test set (15% of data)
- Imbalance handling: `scale_pos_weight=1.71` to address 73.5% negative class dominance
- Early stopping: Not used (all 650 trees trained)

### Model 2: CatBoost (Secondary Model)

| Attribute | Value |
|-----------|-------|
| **Framework** | CatBoost |
| **Task** | Binary Classification (Churn Prediction) |
| **Test AUC** | 0.8367 |
| **Test Precision** | 0.4601 |
| **Test Recall** | 0.8643 |
| **Test F1-Score** | 0.6005 |
| **Format** | CatBoost native (model.cbm) |

#### Hyperparameters (Phase 2 Optimized)

```yaml
iterations: 400
depth: 5
learning_rate: 0.14343748047344063
l2_leaf_reg: 6.719179654273406
bagging_temperature: 1.1036568304070327
border_count: 64
random_strength: 2.6011875463135556
scale_pos_weight: 4.276934155166999
min_data_in_leaf: 22
leaf_estimation_iterations: 2
```

**Training Details:**
- Trained on combined train+validation split (85% of data)
- Evaluated on held-out test set (15% of data)
- Native categorical feature handling (14 categorical features)
- Early stopping: Enabled (rounds=40)

---

## Model Performance

### Test Set Metrics Comparison

```
       Model  Test_AUC  Precision  Recall     F1
   LightGBM    0.8406     0.5600  0.7000 0.6222
    CatBoost    0.8367     0.4601  0.8643 0.6005
     XGBoost    0.8287     0.5282  0.7036 0.6034
```

### Threshold Optimization (LightGBM)

| Metric | Default (0.5) | Optimized (0.3735) | Change |
|--------|---------------|--------------------|--------|
| Precision | 0.50 | 0.56 | +0.06 |
| Recall | 0.82 | 0.70 | -0.12 |
| F1-Score | 0.62 | 0.62 | 0.00 |
| True Positives | 51 | 43 | -8 |
| False Positives | 51 | 34 | -17 |

**Interpretation:** The optimized threshold reduces false positives by 33% while maintaining F1-score, making the model more precision-oriented for business deployment.

---

## Input Features

### Training Data
- **Source:** `data/processed/df_with_segment_labels.csv`
- **Total Samples:** 7043
- **Target Variable:** `Churn` (binary: 0=No Churn, 1=Churn)
- **Class Distribution:** 73.5% Class 0, 26.5% Class 1

### Feature Groups

#### Numerical Features (8)
- `tenure` - Customer tenure in months
- `MonthlyCharges` - Monthly subscription charges
- `TotalCharges` - Total charges accumulated
- `streaming_count` - Number of streaming services (0-2)
- `security_count` - Number of security services (0-4)
- `month_to_month_paperless` - Boolean indicator
- `no_support_services` - Boolean indicator
- `is_isolated` - Boolean indicator

#### Categorical Features (14)
- `gender` - Customer gender
- `PhoneService` - Phone service subscription
- `MultipleLines` - Multiple phone lines
- `InternetService` - Type of internet service
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` - Service subscriptions
- `StreamingTV`, `StreamingMovies` - Streaming services
- `Contract` - Contract type
- `PaperlessBilling` - Paperless billing status
- `PaymentMethod` - Payment method
- `segment` - Customer segment (from segmentation model)
- `tenure_band` - Engineered: Tenure binned into categories

---

## Training Methodology

### Optimization Pipeline

#### Phase 1: Broad Exploration (RandomizedSearchCV + Manual Grid Search)
- **XGBoost:** 60 random trials, CV AUC = 0.8519
- **LightGBM:** 60 random trials, CV AUC = 0.8504
- **CatBoost:** 10 manual iterations, CV AUC = 0.8522

#### Phase 2: Focused Refinement (Optuna with TPE Sampler)
- **XGBoost:** 80 trials, Val AUC = 0.8559
- **LightGBM:** 80 trials, Val AUC = 0.8563
- **CatBoost:** 60 trials, Val AUC = 0.8580

#### Phase 3: Final Evaluation
- Retrained best models on combined train+validation split
- Evaluated on held-out test set (15% of original data)
- Applied threshold tuning for LightGBM
- **LightGBM Test AUC: 0.8406** (selected as primary model)
- **CatBoost Test AUC: 0.8367** (selected as secondary model)

---

## Use Cases

### Primary Use Case
**Real-time Churn Prediction for Intervention Campaigns**
- Input: Customer profile (demographics, service usage, contract details)
- Output: Churn probability + binary prediction
- Recommended Model: LightGBM with optimized threshold (0.3735)
- Business Metric: Precision-focused to prioritize high-confidence churn predictions

### Secondary Use Case
**Segment-based Churn Risk Assessment**
- Input: Customer cohort data
- Output: Churn rate predictions + risk stratification
- Model: Either LightGBM or CatBoost (similar performance)
- Business Metric: Recall-focused (CatBoost) to identify at-risk customers

### Model Selection Guidelines

| Scenario | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **Maximize Sensitivity** | CatBoost | Recall=0.8643 (catches 86% of churners) |
| **Maximize Precision** | LightGBM (threshold=0.3735) | Precision=0.56 (56% of predicted churners actually churn) |
| **Balanced F1-Score** | LightGBM | F1=0.6222 (best overall balance) |
| **Inference Speed** | LightGBM | Typically 20-30% faster than CatBoost |
| **Interpretability** | CatBoost | Native categorical feature handling, better SHAP compatibility |

---

## Limitations

### Model Scope
1. **Binary Classification Only:** Predicts churn vs. no-churn; does not stratify by churn reason
2. **Cross-sectional:** Trained on snapshot data; temporal patterns not captured
3. **Imbalance:** Class imbalance (73.5% vs 26.5%) handled via `scale_pos_weight`, not resampling

### Data Limitations
1. **Segment Leakage:** Excludes raw customer context; relies on pre-computed segment labels
2. **Feature Engineering:** Engineered features may overfit to training distribution
3. **Temporal Validity:** Model trained on 2024 data; may degrade if customer behavior shifts

### Performance Boundaries
1. **Low Precision (~56%):** False positive rate of 44% may require business-level filtering
2. **Threshold Sensitivity:** Small threshold changes significantly impact precision/recall tradeoff
3. **Segment Dependency:** Performance may degrade for underrepresented segments

### Known Issues
- **Feature Scaling:** LightGBM/CatBoost are tree-based and scale-invariant; sklearn preprocessing may be redundant
- **Categorical Imbalance:** Some categorical values (e.g., rare contract types) may have poor predictions

---

## Ethical Considerations

### Fairness
- Model evaluated on overall test set; fairness across demographic subgroups not explicitly tested
- Recommend: Audit predictions by gender, age, and region before deployment

### Bias
- Training data reflects historical churn patterns; may encode existing service biases
- Recommend: Monitor predictions for disparate impact over time

### Transparency
- Model provides probability scores and feature importance (SHAP available)
- Recommend: Use threshold=0.3735 for LightGBM to maximize interpretability

---

## Deployment Instructions

### Loading the Model

```python
from pathlib import Path
from artifacts import load_artifacts

# Load artifacts from the version directory
artifact_dir = Path("models/v1.0.0-20260411_224248")
loader, predictor = load_artifacts(artifact_dir)

# Make predictions
X_test = pd.read_csv("test_data.csv")
predictions = predictor.predict(X_test, model_name='LightGBM')
probabilities = predictor.predict_proba(X_test, model_name='LightGBM')
```

### Production Checklist
- [ ] Verify model file integrity (model.txt / model.cbm)
- [ ] Test preprocessing pipeline with sample data
- [ ] Validate threshold calibration for business use case
- [ ] Set up monitoring for prediction distribution drift
- [ ] Document threshold choice and rationale for stakeholders
- [ ] Implement logging for all predictions (for performance tracking)
- [ ] Set up alerts for degraded AUC or calibration shift

---

## Maintenance & Monitoring

### Recommended Monitoring Metrics
1. **Prediction Distribution:** Monitor shift in predicted churn rates
2. **Calibration:** Track observed churn rate vs. predicted probability
3. **Feature Drift:** Alert if feature distributions deviate from training set
4. **Model Staleness:** Retrain models quarterly or if AUC drops below 0.80

### Retraining Triggers
- [ ] New labeled churn data available (≥500 samples)
- [ ] AUC drops below 0.80 on recent data
- [ ] Prediction calibration degrades (observed vs. predicted churn >5%)
- [ ] Feature distributions shift significantly (KS statistic >0.15)

### Version Control
- Current Version: **v1.0.0-20260411_224248**
- Training Date: **2026-04-11**
- Last Updated: See `metadata.json` timestamp

---

## References

- **Optimization Framework:** Optuna (TPE Sampler)
- **Cross-Validation:** Stratified K-Fold (5 splits)
- **Feature Engineering:** Documented in `notebooks/tree_based_modelling.ipynb`
- **Threshold Tuning:** Precision-Recall Curve Optimization (F1-maximized)

---

## Questions & Support

For questions about model performance, hyperparameters, or deployment:
- Refer to `metadata.json` for optimization history
- See `preprocessing/data_splits.json` for train/val/test splits
- Check `lightgbm/config.yaml` and `catboost/config.yaml` for full configurations

**Model prepared by:** ML Pipeline
**Date:** 2026-04-11
