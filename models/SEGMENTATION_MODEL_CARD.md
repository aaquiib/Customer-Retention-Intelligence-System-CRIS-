# Segmentation Model Card

**Model Version:** v1.0.0-segmentation-20260411_224248  
**Model Type:** KPrototypes Clustering  
**Release Date:** 2026-04-11  
**Status:** PRODUCTION  

---

## Model Overview

### Purpose
The Segmentation Model uses KPrototypes clustering to identify and classify customers into meaningful behavioral segments based on demographics, service usage, tenure, and financial characteristics. This enables targeted customer engagement strategies and personalized retention efforts.

### Use Case
- **Primary:** Customer segmentation for targeted retention and engagement campaigns
- **Secondary:** Risk stratification (identify at-risk high-value customers), cross-sell/upsell targeting
- **Stakeholders:** Marketing, Customer Success, Retention teams

### Audience
- Marketing teams designing segment-specific campaigns
- Customer Success managers prioritizing retention efforts
- Business analysts understanding customer composition

---

## Model Details

### Algorithm
- **Algorithm:** K-Prototypes Clustering (mixed numeric + categorical data)
- **Library:** kmodes (Python)
- **Initialization:** Cao initialization with 10 random seeds for stability
- **Number of Clusters:** 4 (determined via elbow method)

### Hyperparameters
```json
{
  "n_clusters": 4,
  "init": "Cao",
  "n_init": 10,
  "random_state": 42
}
```

### Training Data

| Metric | Value |
|--------|-------|
| **Total Samples** | 7,032 customers |
| **Features Used** | 25 (15 numeric, 10 categorical) |
| **Training Date** | 2026-04-11 |

### Features

#### Numeric Features (15)
- **Lifecycle:** tenure, avg_monthly_spend, tenure_band
- **Value:** MonthlyCharges, TotalCharges, charge_gap, is_high_value
- **Services:** streaming_count, security_count
- **Behavior:** payment_electronic_check, month_to_month_paperless, no_support_services, is_isolated, fiber_no_security, no_internet_services, SeniorCitizen

#### Categorical Features (10)
- **Demographics:** gender, Partner, Dependents, SeniorCitizen, tenure_band
- **Services:** PhoneService, MultipleLines, InternetService
- **Contract:** Contract, PaperlessBilling, PaymentMethod

---

## Customer Segments

### Segment 0: Loyal High-Value 💎
- **Size:** 1,456 customers (20.7%)
- **Churn Rate:** 8.4%
- **Profile:** Premium customers with strong loyalty (median tenure: 51 months, high spending: $88.95/mo)
- **Strategy:** Nurture for expansion, loyalty rewards, VIP treatment

### Segment 1: Low Engagement ⚠️
- **Size:** 981 customers (13.9%)
- **Churn Rate:** 31.2%
- **Profile:** Customers with limited service adoption (low charges: $32.15/mo, short tenure: 4 months)
- **Strategy:** Activation campaigns, service discovery, education programs

### Segment 2: Stable Mid-Value 👍
- **Size:** 3,142 customers (44.7%)
- **Churn Rate:** 21.7%
- **Profile:** Stable mid-tier customers (median tenure: 28 months, moderate spending: $52.35/mo)
- **Strategy:** Cross-sell opportunities, value bundle promotions, engagement programs

### Segment 3: At risk High-value 🚨
- **Size:** 1,453 customers (20.7%)
- **Churn Rate:** 50.4%
- **Profile:** High-value customers showing red flags (high spending: $94.65/mo, short tenure: 5 months, high churn)
- **Strategy:** **Immediate intervention:** win-back campaigns, loyalty incentives, dedicated support

---

## Model Performance

### Clustering Quality
| Metric | Value |
|--------|-------|
| **Final Model Cost** | 7,412.8 |
| **Elbow Point** | k=4 (clear reduction in cost from k=2 to k=4) |
| **Cluster Balance** | Relatively balanced (13.9% to 44.7% per cluster) |

### Elbow Curve (Cost by k)
```
k=2:  8,645.2
k=3:  7,832.1
k=4:  7,412.8  ← Selected
k=5:  7,154.3
k=6:  6,989.5
k=7:  6,842.1
k=8:  6,721.3
k=9:  6,615.8
k=10: 6,526.4
```

---

## Technical Specifications

### Preprocessing
- **Scaling:** StandardScaler applied to numeric features (mean: listed in feature_config.json)
- **Categorical Handling:** KPrototypes handles categorical features natively via distance metric
- **Missing Values:** None (data cleaned upstream in preprocessing pipeline)

### Feature Engineering
- **Value Features:** avg_monthly_spend, charge_gap, is_high_value (binary)
- **Service Counts:** streaming_count, security_count (aggregated from multi-dimensional features)
- **Risk Flags:** Engineered indicators (fiber_no_security, is_isolated, month_to_month_paperless)
- **Tenure Bands:** Categorical bins [0-12, 12-36, 36+] months

### Model Artifacts
```
models/v1.0.0-segmentation-20260411_224248/
├── metadata.json                 # Complete model metadata & cluster stats
├── kprototypes/
│   └── model.pkl                 # Serialized KPrototypes object
├── preprocessing/
│   ├── scaler.pkl                # StandardScaler for numeric features
│   ├── feature_config.json       # Feature names, types, scaler params
│   └── categorical_indices.json  # Column indices of categorical features
└── metadata/
    └── segment_labels.json       # Human-friendly segment names & descriptions
```

---

## Usage

### Load Model & Inference
```python
from models.artifacts import SegmentationModelLoader

loader = SegmentationModelLoader("models/v1.0.0-segmentation-20260411_224248")
segments = loader.predict(new_customer_data)  # Returns segment assignments (0-3)
labels = loader.predict_with_labels(new_customer_data)  # Includes human-friendly labels
```

### Expected Input
- **Format:** pandas DataFrame or numpy array
- **Features:** Must include all 25 segmentation features in correct order
- **Data Type:** Numeric for scaled features, categorical strings for categorical features

### Expected Output
- **Segment ID:** Integer 0-3 (cluster assignments)
- **Segment Label:** Human-readable segment name (e.g., "Loyal High-Value 💎")
- **Confidence:** Distance to centroid (optional, indicates strength of assignment)

---

## Limitations & Considerations

- **Not Causal:** Segments describe correlation patterns, not causal relationships
- **K Selection:** Elbow point at k=4 is visual; k=3 or k=5 may also be reasonable
- **Temporal Stability:** First trained 2026-04-11; performance may drift if customer characteristics change significantly
- **Data Requirements:** Assumes feature engineering pipeline (avg_monthly_spend, tenure_band, etc.) is applied before prediction
- **Scaling:** Numeric features MUST be scaled using the provided StandardScaler object

---

## Monitoring & Maintenance

### Key Metrics to Track
1. **Segment Distribution Drift:** Monitor % of customers in each segment monthly
2. **Churn Rates:** Track actual churn within predicted segments vs. baseline (8.4%, 31.2%, 21.7%, 50.4%)
3. **Feature Drift:** Monitor changes in mean/std of numeric features (compare to scaler stats in feature_config.json)

### Retraining Plan
- **Trigger:** >10% shift in segment distribution, >5% churn rate deviation, new business requirements
- **Frequency:** Quarterly review; retrain annually or when triggered
- **Validation:** Compare new model segments to existing on holdout test set

### Contact
- **Model Owner:** Customer Analytics Team
- **Questions:** analytics@company.com

---

## Model Versions

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| v1.0.0 | 2026-04-11 | PRODUCTION | Initial release with 4 clusters (KPrototypes) |

---

**Last Updated:** 2026-04-11
