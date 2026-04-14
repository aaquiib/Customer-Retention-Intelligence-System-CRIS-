# Explanation API Testing Report

## Summary
The explanation API endpoints are **PARTIALLY WORKING** with identified issues.

---

## Test Results

| Endpoint | Status | Issue |
|----------|--------|-------|
| `GET /api/explanations/methods` | ✅ WORKING | None - returns available explanation methods |
| `GET /api/explanations/model-info` | ✅ WORKING | None - returns model architecture info |
| `GET /api/feature-importance/global` | ❌ ERROR | Background data preprocessing failure |
| `POST /api/feature-importance/instance` | ❌ ERROR | Feature engineering validation failure |

---

## Issues Identified

### Issue 1: Global Feature Importance - Data Preprocessing Error

**Endpoint:** `GET /api/feature-importance/global`

**Error:**
```
ValueError: could not convert string to float: 'Male'
```

**Root Cause:**
The background data is generated with raw categorical string values ('Male', 'Yes', 'No', etc.), but when passed to SHAP's TreeExplainer, these need to be properly encoded as numeric values by the preprocessing pipeline.

**Code Location:**
- [src/inference/shap_explainer.py](inference/shap_explainer.py) - `_create_synthetic_background()` method creates raw data
- [src/inference/shap_explainer.py](inference/shap_explainer.py) - `_initialize_explainer()` attempts preprocessing but falls back to raw data

**What's Happening:**
1. Synthetic background data is created with categorical strings
2. The code tries to preprocess it using `pipeline.churn_preprocessor.transform()`
3. The preprocessing fails, and it catches the exception
4. It then tries to use raw data with SHAP, which fails because raw data contains strings
5. Error gets logged but returns `success=false` to the client

**Stack Trace:**
```
File "api/endpoints/explanations.py", line 50, in get_global_feature_importance
    explanation = model_cache.shap_explainer.get_global_importance(top_n=top_n)
File "inference/shap_explainer.py", line 179, in get_global_importance
    self._shap_values_global = self.explainer.shap_values(self.background_X)
File "lightgbm/basic.py", line 1073, in __inner_predict_np2d
    data = np.array(mat.reshape(mat.size), dtype=np.float32)
ValueError: could not convert string to float: 'Male'
```

---

### Issue 2: Instance Feature Importance - Feature Engineering Failure

**Endpoint:** `POST /api/feature-importance/instance`

**Error:**
```
ValueError: [SEGMENT_ASSIGNMENT] Missing numeric columns: ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_spend', 'charge_gap', 'streaming_count', 'security_count']
```

**Root Cause:**
The instance explanation endpoint receives raw customer data, but the prediction pipeline requires engineered features. The feature engineering step fails during validation because these derived features haven't been created yet.

**Code Location:**
- [api/endpoints/explanations.py](api/endpoints/explanations.py) - `get_instance_feature_importance()` method
- [inference/pipeline.py](inference/pipeline.py) - `_assign_segment()` method calls feature validation

**What's Happening:**
1. Raw customer data is passed to `pipeline.predict_single()`
2. Data goes through preprocessing successfully
3. Data should undergo feature engineering to create derived features
4. Feature validation check fails before engineered features are complete
5. The engineered features include: `avg_monthly_spend`, `charge_gap`, `streaming_count`, `security_count`

**Expected Flow:**
```
Raw Input → Preprocessing → Feature Engineering → Validation → Segmentation & Churn Prediction
```

**Stack Trace:**
```
File "api/endpoints/explanations.py", line 111, in get_instance_feature_importance
    prediction = model_cache.pipeline.predict_single(customer_dict)
File "inference/pipeline.py", line 122, in predict_single
    segment, segment_label, seg_confidence = self._assign_segment(df_engineered)
File "inference/pipeline.py", line 228, in _assign_segment
    validate_feature_consistency(...)
File "src/utils/feature_validation.py", line 43, in validate_feature_consistency
    raise ValueError(f"[{phase}] Missing numeric columns: {missing_num}")
```

---

## Required Fixes

### Fix #1: Properly Preprocess Background Data for SHAP

**Location:** [inference/shap_explainer.py](inference/shap_explainer.py#L73-L85)

**Current Code:**
```python
# Preprocess background data
try:
    df_background = self.pipeline.churn_preprocessor.transform(df_background)
except Exception as e:
    logger.warning(f"Could not transform background data: {e}. Using raw data.")
    df_background = df_background.to_numpy() if isinstance(df_background, pd.DataFrame) else df_background

if isinstance(df_background, pd.DataFrame):
    df_background = df_background.to_numpy()
```

**Problem:**
- Tries to transform but catches all exceptions and falls back to raw data
- Raw data contains strings which SHAP cannot use

**Solution:**
- Use `churn_preprocessor.fit_transform()` instead of just `transform()`
- Or pass raw data through the full preprocessing function used in the pipeline

---

### Fix #2: Ensure Feature Engineering Happens Before Validation

**Location:** [inference/pipeline.py](inference/pipeline.py#L105-L130)

**Current Flow:**
The pipeline does engineer features correctly, but there may be a validation issue. Need to:
1. Verify feature engineering happens completely
2. Check if validation is happening too early
3. Ensure all required engineered features are created before validation

---

## API Endpoints Summary

### ✅ Working Endpoints

#### 1. `GET /api/explanations/methods`
Returns available explanation methods (SHAP variants)

**Response:**
```json
{
  "available_methods": [
    {
      "name": "SHAP - Kernel Explainer",
      "type": "shap_kernel",
      "description": "Model-agnostic SHAP explanations using Kernel method",
      "speed": "slow",
      "accuracy": "high"
    },
    {
      "name": "SHAP - Tree Explainer",
      "type": "shap_tree",
      "description": "Fast SHAP explanations for tree-based models (LightGBM)",
      "speed": "fast",
      "accuracy": "high"
    }
  ],
  "default_method": "shap_tree",
  "note": "Tree Explainer recommended for LightGBM churn model"
}
```

#### 2. `GET /api/explanations/model-info`
Returns complete model architecture and training metrics

**Response:**
```json
{
  "model_type": "LightGBM Classifier",
  "n_estimators": 650,
  "max_depth": 13,
  "n_features": 33,
  "feature_columns": [...],
  "churn_threshold": 0.4356,
  "training_metrics": {
    "roc_auc": 0.8398,
    "precision": 0.53,
    "recall": 0.786,
    "f1": 0.633
  }
}
```

---

### ❌ Broken Endpoints

#### 3. `GET /api/feature-importance/global`
Currently returns error response due to background data preprocessing issue

**Expected Response (when fixed):**
```json
{
  "success": true,
  "explanation": {
    "top_features": [
      {
        "feature": "tenure",  
        "importance": 0.145,
        "sign": "positive"
      },
      {
        "feature": "MonthlyCharges",
        "importance": 0.089,
        "sign": "negative"
      }
    ],
    "explainer_type": "tree",
    "sample_size": 200
  }
}
```

#### 4. `POST /api/feature-importance/instance`
Currently returns error response due to feature engineering validation failure

**Request Body:**
```json
{
  "customer": {
    "gender": "Male",
    "tenure": 24,
    "MonthlyCharges": 65.50,
    "Contract": "Month-to-month",
    ...
  },
  "top_n": 5,
  "explanation_type": "shap"
}
```

**Expected Response (when fixed):**
```json
{
  "success": true,
  "explanation": {
    "prediction": {
      "segment": 2,
      "segment_label": "At-risk High-value",
      "churn_probability": 0.5006,
      "is_churner": true,
      ...
    },
    "top_features": [
      {
        "feature_name": "tenure",
        "importance": 0.23,
        "sign": "positive"
      }
    ]
  }
}
```

---

## Next Steps

1. **Fix Background Data Preprocessing** in SHAP explainer
   - Ensure synthetic background data goes through proper categorical encoding
   - Use the same preprocessing as the main pipeline

2. **Fix Feature Engineering Validation**
   - Verify feature engineering completes before validation
   - Check if validation check is missing engineered features

3. **Re-test Both Endpoints** after fixes

4. **Add Error Handling** to provide better error messages to clients

---

## Test Data Used

Customer profile for instance test:
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "MonthlyCharges": 65.50,
  "TotalCharges": 1570.70,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

---

## Conclusion

The explanation API infrastructure is in place and partially functional:
- ✅ 2/4 endpoints working (methods, model-info)
- ❌ 2/4 endpoints failing due to data preprocessing issues
- Both issues are fixable with targeted corrections to the SHAP and feature engineering pipeline
