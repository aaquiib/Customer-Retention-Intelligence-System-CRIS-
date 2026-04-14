# API Integration Report for Dashboard Development
## Comprehensive API Documentation

**Project:** Churn Segmentation Decision System  
**Version:** 1.0  
**Date:** April 14, 2026  
**Status:** Production Ready ✅

---

## Table of Contents
1. [API Overview](#api-overview)
2. [Server Configuration](#server-configuration)
3. [Authentication & CORS](#authentication--cors)
4. [Prediction Endpoints](#prediction-endpoints)
5. [Explanation Endpoints](#explanation-endpoints)
6. [What-If Simulation Endpoints](#what-if-simulation-endpoints)
7. [Data Schemas](#data-schemas)
8. [Error Handling](#error-handling)
9. [Integration Examples](#integration-examples)
10. [Performance & Limits](#performance--limits)

---

## API Overview

### Base URL
```
http://<hostname>:8000/api
```

### API Features
- ✅ RESTful design with JSON request/response
- ✅ CORS enabled for cross-origin requests
- ✅ Comprehensive error handling
- ✅ Request validation via Pydantic schemas
- ✅ OpenAPI documentation at `/docs`

### Health Check
```
GET http://<hostname>:8000/health
```
**Response:** `200 OK` when server is running

---

## Server Configuration

### Starting the API Server

**Using Uvicorn (Recommended):**
```bash
cd "e:\ML PROJECTS\churn-segmentation-decision_system"
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Parameters:**
- `--host 0.0.0.0` - Listen on all network interfaces
- `--port 8000` - Run on port 8000
- `--reload` - Auto-reload on code changes (development only)

**For Production:**
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Setup
```bash
# Activate virtual environment
cd "e:\ML PROJECTS\churn-segmentation-decision_system"
churn_env\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Required Files
- `api/app.py` - Application entry point
- `api/endpoints/` - Endpoint implementations
- `inference/pipeline.py` - ML pipeline
- `inference/shap_explainer.py` - SHAP explainability
- `models/` - Trained models (churn classifier, segmentation)
- `config/` - Configuration files

---

## Authentication & CORS

### Authentication
**Current:** No authentication required  
**Recommendation:** Add JWT tokens for production deployment

### CORS Configuration
```python
# Already enabled in api/app.py
CORSMiddleware(
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For restricting origins:**
```python
allow_origins=["http://localhost:3000", "https://yourdomain.com"]
```

---

## Prediction Endpoints

### 1. Single Customer Prediction
**Endpoint:** `POST /api/predict`

**Purpose:** Get churn prediction and segment assignment for a single customer

#### Request Schema
```json
{
  "customer": {
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
  },
  "return_features": false
}
```

#### Input Field Specifications

| Field | Type | Required | Valid Values | Description |
|-------|------|----------|--------------|-------------|
| customer | Object | Yes | - | Customer features object |
| return_features | Boolean | No | true/false | Include engineered features in response |

#### Customer Features (19 fields)

**Demographic (4 fields):**
- `gender` (string): "Male" or "Female"
- `SeniorCitizen` (integer): 0 or 1
- `Partner` (string): "Yes" or "No"
- `Dependents` (string): "Yes" or "No"

**Tenure & Charges (3 fields):**
- `tenure` (integer): Months with company (0-72)
- `MonthlyCharges` (float): Monthly recurring charge ($)
- `TotalCharges` (float): Total charges to date ($)

**Services (10 fields):**
- `PhoneService` (string): "Yes" or "No"
- `MultipleLines` (string): "Yes", "No", "No phone service"
- `InternetService` (string): "DSL", "Fiber optic", "No"
- `OnlineSecurity` (string): "Yes", "No", "No internet service"
- `OnlineBackup` (string): "Yes", "No", "No internet service"
- `DeviceProtection` (string): "Yes", "No", "No internet service"
- `TechSupport` (string): "Yes", "No", "No internet service"
- `StreamingTV` (string): "Yes", "No", "No internet service"
- `StreamingMovies` (string): "Yes", "No", "No internet service"

**Contract & Billing (2 fields):**
- `Contract` (string): "Month-to-month", "One year", "Two year"
- `PaperlessBilling` (string): "Yes" or "No"
- `PaymentMethod` (string): "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

#### Response Schema
```json
{
  "success": true,
  "prediction": {
    "segment": 1,
    "segment_label": "Low Engagement",
    "segment_confidence": 0.95,
    "churn_probability": 0.8432,
    "is_churner": true,
    "threshold": 0.4356,
    "top_features": null,
    "recommended_action": null,
    "input_features": {
      "gender": "Male",
      "SeniorCitizen": 0,
      ...
    },
    "engineered_features": null
  },
  "error": null
}
```

#### Response Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| success | Boolean | Request succeeded |
| prediction.segment | Integer | Segment ID (0-3) |
| prediction.segment_label | String | Human-readable segment name |
| prediction.segment_confidence | Float | Confidence of segment assignment (0-1) |
| prediction.churn_probability | Float | Probability customer will churn (0-1) |
| prediction.is_churner | Boolean | True if churn_probability > threshold |
| prediction.threshold | Float | Decision threshold used (0.4356) |
| prediction.recommended_action | Object | Retention action recommendation |
| prediction.input_features | Object | Echo of input features |
| prediction.engineered_features | Object | Derived features (if return_features=true) |

#### Segment Definitions

| ID | Label | Characteristics |
|----|-------|-----------------|
| 0 | Long-term Loyal | High tenure, stable charges, low churn risk |
| 1 | Low Engagement | Low tenure, few services, high churn risk |
| 2 | Medium Engagement | Moderate tenure, some services, medium risk |
| 3 | New/High-Value | New customer or high monthly charges |

#### HTTP Status Codes
- `200 OK` - Prediction successful
- `422 Unprocessable Entity` - Invalid input data
- `503 Service Unavailable` - Models not loaded

#### Example cURL Request
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
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
    },
    "return_features": false
  }'
```

#### Example JavaScript/TypeScript Integration
```typescript
async function predictChurn(customerData: CustomerInput) {
  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      customer: customerData,
      return_features: false
    })
  });

  const result = await response.json();
  
  if (result.success) {
    console.log(`Churn Probability: ${result.prediction.churn_probability.toFixed(4)}`);
    console.log(`Segment: ${result.prediction.segment_label}`);
    console.log(`Action: ${result.prediction.recommended_action.action_label}`);
  } else {
    console.error(`Error: ${result.error}`);
  }
  
  return result;
}
```

---

### 2. Batch Prediction
**Endpoint:** `POST /api/predict-batch`

**Purpose:** Bulk score multiple customers via CSV file upload (up to 50,000 customers)

#### Request Format
**Content-Type:** `multipart/form-data`  
**Body:** CSV file upload

The CSV must contain exactly 19 columns with customer data. Column order doesn't matter, but all 19 columns required:

```csv
gender,SeniorCitizen,Partner,Dependents,tenure,MonthlyCharges,TotalCharges,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod
Male,0,Yes,No,24,65.50,1570.70,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check
Female,1,No,Yes,12,45.25,543.00,No,No,DSL,Yes,Yes,No,Yes,Yes,No,One year,No,Credit card (automatic)
```

#### Parameters
- `file` (File, required): CSV file with 19-column customer data
  - Supports up to **50,000 rows** per request
  - Column names are **case-insensitive**
  - All 19 required columns must be present
  - Handles missing values gracefully

#### Response Schema
```json
{
  "success": true,
  "results": [
    {
      "customer_id": "Row 0",
      "prediction": {
        "churn_probability": 0.8432,
        "segment": 1,
        "segment_confidence": 0.95,
        "top_features": [
          {
            "name": "Contract",
            "impact": -0.45,
            "base_value": 0.32
          }
        ],
        "recommended_action": "Retention Call"
      },
      "error": null,
      "success": true
    }
  ],
  "summary": {
    "total_rows": 2,
    "rows_processed": 2,
    "rows_failed": 0,
    "churn_rate": 0.50,
    "avg_churn_probability": 0.8432,
    "avg_segment_confidence": 0.95,
    "segment_distribution": {
      "0": 0,
      "1": 1,
      "2": 1,
      "3": 0
    },
    "action_distribution": {
      "Retention Call": 1,
      "Loyalty Program": 0,
      "Service Upgrade": 1,
      "Billing Review": 0
    }
  },
  "error": null
}
```

#### HTTP Status Codes
- `200 OK` - Batch processing completed (check individual results)
- `400 Bad Request` - CSV format issues or missing columns
- `413 Payload Too Large` - File exceeds maximum size
- `422 Unprocessable Entity` - Invalid data format or values
- `503 Service Unavailable` - Models not loaded

#### Key Features
- Individual row predictions with SHAP values
- Batch summary statistics (churn rate, segment distribution)
- Error handling per row (partial failures don't stop batch)
- Automatic column normalization (case-insensitive)
- Processing time: ~50-100ms for 1,000 customers, ~500-1000ms for 10,000 customers

#### Limits
- **Maximum 50,000 rows** per batch request
- Column count must be exactly 19
- File size limit typically 100MB+ (depends on server config)

---

### 3. Batch Template
**Endpoint:** `GET /api/predict-batch/template`

**Purpose:** Get CSV template for batch uploads

#### Response
Returns a CSV file with example structure:
```csv
gender,SeniorCitizen,Partner,Dependents,tenure,MonthlyCharges,TotalCharges,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod
Male,0,Yes,No,24,65.50,1570.70,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check
```

#### Example cURL
```bash
curl "http://localhost:8000/api/predict-batch/template" -o batch_template.csv
```

---

## Explanation Endpoints

### 1. Global Feature Importance
**Endpoint:** `GET /api/feature-importance/global`

**Purpose:** Get top features driving churn predictions across all customers

#### Query Parameters
| Parameter | Type | Default | Max | Description |
|-----------|------|---------|-----|-------------|
| top_n | Integer | 10 | 33 | Number of features to return |

#### Response Schema
```json
{
  "success": true,
  "explanation": {
    "top_features": [
      {
        "feature_name": "num__tenure",
        "importance": 0.342,
        "sign": "negative"
      },
      {
        "feature_name": "cat__Contract_Two year",
        "importance": 0.187,
        "sign": "negative"
      }
    ],
    "explainer_type": "tree",
    "sample_size": 200
  },
  "error": null
}
```

#### Field Descriptions
- `feature_name`: Name of the feature (prefix indicates type: `num__` = numeric, `cat__` = categorical)
- `importance`: Average absolute SHAP value (higher = more important)
- `sign`: Direction of impact ("positive" increases churn, "negative" decreases churn)
- `sample_size`: Number of background samples used (200)

#### HTTP Status Codes
- `200 OK` - Successfully computed
- `503 Service Unavailable` - SHAP explainer not initialized

#### Example cURL
```bash
curl "http://localhost:8000/api/feature-importance/global?top_n=10"
```

#### Example JavaScript Integration
```javascript
async function getGlobalImportance() {
  const response = await fetch('http://localhost:8000/api/feature-importance/global?top_n=10');
  const data = await response.json();
  
  if (data.success) {
    data.explanation.top_features.forEach(feature => {
      console.log(`${feature.feature_name}: ${feature.importance.toFixed(4)} (${feature.sign})`);
    });
  }
}
```

---

### 2. Instance Feature Importance
**Endpoint:** `POST /api/feature-importance/instance`

**Purpose:** Get feature contributions to a specific customer's prediction (SHAP values)

#### Request Schema
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
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

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| top_n | Integer | 5 | Number of top contributing features |

#### Response Schema
```json
{
  "success": true,
  "explanation": {
    "prediction": {
      "segment": 1,
      "segment_label": "Low Engagement",
      "segment_confidence": 0.95,
      "churn_probability": 0.8432,
      "is_churner": true,
      "threshold": 0.4356,
      "input_features": { ... }
    },
    "top_features": [
      {
        "feature_name": "num__tenure",
        "importance": 0.852,
        "sign": "positive"
      },
      {
        "feature_name": "cat__Contract_Two year",
        "importance": 0.532,
        "sign": "positive"
      }
    ],
    "base_value": -1.085
  },
  "error": null
}
```

#### Field Descriptions
- `top_features[].importance`: SHAP value for this feature (positive = increases churn)
- `base_value`: Expected value (baseline) for the model
- `sign`: Direction of impact for this specific instance

#### HTTP Status Codes
- `200 OK` - Successfully computed
- `422 Unprocessable Entity` - Invalid customer data
- `503 Service Unavailable` - SHAP explainer not initialized

#### Example cURL
```bash
curl -X POST "http://localhost:8000/api/feature-importance/instance?top_n=5" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "SeniorCitizen": 0, ...}'
```

---

### 3. Available Explanation Methods
**Endpoint:** `GET /api/explanations/methods`

**Purpose:** List available explainability methods

#### Response Schema
```json
{
  "success": true,
  "explanation": {
    "available_methods": [
      {
        "name": "SHAP TreeExplainer",
        "type": "tree",
        "description": "SHAP TreeExplainer for tree-based models (LightGBM)",
        "applicable_to": "churn_model"
      }
    ]
  },
  "error": null
}
```

#### HTTP Status Codes
- `200 OK` - Successfully retrieved

---

### 4. Model Information
**Endpoint:** `GET /api/explanations/model-info`

**Purpose:** Get model architecture, training details, and metadata

#### Response Schema
```json
{
  "success": true,
  "explanation": {
    "churn_model": {
      "name": "LightGBM Binary Classifier",
      "type": "Gradient Boosting",
      "features": 33,
      "feature_names": [
        "num__tenure",
        "num__MonthlyCharges",
        ...
      ],
      "performance": {
        "roc_auc": 0.8456,
        "accuracy": 0.8034,
        "precision": 0.7821,
        "recall": 0.6234
      },
      "threshold": 0.4356,
      "training_data": "7000+ customer records"
    },
    "segmentation_model": {
      "name": "K-Prototypes Clustering",
      "type": "Hybrid Clustering",
      "n_clusters": 4,
      "algorithm": "K-Prototypes with categorical + numerical features",
      "segments": [
        {
          "id": 0,
          "label": "Long-term Loyal",
          "characteristics": "High tenure, stable charges, low churn"
        },
        {
          "id": 1,
          "label": "Low Engagement",
          "characteristics": "Low tenure, few services, high churn"
        },
        {
          "id": 2,
          "label": "Medium Engagement",
          "characteristics": "Moderate tenure, some services"
        },
        {
          "id": 3,
          "label": "New/High-Value",
          "characteristics": "New customer or high value"
        }
      ]
    }
  },
  "error": null
}
```

---

## What-If Simulation Endpoints

### 1. Single What-If Scenario
**Endpoint:** `POST /api/what-if`

**Purpose:** Simulate prediction changes by modifying customer features

#### Request Schema
```json
{
  "customer": {
    "gender": "Male",
    "SeniorCitizen": 0,
    ...
  },
  "modifications": {
    "Contract": "Two year",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes"
  }
}
```

#### Response Schema
```json
{
  "success": true,
  "original_prediction": {
    "segment": 1,
    "segment_label": "Low Engagement",
    "churn_probability": 0.8432,
    "is_churner": true,
    ...
  },
  "modified_prediction": {
    "segment": 1,
    "segment_label": "Low Engagement",
    "churn_probability": 0.2734,
    "is_churner": false,
    ...
  },
  "delta": {
    "segment_changed": false,
    "segment_delta": null,
    "churn_probability_delta": -0.5698,
    "is_churner_changed": true,
    "action_changed": false
  },
  "modified_features": {
    "Contract": "Two year",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes"
  },
  "error": null
}
```

#### Delta Field Descriptions
- `churn_probability_delta`: Change in churn probability (negative = improvement)
- `is_churner_changed`: Whether customer crossed threshold
- `segment_changed`: Whether segment assignment changed

#### HTTP Status Codes
- `200 OK` - Simulation completed
- `422 Unprocessable Entity` - Invalid data
- `503 Service Unavailable` - Models not loaded

#### Example Use Cases
- Contract upgrade: `{"Contract": "Two year"}`
- Add services: `{"OnlineSecurity": "Yes", "TechSupport": "Yes"}`
- Payment method: `{"PaymentMethod": "Bank transfer (automatic)"}`
- Tenure increase: `{"tenure": 50}`

---

### 2. Batch What-If Scenarios
**Endpoint:** `POST /api/what-if/batch`

**Purpose:** Run multiple what-if simulations in parallel

#### Request Schema
```json
[
  {
    "customer": { ... },
    "modifications": { "Contract": "Two year" }
  },
  {
    "customer": { ... },
    "modifications": { "OnlineSecurity": "Yes", "TechSupport": "Yes" }
  }
]
```

#### Response Schema
```json
{
  "success": true,
  "total_simulations": 2,
  "successful": 2,
  "results": [
    { ... },
    { ... }
  ]
}
```

#### Limits
- Maximum 100 scenarios per batch request

---

### 3. Pre-defined Policy Scenarios
**Endpoint:** `GET /api/what-if/policy-changes`

**Purpose:** Get common retention strategy scenarios

#### Response Schema
```json
{
  "scenarios": [
    {
      "name": "Contract Upgrade",
      "description": "Customer upgrades from month-to-month to 2-year contract",
      "modifications": {
        "Contract": "Two year"
      }
    },
    {
      "name": "Add Security Services",
      "description": "Customer adds OnlineSecurity and TechSupport services",
      "modifications": {
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes"
      }
    },
    {
      "name": "Switch to Auto-Pay",
      "description": "Customer switches to automatic bank transfer billing",
      "modifications": {
        "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "Yes"
      }
    },
    {
      "name": "Premium Services Bundle",
      "description": "Customer adds all premium services",
      "modifications": {
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes"
      }
    }
  ]
}
```

#### HTTP Status Codes
- `200 OK` - Successfully retrieved

---

## Data Schemas

### CustomerInput (Pydantic Model)
```python
class CustomerInput(BaseModel):
    # Demographic
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    
    # Tenure & Charges
    tenure: Optional[int] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None
    
    # Services
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    
    # Contract & Billing
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
```

### PredictionOutput (Pydantic Model)
```python
class PredictionOutput(BaseModel):
    # Segment Information
    segment: int  # 0-3
    segment_label: str
    segment_confidence: float  # 0.0-1.0
    
    # Churn Prediction
    churn_probability: float  # 0.0-1.0
    is_churner: bool
    threshold: float
    
    # Optional Fields
    top_features: Optional[List[SHAPFeature]] = None
    recommended_action: Optional[RecommendedAction] = None
    
    # Input Echo
    input_features: Dict[str, Any]
    engineered_features: Optional[Dict[str, Any]] = None
```

---

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "prediction": null,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

#### 422 Unprocessable Entity
**Cause:** Invalid input data (missing fields, wrong types)
```json
{
  "detail": [
    {
      "type": "int_parsing",
      "loc": ["body", "SeniorCitizen"],
      "msg": "Input should be a valid integer",
      "input": "invalid"
    }
  ]
}
```
**Solution:** Check field types match specification

#### 503 Service Unavailable
**Cause:** Models not loaded (server still starting)
```json
{
  "detail": "Models not loaded"
}
```
**Solution:** Wait for server to fully initialize (~2 seconds)

#### Unknown Features in What-If
**Error:** `"Unknown feature XYZ in modifications - skipping"`
**Solution:** Use only valid customer feature names

#### Invalid Category Values
**Error:** `"Found unknown categories ['Bank transfer'] in column 15"`
**Solution:** Use valid values from the categorical features list

### Error Recovery Strategy
1. Check `success` field in response
2. If `false`, read `error` field for description
3. For validation errors, check field names and types
4. Retry request after fixing issues
5. Log errors with timestamps for debugging

---

## Integration Examples

### React Component Example
```typescript
import React, { useState } from 'react';

interface ChurnDashboardProps {
  apiUrl: string;
}

export const ChurnDashboard: React.FC<ChurnDashboardProps> = ({ apiUrl }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState<string | null>(null);

  const predictChurn = async (customerData: any) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customer: customerData,
          return_features: false
        })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.prediction);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(`Network error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getExplanation = async (customerData: any) => {
    try {
      const response = await fetch(
        `${apiUrl}/feature-importance/instance?top_n=5`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(customerData)
        }
      );

      const data = await response.json();
      return data.explanation?.top_features || [];
    } catch (err) {
      console.error('Failed to get explanation:', err);
      return [];
    }
  };

  return (
    <div className="churn-dashboard">
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="results">
          <h2>Prediction Results</h2>
          <p>Churn Probability: {(result.churn_probability * 100).toFixed(2)}%</p>
          <p>Segment: {result.segment_label}</p>
          <p>Risk: {result.is_churner ? 'High' : 'Low'}</p>
        </div>
      )}
    </div>
  );
};
```

### Python Integration Example
```python
import requests
import json

class ChurnAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.session = requests.Session()

    def predict(self, customer_data: dict) -> dict:
        """Get single prediction"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json={"customer": customer_data, "return_features": False}
        )
        return response.json()

    def predict_batch(self, customers: list) -> dict:
        """Get batch predictions"""
        response = self.session.post(
            f"{self.base_url}/predict-batch",
            json={"customers": customers, "return_features": False}
        )
        return response.json()

    def get_global_importance(self, top_n: int = 10) -> dict:
        """Get global feature importance"""
        response = self.session.get(
            f"{self.base_url}/feature-importance/global",
            params={"top_n": top_n}
        )
        return response.json()

    def get_instance_importance(self, customer_data: dict, top_n: int = 5) -> dict:
        """Get instance-level SHAP values"""
        response = self.session.post(
            f"{self.base_url}/feature-importance/instance",
            params={"top_n": top_n},
            json=customer_data
        )
        return response.json()

    def what_if_scenario(self, customer_data: dict, modifications: dict) -> dict:
        """Simulate what-if scenario"""
        response = self.session.post(
            f"{self.base_url}/what-if",
            json={"customer": customer_data, "modifications": modifications}
        )
        return response.json()

    def get_policy_scenarios(self) -> dict:
        """Get pre-defined policy scenarios"""
        response = self.session.get(f"{self.base_url}/what-if/policy-changes")
        return response.json()

# Usage
client = ChurnAPIClient()

customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    # ... other fields
}

# Get prediction
result = client.predict(customer)
print(f"Churn Probability: {result['prediction']['churn_probability']:.4f}")

# Get explanation
explanation = client.get_instance_importance(customer, top_n=5)
for feature in explanation['explanation']['top_features']:
    print(f"{feature['feature_name']}: {feature['importance']:.4f}")

# What-if simulation
scenario = client.what_if_scenario(customer, {"Contract": "Two year"})
print(f"Original churn prob: {scenario['original_prediction']['churn_probability']:.4f}")
print(f"Modified churn prob: {scenario['modified_prediction']['churn_probability']:.4f}")
print(f"Delta: {scenario['delta']['churn_probability_delta']:.4f}")
```

### cURL Integration Examples
```bash
# Single prediction
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"customer": {"gender": "Male", "SeniorCitizen": 0, ...}}'

# Batch predictions
curl -X POST "http://localhost:8000/api/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"customers": [{...}, {...}]}'

# Global importance
curl "http://localhost:8000/api/feature-importance/global?top_n=10"

# Instance importance
curl -X POST "http://localhost:8000/api/feature-importance/instance?top_n=5" \
  -H "Content-Type: application/json" \
  -d '{"gender": "Male", "SeniorCitizen": 0, ...}'

# What-if scenario
curl -X POST "http://localhost:8000/api/what-if" \
  -H "Content-Type: application/json" \
  -d '{"customer": {...}, "modifications": {"Contract": "Two year"}}'

# Policy scenarios
curl "http://localhost:8000/api/what-if/policy-changes"
```

---

## Performance & Limits

### Response Time (Typical)
| Operation | Latency | Notes |
|-----------|---------|-------|
| Single Prediction | ~100ms | Includes preprocessing & feature engineering |
| Global Importance | ~150ms | SHAP computation on 200 samples |
| Instance Importance | ~150ms | Per-customer SHAP values |
| What-If Scenario | ~50ms | Just prediction, no SHAP |
| Batch (100 customers) | ~200ms | Parallel processing |

### Throughput
- Single predictions: ~10 req/sec
- Batch predictions: Up to 50,000 customers per request
- Global importance: Cached (computed once at startup)

### Limits
| Limit | Value | Notes |
|-------|-------|-------|
| Max customers per batch | 50,000 | CSV file upload with 19 columns |
| Max what-if simulations | 100 | Split larger batches |
| Max features to return | 33 | Total engineered features |
| Request timeout | 30s | Uvicorn default |
| Memory usage | ~500MB | Model + preprocessor |

### Scaling Recommendations
- **Single Server:** ~100 concurrent requests
- **Production Deployment:** Use multiple workers with load balancer
- **Large Batches:** Batch processing supports up to 50,000 customers in a single request
- **Database:** Cache predictions for batch results if needed
- **Monitoring:** Track response times and error rates

### Resource Requirements
- **CPU:** 2+ cores recommended
- **RAM:** 2GB minimum, 4GB+ recommended
- **Disk:** 500MB for models + data
- **Network:** Low latency preferred (<50ms to clients)

---

## Dashboard Integration Checklist

- [ ] API server deployed and accessible
- [ ] Base URL configured in dashboard settings
- [ ] Customer form follows 19-field schema exactly
- [ ] CORS configured for dashboard domain
- [ ] Error handling implemented for all 3 error types
- [ ] Loading states for slow endpoints (explanation requires 150ms)
- [ ] Batch processing integrated for bulk scoring
- [ ] What-if scenarios displayed in UI
- [ ] SHAP feature importance visualized
- [ ] Segment information displayed with context
- [ ] Monitoring/logging integrated
- [ ] Fallback behavior if API unavailable
- [ ] Rate limiting implemented if needed
- [ ] Authentication added for production

---

## Support & Troubleshooting

### API Not Starting
```bash
# Check if models are loading
python -c "from inference.pipeline import InferencePipeline; InferencePipeline()"

# Check Python environment
python -m pip list | grep -E "lightgbm|shap|fastapi"
```

### Slow Predictions
- Check CPU/memory usage
- Increase workers: `--workers 4`
- Profile with: `python -m cProfile -s cumtime api/app.py`

### Models Not Loaded
- Verify model files exist in `models/` directory
- Check `config/config.yaml` for correct paths
- Review startup logs for errors

### CORS Issues
- Check dashboard URL in CORSMiddleware
- Test with: `curl -H "Origin: http://localhost:3000" -v http://localhost:8000`

---

## Version Information

- **API Version:** 1.0
- **Python:** 3.8+
- **FastAPI:** 0.95+
- **LightGBM:** 3.3+
- **SHAP:** 0.41+
- **Release Date:** April 14, 2026
- **Status:** Production Ready ✅

---

**For questions or issues, review the endpoint examples and error codes above.**

