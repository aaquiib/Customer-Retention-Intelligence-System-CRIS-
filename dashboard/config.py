"""Configuration and constants for the dashboard."""

import os
from dotenv import load_dotenv

# Load environment variables from root directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000") + "/api"
API_TIMEOUT = 10  # seconds
API_RETRIES = 3

# Model Configuration
CHURN_THRESHOLD = 0.4356

# Segment Configuration
SEGMENT_LABELS = {
    0: "Long-term Loyal",
    1: "Low Engagement",
    2: "Medium Engagement",
    3: "New/High-Value"
}

SEGMENT_DESCRIPTIONS = {
    0: "High tenure, stable charges, low churn risk",
    1: "Low tenure, few services, high churn risk",
    2: "Moderate tenure, some services, medium risk",
    3: "New customer or high monthly charges"
}

SEGMENT_COLORS = {
    0: "#2ecc71",  # Green
    1: "#e74c3c",  # Red
    2: "#f39c12",  # Orange
    3: "#3498db"   # Blue
}

# Risk Band Configuration
RISK_BANDS = {
    "Low": (0.0, 0.35),
    "Medium": (0.35, 0.65),
    "High": (0.65, 1.0)
}

RISK_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c"
}

# Customer Required Fields (19 total)
CUSTOMER_FIELDS = [
    # Demographic (4)
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    # Tenure & Charges (3)
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    # Services (10)
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    # Contract & Billing (2)
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Valid categorical values
CATEGORICAL_VALUES = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1, "0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# Numeric field ranges
NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 10000)
}

# Cache TTLs (in seconds)
CACHE_TTL_HEALTH = 300  # 5 minutes
CACHE_TTL_MODEL_INFO = 3600  # 1 hour
CACHE_TTL_GLOBAL_IMPORTANCE = 3600  # 1 hour
