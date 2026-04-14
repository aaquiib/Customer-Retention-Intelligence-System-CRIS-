"""Data processing utilities for transformations and aggregations."""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

from config import SEGMENT_LABELS, RISK_BANDS

logger = logging.getLogger(__name__)


def aggregate_batch_summary(predictions_list: List[Dict]) -> Dict[str, Any]:
    """Extract summary statistics from batch predictions.
    
    Args:
        predictions_list: List of prediction dictionaries from API response
        
    Returns:
        Dictionary with summary stats
    """
    if not predictions_list:
        return {
            "total_rows": 0,
            "rows_processed": 0,
            "rows_failed": 0,
            "churn_rate": 0.0,
            "avg_churn_probability": 0.0,
            "segment_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
            "action_distribution": {}
        }
    
    df = pd.DataFrame(predictions_list)
    
    total_rows = len(df)
    rows_failed = df["error"].notna().sum() if "error" in df.columns else 0
    rows_processed = total_rows - rows_failed
    
    # Filter out failed rows
    df_valid = df[df["error"].isna()] if "error" in df.columns else df
    
    churn_rate = 0.0
    avg_churn_prob = 0.0
    segment_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    action_dist = {}
    
    if len(df_valid) > 0:
        churn_rate = df_valid["is_churner"].sum() / len(df_valid)
        avg_churn_prob = df_valid["churn_probability"].mean()
        
        # Segment distribution
        if "segment" in df_valid.columns:
            segment_counts = df_valid["segment"].value_counts().to_dict()
            for seg_id in range(4):
                segment_dist[seg_id] = segment_counts.get(seg_id, 0)
        
        # Action distribution
        if "recommended_action" in df_valid.columns:
            # Extract action labels from dict objects (API returns dict with action_label, priority_score, reason)
            actions = df_valid["recommended_action"].apply(
                lambda x: x.get("action_label", "Unknown") if isinstance(x, dict) else str(x)
            )
            action_counts = actions.value_counts().to_dict()
            for action, count in action_counts.items():
                action_dist[action] = int(count)
    
    return {
        "total_rows": total_rows,
        "rows_processed": rows_processed,
        "rows_failed": rows_failed,
        "churn_rate": round(churn_rate, 4),
        "avg_churn_probability": round(avg_churn_prob, 4),
        "segment_distribution": segment_dist,
        "action_distribution": action_dist
    }


def build_enriched_csv(predictions_list: List[Dict], original_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build enriched CSV with predictions and top features.
    
    Args:
        predictions_list: List of prediction dictionaries
        original_df: Original CSV data (optional, for input fields)
        
    Returns:
        Pandas DataFrame with enriched data
    """
    enriched_rows = []
    
    for idx, pred in enumerate(predictions_list):
        # Extract action label from dict if present
        action = pred.get("recommended_action", "")
        action_label = action.get("action_label", "Unknown") if isinstance(action, dict) else str(action)
        
        row = {
            "customerID": idx,
            "segment": pred.get("segment", ""),
            "segment_label": pred.get("segment_label", ""),
            "segment_confidence": round(pred.get("segment_confidence", 0), 4),
            "churn_probability": round(pred.get("churn_probability", 0), 4),
            "is_churner": pred.get("is_churner", False),
            "recommended_action": action_label,
            "MonthlyCharges": pred.get("input_features", {}).get("MonthlyCharges", ""),
            "tenure": pred.get("input_features", {}).get("tenure", ""),
            "Contract": pred.get("input_features", {}).get("Contract", ""),
        }
        
        # Add top 3 features
        top_features = pred.get("top_features", [])
        for i in range(3):
            if i < len(top_features):
                row[f"top_feature_{i+1}"] = top_features[i].get("feature_name", "")
            else:
                row[f"top_feature_{i+1}"] = ""
        
        enriched_rows.append(row)
    
    return pd.DataFrame(enriched_rows)


def segment_filter(df: pd.DataFrame, segment_ids: List[int]) -> pd.DataFrame:
    """Filter dataframe by segment IDs.
    
    Args:
        df: Input DataFrame with 'segment' column
        segment_ids: List of segment IDs to keep
        
    Returns:
        Filtered DataFrame
    """
    if not segment_ids or "segment" not in df.columns:
        return df
    return df[df["segment"].isin(segment_ids)]


def risk_band_filter(df: pd.DataFrame, risk_bands: List[str]) -> pd.DataFrame:
    """Filter dataframe by risk bands.
    
    Args:
        df: Input DataFrame with 'churn_probability' column
        risk_bands: List of band names ("Low", "Medium", "High")
        
    Returns:
        Filtered DataFrame
    """
    if not risk_bands or "churn_probability" not in df.columns:
        return df
    
    mask = pd.Series([False] * len(df))
    for band in risk_bands:
        if band == "Low":
            mask |= df["churn_probability"] < 0.35
        elif band == "Medium":
            mask |= (df["churn_probability"] >= 0.35) & (df["churn_probability"] < 0.65)
        elif band == "High":
            mask |= df["churn_probability"] >= 0.65
    
    return df[mask]


def calculate_revenue_at_risk(df: pd.DataFrame) -> float:
    """Calculate total revenue at risk (churners' monthly charges).
    
    Args:
        df: DataFrame with 'MonthlyCharges' and 'is_churner' columns
        
    Returns:
        Total revenue at risk
    """
    if "MonthlyCharges" not in df.columns or "is_churner" not in df.columns:
        return 0.0
    
    churners = df[df["is_churner"] == True]
    return float(churners["MonthlyCharges"].sum())


def build_segment_stats(df: pd.DataFrame, segment_id: int) -> Dict[str, Any]:
    """Build statistics for a specific segment.
    
    Args:
        df: DataFrame with predictions and input features
        segment_id: Segment ID (0-3)
        
    Returns:
        Dictionary with segment statistics
    """
    if "segment" not in df.columns:
        return {
            "size": 0,
            "churn_rate": 0.0,
            "avg_tenure": 0.0,
            "avg_monthly_charges": 0.0,
            "avg_confidence": 0.0,
            "dominant_contract": "N/A",
            "dominant_internet_service": "N/A"
        }
    
    seg_df = df[df["segment"] == segment_id]
    
    if len(seg_df) == 0:
        return {
            "size": 0,
            "churn_rate": 0.0,
            "avg_tenure": 0.0,
            "avg_monthly_charges": 0.0,
            "avg_confidence": 0.0,
            "dominant_contract": "N/A",
            "dominant_internet_service": "N/A"
        }
    
    churn_rate = (seg_df["is_churner"].sum() / len(seg_df)) if "is_churner" in seg_df.columns else 0
    
    avg_tenure = round(seg_df["tenure"].mean(), 2) if "tenure" in seg_df.columns else 0.0
    
    # Safe MonthlyCharges conversion (might be string from API)
    avg_charges = 0.0
    if "MonthlyCharges" in seg_df.columns:
        try:
            monthly_numeric = pd.to_numeric(seg_df["MonthlyCharges"], errors="coerce")
            avg_charges = round(monthly_numeric.mean(), 2)
        except:
            avg_charges = 0.0
    
    avg_confidence = round(seg_df["segment_confidence"].mean(), 4) if "segment_confidence" in seg_df.columns else 0.0
    
    dominant_contract = "N/A"
    if "Contract" in seg_df.columns and len(seg_df["Contract"]) > 0:
        mode_result = seg_df["Contract"].mode()
        if len(mode_result) > 0 and not mode_result.empty:
            dominant_contract = str(mode_result.iloc[0])
    
    dominant_service = "N/A"
    if "InternetService" in seg_df.columns and len(seg_df["InternetService"]) > 0:
        mode_result = seg_df["InternetService"].mode()
        if len(mode_result) > 0 and not mode_result.empty:
            dominant_service = str(mode_result.iloc[0])
    
    stats = {
        "size": len(seg_df),
        "churn_rate": round(churn_rate, 4),
        "avg_tenure": avg_tenure,
        "avg_monthly_charges": avg_charges,
        "avg_confidence": avg_confidence,
        "dominant_contract": dominant_contract,
        "dominant_internet_service": dominant_service
    }
    
    return stats


def create_risk_distribution_bins(df: pd.DataFrame, n_bins: int = 10) -> Dict[str, int]:
    """Create risk distribution bins for histogram.
    
    Args:
        df: DataFrame with 'churn_probability' column
        n_bins: Number of bins
        
    Returns:
        Dictionary with bin ranges and counts
    """
    if "churn_probability" not in df.columns or len(df) == 0:
        return {}
    
    bins = pd.cut(df["churn_probability"], bins=n_bins)
    counts = bins.value_counts().sort_index()
    
    result = {}
    for interval, count in counts.items():
        bin_label = f"{interval.left:.2f} - {interval.right:.2f}"
        result[bin_label] = int(count)
    
    return result


def get_top_customers_by_risk(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Get top N highest-risk customers.
    
    Args:
        df: DataFrame with predictions
        n: Number of customers to return
        
    Returns:
        DataFrame sorted by churn probability descending
    """
    if "churn_probability" not in df.columns:
        return df.head(n)
    
    return df.nlargest(n, "churn_probability")


def prepare_batch_result_df(batch_predictions: List[Dict]) -> pd.DataFrame:
    """Convert batch prediction API response to DataFrame for analysis.
    
    Args:
        batch_predictions: List of prediction dicts from predict-batch endpoint
        
    Returns:
        Pandas DataFrame with all predictions (skips errors)
    """
    if not batch_predictions:
        # Return empty DataFrame with expected schema to avoid errors downstream
        # Include all 19 customer fields + analysis fields for What-If simulator compatibility
        return pd.DataFrame(columns=[
            "customerID", "segment", "segment_label", "segment_confidence",
            "churn_probability", "is_churner", "recommended_action",
            # All 19 customer input fields
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "MonthlyCharges", "TotalCharges", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
            # Feature importance
            "top_feature_1", "top_feature_2", "top_feature_3"
        ])
    
    rows = []
    for idx, pred in enumerate(batch_predictions):
        if pred.get("error"):
            continue  # Skip failed predictions
        
        # Extract action label from dict if present
        action = pred.get("recommended_action", "")
        action_label = action.get("action_label", "Unknown") if isinstance(action, dict) else str(action)
        
        # Safe MonthlyCharges conversion (API might return string)
        input_features = pred.get("input_features", {})
        monthly_charges = input_features.get("MonthlyCharges", 0)
        try:
            monthly_charges = float(monthly_charges) if monthly_charges else 0.0
        except (ValueError, TypeError):
            monthly_charges = 0.0
        
        row = {
            "customerID": idx,
            "segment": pred.get("segment", -1),
            "segment_label": pred.get("segment_label", ""),
            "segment_confidence": pred.get("segment_confidence", 0),
            "churn_probability": pred.get("churn_probability", 0),
            "is_churner": pred.get("is_churner", False),
            "recommended_action": action_label,
            "MonthlyCharges": monthly_charges,
        }
        
        # Add ALL 19 customer input features (required for What-If simulator)
        # These are the features needed for model predictions
        all_customer_fields = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "TotalCharges", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod"
        ]
        for field in all_customer_fields:
            row[field] = input_features.get(field, "")
        
        # Add top 3 features
        top_features = pred.get("top_features", [])
        for i in range(3):
            if i < len(top_features):
                row[f"top_feature_{i+1}"] = top_features[i].get("feature_name", "")
            else:
                row[f"top_feature_{i+1}"] = ""
        
        rows.append(row)
    
    return pd.DataFrame(rows)
