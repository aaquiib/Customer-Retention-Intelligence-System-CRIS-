"""Reusable Plotly chart builders."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import SEGMENT_LABELS, SEGMENT_COLORS, RISK_COLORS, RISK_BANDS, CHURN_THRESHOLD

logger = logging.getLogger(__name__)


def create_kpi_card(label: str, value: Any, unit: str = "", precision: int = 2) -> str:
    """Create a KPI metric display using Streamlit markdown.
    
    Args:
        label: Metric label
        value: Metric value
        unit: Unit suffix (%, $, etc.)
        precision: Decimal precision
        
    Returns:
        HTML-formatted card string
    """
    if isinstance(value, float):
        value = f"{value:.{precision}f}"
    
    return f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 8px; text-align: center;">
        <p style="font-size: 14px; color: #666; margin: 0;">{label}</p>
        <p style="font-size: 28px; font-weight: bold; color: #333; margin: 5px 0;">{value}{unit}</p>
    </div>
    """


def create_gauge_chart(
    value: float,
    label: str,
    threshold: float = None,
    color: str = None,
    domain: Tuple[float, float] = (0, 1)
) -> go.Figure:
    """Create a gauge chart for churn probability.
    
    Args:
        value: Current value
        label: Chart title
        threshold: Threshold line position
        color: Bar color
        domain: Min/max domain values
        
    Returns:
        Plotly figure
    """
    if color is None:
        if value < 0.35:
            color = "#2ecc71"  # Green
        elif value < 0.65:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": label},
        domain={"x": domain, "y": domain},
        gauge={
            "axis": {"range": domain},
            "bar": {"color": color},
            "steps": [
                {"range": [domain[0], 0.35], "color": "rgba(46, 204, 113, 0.1)"},
                {"range": [0.35, 0.65], "color": "rgba(243, 156, 18, 0.1)"},
                {"range": [0.65, domain[1]], "color": "rgba(231, 76, 60, 0.1)"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 2},
                "thickness": 0.75,
                "value": threshold if threshold else 0.4356
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def create_distribution_histogram(
    data: list,
    title: str,
    nbins: int = 30,
    threshold: float = None,
    xaxis_label: str = "Value"
) -> go.Figure:
    """Create distribution histogram with optional threshold line.
    
    Args:
        data: List of values
        title: Chart title
        nbins: Number of bins
        threshold: Optional threshold line
        xaxis_label: X-axis label
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Handle empty data
    if not data or len(data) == 0:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_label,
            yaxis_title="Count",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        return fig
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=nbins,
        marker=dict(color="#3498db"),
        opacity=0.7
    ))
    
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.4f}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_label,
        yaxis_title="Count",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_segment_donut(segment_dist: Dict[int, int]) -> go.Figure:
    """Create donut chart for segment distribution.
    
    Args:
        segment_dist: Dict of segment_id -> count
        
    Returns:
        Plotly figure
    """
    labels = [SEGMENT_LABELS[seg_id] for seg_id in sorted(segment_dist.keys())]
    values = [segment_dist[seg_id] for seg_id in sorted(segment_dist.keys())]
    colors = [SEGMENT_COLORS[seg_id] for seg_id in sorted(segment_dist.keys())]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textposition="auto",
        hoverinfo="label+value+percent"
    )])
    
    fig.update_layout(
        title="Segment Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def create_action_donut(action_dist: Dict[str, int]) -> go.Figure:
    """Create donut chart for action distribution.
    
    Args:
        action_dist: Dict of action_name -> count
        
    Returns:
        Plotly figure
    """
    if not action_dist:
        return go.Figure()
    
    labels = list(action_dist.keys())
    values = list(action_dist.values())
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textposition="auto",
        hoverinfo="label+value+percent"
    )])
    
    fig.update_layout(
        title="Action Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def create_heatmap(segments: List[int], churn_rates: Dict[int, float]) -> go.Figure:
    """Create segment x churn rate heatmap.
    
    Args:
        segments: List of segment IDs
        churn_rates: Dict of segment_id -> churn_rate
        
    Returns:
        Plotly figure
    """
    segment_names = [SEGMENT_LABELS[seg] for seg in segments]
    values = [[churn_rates.get(seg, 0)] for seg in segments]
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        y=segment_names,
        colorscale="RdYlGn_r",
        text=[[f"{churn_rates.get(seg, 0):.2%}"] for seg in segments],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Churn Rate")
    ))
    
    fig.update_layout(
        title="Churn Rate by Segment",
        yaxis_title="Segment",
        height=300,
        xaxis=dict(showticklabels=False)
    )
    
    return fig


def create_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    color: str = "#3498db",
    orientation: str = "v",
    color_map: Dict[str, str] = None
) -> go.Figure:
    """Create generic bar chart.
    
    Args:
        categories: Category labels
        values: Values per category
        title: Chart title
        color: Bar color (if not using color_map)
        orientation: 'v' for vertical, 'h' for horizontal
        color_map: Dict of category -> color for per-category coloring
        
    Returns:
        Plotly figure
    """
    colors = [color_map.get(cat, color) for cat in categories] if color_map else color
    
    if orientation == "h":
        fig = go.Figure([go.Bar(
            y=categories,
            x=values,
            marker=dict(color=colors),
            orientation="h",
            text=values,
            textposition="auto"
        )])
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Category",
            height=400
        )
    else:
        fig = go.Figure([go.Bar(
            x=categories,
            y=values,
            marker=dict(color=colors),
            text=values,
            textposition="auto"
        )])
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Value",
            height=400
        )
    
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def create_waterfall_chart(
    features: List[str],
    shap_values: List[float],
    base_value: float,
    prediction_value: float
) -> go.Figure:
    """Create SHAP waterfall chart.
    
    Args:
        features: Feature names
        shap_values: SHAP values per feature
        base_value: Base prediction (intercept)
        prediction_value: Final prediction value
        
    Returns:
        Plotly figure
    """
    # Build waterfall measures
    x_vals = ["Base Value"] + features + ["Final Prediction"]
    y_vals = [base_value] + shap_values
    
    # Calculate cumulative for waterfall
    cumulative = base_value
    measure_list = ["absolute"]
    
    for i, shap_val in enumerate(shap_values):
        cumulative += shap_val
        measure_list.append("relative")
    
    measure_list.append("absolute")
    
    fig = go.Figure(go.Waterfall(
        x=x_vals,
        measure=measure_list,
        y=y_vals + [prediction_value],
        text=[f"{v:.4f}" for v in y_vals] + [f"{prediction_value:.4f}"],
        textposition="auto",
        increasing={"marker": {"color": "#2ecc71"}},  # Green for positive contributions (push toward churn)
        decreasing={"marker": {"color": "#e74c3c"}},  # Red for negative contributions (protect from churn)
        totals={"marker": {"color": "#3498db"}},
        connector={"line": {"color": "rgba(0,0,0,0.4)"}}
    ))
    
    fig.update_layout(
        title="SHAP Force Plot (Waterfall)",
        yaxis_title="Model Output",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_feature_importance_bar(
    features: List[str],
    importances: List[float],
    signs: List[str] = None,
    top_n: int = 10
) -> go.Figure:
    """Create feature importance bar chart with color coding.
    
    Args:
        features: Feature names
        importances: Importance values (absolute SHAP)
        signs: List of "positive" or "negative" for coloring
        top_n: Limit to top N features
        
    Returns:
        Plotly figure
    """
    # Limit to top N
    features = features[:top_n]
    importances = importances[:top_n]
    
    if signs:
        signs = signs[:top_n]
        colors = ["#e74c3c" if s == "positive" else "#2ecc71" for s in signs]
    else:
        colors = "#3498db"
    
    fig = go.Figure([go.Bar(
        y=features,
        x=importances,
        marker=dict(color=colors),
        orientation="h",
        text=[f"{v:.4f}" for v in importances],
        textposition="auto"
    )])
    
    fig.update_layout(
        title="Global Feature Importance (SHAP)",
        xaxis_title="Average |SHAP value|",
        yaxis_title="Feature",
        height=max(300, 20 * len(features)),
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def create_cdf_curve(
    data: List[float],
    title: str,
    per_segment: bool = False,
    segment_data: Dict[int, List[float]] = None
) -> go.Figure:
    """Create CDF curve for churn probability.
    
    Args:
        data: Values for CDF
        title: Chart title
        per_segment: If True, plot per segment
        segment_data: Dict of segment_id -> list of values
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if per_segment and segment_data:
        # Filter out empty segments
        segments_with_data = {seg_id: vals for seg_id, vals in segment_data.items() if vals and len(vals) > 0}
        
        if not segments_with_data:
            fig.add_annotation(
                text="No data available for any segment",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            for seg_id in sorted(segments_with_data.keys()):
                values = sorted(segments_with_data[seg_id])
                cdf = np.arange(1, len(values) + 1) / len(values)
                fig.add_trace(go.Scatter(
                    x=values,
                    y=cdf,
                    mode="lines",
                    name=SEGMENT_LABELS.get(seg_id, f"Segment {seg_id}"),
                    line=dict(color=SEGMENT_COLORS.get(seg_id, "#3498db"), width=2)
                ))
    else:
        values = sorted(data) if data else []
        if not values:
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        else:
            cdf = np.arange(1, len(values) + 1) / len(values)
            fig.add_trace(go.Scatter(
                x=values,
                y=cdf,
                mode="lines",
                name="CDF",
                line=dict(color="#3498db", width=2),
                fill="tozeroy"
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Churn Probability",
        yaxis_title="Cumulative Probability",
        height=400,
        template="plotly_white"
    )
    
    return fig


def create_contract_pie(contracts: Dict[str, int], segment_name: str = "") -> go.Figure:
    """Create pie chart for contract type distribution.
    
    Args:
        contracts: Dict of contract_type -> count
        segment_name: Optional segment name for title
        
    Returns:
        Plotly figure
    """
    if not contracts:
        return go.Figure()
    
    fig = go.Figure(data=[go.Pie(
        labels=list(contracts.keys()),
        values=list(contracts.values()),
        hole=0.3,
        textposition="auto",
        hoverinfo="label+value+percent"
    )])
    
    title = f"Contract Types{' - ' + segment_name if segment_name else ''}"
    fig.update_layout(
        title=title,
        height=350,
        showlegend=True
    )
    
    return fig


def create_service_adoption_bar(services: Dict[str, int], segment_name: str = "") -> go.Figure:
    """Create stacked/grouped bar for service adoption.
    
    Args:
        services: Dict of service_name -> adoption_count
        segment_name: Optional segment name for title
        
    Returns:
        Plotly figure
    """
    if not services:
        return go.Figure()
    
    fig = go.Figure([go.Bar(
        x=list(services.keys()),
        y=list(services.values()),
        marker=dict(color="#3498db"),
        text=list(services.values()),
        textposition="auto"
    )])
    
    title = f"Service Adoption{' - ' + segment_name if segment_name else ''}"
    fig.update_layout(
        title=title,
        xaxis_title="Service",
        yaxis_title="Customers",
        height=350,
        template="plotly_white",
        xaxis_tickangle=-45
    )
    
    return fig


def create_tenure_distribution(data: List[float], segment_name: str = "") -> go.Figure:
    """Create histogram for tenure distribution.
    
    Args:
        data: List of tenure values
        segment_name: Optional segment name
        
    Returns:
        Plotly figure
    """
    fig = go.Figure([go.Histogram(
        x=data,
        nbinsx=15,
        marker=dict(color="#2ecc71"),
        opacity=0.7
    )])
    
    title = f"Tenure Distribution{' - ' + segment_name if segment_name else ''}"
    fig.update_layout(
        title=title,
        xaxis_title="Tenure (months)",
        yaxis_title="Count",
        height=350,
        template="plotly_white"
    )
    
    return fig


def create_monthly_charges_distribution(data: List[float], segment_name: str = "") -> go.Figure:
    """Create histogram for monthly charges distribution.
    
    Args:
        data: List of monthly charge values
        segment_name: Optional segment name
        
    Returns:
        Plotly figure
    """
    fig = go.Figure([go.Histogram(
        x=data,
        nbinsx=15,
        marker=dict(color="#f39c12"),
        opacity=0.7
    )])
    
    title = f"Monthly Charges Distribution{' - ' + segment_name if segment_name else ''}"
    fig.update_layout(
        title=title,
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Count",
        height=350,
        template="plotly_white"
    )
    
    return fig
