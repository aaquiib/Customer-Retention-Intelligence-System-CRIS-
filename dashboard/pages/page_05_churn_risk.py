"""Page 05: Churn Risk Analysis - Risk distributions and revenue at risk."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_processors import prepare_batch_result_df, aggregate_batch_summary
from utils.chart_builders import (
    create_distribution_histogram,
    create_bar_chart,
    create_cdf_curve
)
from config import CHURN_THRESHOLD, SEGMENT_LABELS, RISK_COLORS


def render():
    """Render the Churn Risk Analysis page."""
    
    st.title("⚠️ Churn Risk Analysis")
    st.markdown("Comprehensive risk distribution and revenue impact analysis.")
    
    if not st.session_state.batch_predictions:
        st.info("📁 Upload a batch CSV to view risk analysis. Go to **Batch Scoring** page.")
        return
    
    batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
    batch_summary = aggregate_batch_summary(st.session_state.batch_predictions)
    
    # Check if batch data is empty after processing
    if batch_df.empty:
        st.error("❌ No valid predictions in batch data. All predictions may have failed.")
        return
    
    # ─────────────────────────────────────────────────────────────
    # RISK BAND SUMMARY
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Risk Band Summary")
    
    high_risk = len(batch_df[batch_df["churn_probability"] > 0.65])
    medium_risk = len(batch_df[(batch_df["churn_probability"] >= 0.35) & (batch_df["churn_probability"] < 0.65)])
    low_risk = len(batch_df[batch_df["churn_probability"] < 0.35])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔴 High Risk (>0.65)", high_risk)
    
    with col2:
        st.metric("🟡 Medium Risk (0.35-0.65)", medium_risk)
    
    with col3:
        st.metric("🟢 Low Risk (<0.35)", low_risk)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # AVERAGE PRIORITY BY RISK BAND
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Average Priority Score by Risk Band")
    
    # Extract priority scores
    priority_scores = []
    for pred in st.session_state.batch_predictions:
        action_info = pred.get("recommended_action", {})
        if isinstance(action_info, dict):
            priority_score = action_info.get("priority_score", 0.5)
        else:
            priority_score = 0.5
        priority_scores.append(priority_score)
    
    batch_df_priority = batch_df.copy()
    batch_df_priority["priority_score"] = priority_scores
    batch_df_priority["priority_percent"] = (batch_df_priority["priority_score"] * 100).round(0)
    
    # Add risk band
    batch_df_priority["risk_band"] = batch_df_priority["churn_probability"].apply(
        lambda x: "Low" if x < 0.35 else ("Medium" if x < 0.65 else "High")
    )
    
    # Calculate average priority per risk band
    risk_bands = ["Low", "Medium", "High"]
    avg_priority = []
    customer_counts = []
    
    for band in risk_bands:
        band_data = batch_df_priority[batch_df_priority["risk_band"] == band]
        if len(band_data) > 0:
            avg_priority.append(band_data["priority_percent"].mean())
            customer_counts.append(len(band_data))
        else:
            avg_priority.append(0)
            customer_counts.append(0)
    
    fig_priority_by_risk = go.Figure()
    
    fig_priority_by_risk.add_trace(go.Bar(
        x=risk_bands,
        y=avg_priority,
        marker=dict(
            color=["#2ecc71", "#f39c12", "#e74c3c"],  # Green, Orange, Red
        ),
        text=[f"{p:.0f}%" for p in avg_priority],
        textposition="auto",
        hovertemplate="<b>%{x} Risk</b><br>Avg Priority: %{y:.1f}%<extra></extra>"
    ))
    
    fig_priority_by_risk.update_layout(
        title="Average Priority Score by Risk Band",
        xaxis_title="Risk Band",
        yaxis_title="Average Priority Score (%)",
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig_priority_by_risk, use_container_width=True)
    
    # Summary table
    priority_summary = pd.DataFrame({
        "Risk Band": risk_bands,
        "Avg Priority %": [f"{p:.0f}%" for p in avg_priority],
        "Customer Count": customer_counts,
        "% of Total": [f"{(c/len(batch_df_priority)*100):.1f}%" if len(batch_df_priority) > 0 else "0%" for c in customer_counts]
    })
    
    st.dataframe(priority_summary, use_container_width=True)
    
    st.divider()
    
    st.subheader("Churn Probability Distribution")
    
    churn_prob_data = batch_df["churn_probability"].dropna().tolist()
    
    fig = create_distribution_histogram(
        churn_prob_data,
        "Churn Probability Distribution",
        nbins=30,
        threshold=CHURN_THRESHOLD,
        xaxis_label="Churn Probability"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # REVENUE AT RISK BY SEGMENT
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Revenue at Risk by Segment")
    
    revenue_by_segment = {}
    for seg_id in range(4):
        seg_df = batch_df[(batch_df["segment"] == seg_id) & (batch_df["is_churner"] == True)]
        if len(seg_df) > 0:
            revenue = seg_df["MonthlyCharges"].sum()
        else:
            revenue = 0
        revenue_by_segment[SEGMENT_LABELS[seg_id]] = revenue
    
    fig = create_bar_chart(
        list(revenue_by_segment.keys()),
        list(revenue_by_segment.values()),
        "Revenue at Risk by Segment",
        color="#e74c3c"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # CHURN PROBABILITY CDF PER SEGMENT
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Churn Probability CDF by Segment")
    
    segment_data = {}
    for seg_id in range(4):
        seg_df = batch_df[batch_df["segment"] == seg_id]
        segment_data[seg_id] = seg_df["churn_probability"].dropna().tolist()
    
    fig = create_cdf_curve(
        [],
        "Churn Probability CDF by Segment",
        per_segment=True,
        segment_data=segment_data
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # ACTION DISTRIBUTION
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Recommended Actions")
    
    action_dist = batch_summary["action_distribution"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Action table
        action_df = pd.DataFrame(
            list(action_dist.items()),
            columns=["Action", "Count"]
        )
        action_df = action_df.sort_values("Count", ascending=False)
        st.dataframe(action_df, use_container_width=True)
    
    with col2:
        # Action bar chart
        fig = create_bar_chart(
            list(action_dist.keys()),
            list(action_dist.values()),
            "Action Distribution",
            color="#3498db"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # DETAILED RISK METRICS TABLE
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Segment Risk Breakdown")
    
    risk_metrics = []
    for seg_id in range(4):
        seg_df = batch_df[batch_df["segment"] == seg_id]
        if len(seg_df) > 0:
            churners = seg_df[seg_df["is_churner"] == True]
            risk_metrics.append({
                "Segment": SEGMENT_LABELS[seg_id],
                "Customer Count": len(seg_df),
                "Churn Rate": f"{(len(churners) / len(seg_df)):.2%}",
                "Revenue at Risk": f"${churners['MonthlyCharges'].sum():,.0f}",
                "Avg Churn Prob": f"{seg_df['churn_probability'].mean():.4f}",
                "Churner Count": len(churners)
            })
    
    risk_metrics_df = pd.DataFrame(risk_metrics)
    st.dataframe(risk_metrics_df, use_container_width=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # KEY INSIGHTS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_revenue_at_risk = batch_df[batch_df["is_churner"]]["MonthlyCharges"].sum()
        st.metric(
            "Total Monthly Revenue at Risk",
            f"${total_revenue_at_risk:,.0f}"
        )
    
    with col2:
        avg_churn_prob = batch_df["churn_probability"].mean()
        st.metric(
            "Average Churn Probability",
            f"{avg_churn_prob:.4f}"
        )
    
    with col3:
        overall_churn_rate = (batch_df["is_churner"].sum() / len(batch_df))
        st.metric(
            "Overall Churn Rate",
            f"{overall_churn_rate:.2%}"
        )
