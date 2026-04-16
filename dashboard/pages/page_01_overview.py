"""Page 01: Overview - KPI Cards and High-Level Metrics."""

import streamlit as st
import pandas as pd
import numpy as np
from utils.api_client import APIClient
from utils.chart_builders import (
    create_segment_donut,
    create_heatmap,
    create_action_donut
)
from utils.data_processors import aggregate_batch_summary, build_segment_stats, prepare_batch_result_df
from config import SEGMENT_LABELS, SEGMENT_COLORS, CHURN_THRESHOLD


def render():
    """Render the Overview page."""
    
    st.title("📊 Customer Retention Overview")
    st.markdown("High-level KPIs and segment intelligence for the churn decision system.")
    
    api_client = st.session_state.api_client
    
    # ─────────────────────────────────────────────────────────────
    # KPI CARDS (Top Section)
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Key Performance Indicators")
    
    # Check if batch data available
    if st.session_state.batch_predictions:
        batch_summary = aggregate_batch_summary(st.session_state.batch_predictions)
        batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
        
        # Extract priority scores
        priority_scores = []
        for pred in st.session_state.batch_predictions:
            action_info = pred.get("recommended_action", {})
            if isinstance(action_info, dict):
                priority_score = action_info.get("priority_score", 0.5)
            else:
                priority_score = 0.5
            priority_scores.append(priority_score)
        
        batch_df["priority_score"] = priority_scores
        batch_df["priority_percent"] = (batch_df["priority_score"] * 100).round(0).astype(int)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Total Customers",
                batch_summary["total_rows"],
                delta=f"{batch_summary['rows_processed']} processed"
            )
        
        with col2:
            st.metric(
                "Overall Churn Rate",
                f"{batch_summary['churn_rate']:.2%}",
                delta=f"{batch_summary['segment_distribution'][1]} in Low Engagement"
            )
        
        with col3:
            st.metric(
                "Avg Churn Probability",
                f"{batch_summary['avg_churn_probability']:.4f}",
                delta="Based on model"
            )
        
        with col4:
            # Revenue at risk
            revenue_at_risk = batch_df[batch_df["is_churner"]]["MonthlyCharges"].sum()
            st.metric(
                "Revenue at Risk",
                f"${revenue_at_risk:,.0f}",
                delta="Monthly recurring"
            )
        
        with col5:
            # Model AUC
            success, model_info, _ = api_client.get_model_info()
            if success:
                auc = model_info.get("churn_model", {}).get("performance_metrics", {}).get("roc_auc", 0)
                st.metric(
                    "Model AUC",
                    f"{auc:.4f}",
                    delta="Trust Badge ✅" if auc > 0.8 else "Review needed"
                )
            else:
                st.metric("Model AUC", "N/A", delta="Unable to fetch")
        
        with col6:
            # API health
            success, _, _ = api_client.get_health()
            status = "🟢 Running" if success else "🔴 Offline"
            st.metric(
                "API Status",
                status,
                delta="All systems operational" if success else "Check connection"
            )
    
    else:
        # No batch data loaded
        st.info("📁 Upload a batch CSV to view KPI summary. Go to **Batch Scoring** page.")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Total Customers", "—")
        with col2:
            st.metric("Churn Rate", "—")
        with col3:
            st.metric("Avg Churn Prob", "—")
        with col4:
            st.metric("Revenue at Risk", "—")
        with col5:
            success, model_info, _ = api_client.get_model_info()
            if success:
                auc = model_info.get("churn_model", {}).get("performance_metrics", {}).get("roc_auc", 0)
                st.metric("Model AUC", f"{auc:.4f}")
            else:
                st.metric("Model AUC", "—")
        with col6:
            success, _, _ = api_client.get_health()
            status = "🟢 Running" if success else "🔴 Offline"
            st.metric("API Status", status)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # HIGHEST PRIORITY CUSTOMER
    # ─────────────────────────────────────────────────────────────
    
    if st.session_state.batch_predictions and len(batch_df) > 0 and "priority_score" in batch_df.columns:
        highest_priority_idx = batch_df["priority_score"].idxmax()
        highest_priority = batch_df.loc[highest_priority_idx]
        
        st.subheader("🎯 Highest Priority Customer")
        
        priority_col1, priority_col2, priority_col3, priority_col4 = st.columns(4)
        
        with priority_col1:
            priority_percent = int(highest_priority["priority_percent"])
            if priority_percent >= 67:
                color = "#e74c3c"  # Red
                level = "🔴 High"
            elif priority_percent >= 34:
                color = "#f39c12"  # Orange
                level = "🟡 Medium"
            else:
                color = "#2ecc71"  # Green
                level = "🟢 Low"
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 8px; text-align: center;">
                <p style="color: white; font-size: 12px; margin: 0;">PRIORITY LEVEL</p>
                <p style="color: white; font-size: 24px; font-weight: bold; margin: 10px 0;">{priority_percent}%</p>
                <p style="color: white; font-size: 13px; margin: 0;">{level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with priority_col2:
            st.metric(
                "Customer ID",
                int(highest_priority.get("customerID", 0))
            )
        
        with priority_col3:
            st.metric(
                "Segment",
                highest_priority.get("segment_label", "Unknown"),
                delta=f"Churn: {highest_priority.get('churn_probability', 0):.2%}"
            )
        
        with priority_col4:
            monthly = highest_priority.get("MonthlyCharges", 0)
            st.metric(
                "Monthly Value",
                f"${monthly:,.0f}",
                delta=f"Tenure: {int(highest_priority.get('tenure', 0))} mo"
            )
        
        st.divider()
    
    st.subheader("Segment Intelligence")
    
    if st.session_state.batch_predictions:
        batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
        
        # Segment stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        for seg_id, col in zip(range(4), [col1, col2, col3, col4]):
            seg_df = batch_df[batch_df["segment"] == seg_id]
            
            if len(seg_df) > 0:
                churn_rate = seg_df["is_churner"].sum() / len(seg_df)
                avg_tenure = seg_df["tenure"].mean()
                avg_charges = seg_df["MonthlyCharges"].mean()
                
                with col:
                    st.markdown(f"""
                    **{SEGMENT_LABELS[seg_id]}**
                    
                    - Count: {len(seg_df)}
                    - Churn Rate: {churn_rate:.2%}
                    - Avg Tenure: {avg_tenure:.1f} mo
                    - Avg Charges: ${avg_charges:.2f}
                    """)
            else:
                with col:
                    st.markdown(f"""
                    **{SEGMENT_LABELS[seg_id]}**
                    
                    - Count: 0
                    - No data
                    """)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution donut
            batch_summary = aggregate_batch_summary(st.session_state.batch_predictions)
            fig = create_segment_donut(batch_summary["segment_distribution"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn rate by segment heatmap
            segment_churn_rates = {}
            for seg_id in range(4):
                seg_df = batch_df[batch_df["segment"] == seg_id]
                if len(seg_df) > 0:
                    segment_churn_rates[seg_id] = seg_df["is_churner"].sum() / len(seg_df)
                else:
                    segment_churn_rates[seg_id] = 0.0
            
            fig = create_heatmap(list(range(4)), segment_churn_rates)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Upload a batch CSV to see segment intelligence. Go to **Batch Scoring** page.")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # ACTION DISTRIBUTION (Bottom Section)
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Action Distribution")
    
    if st.session_state.batch_predictions:
        batch_summary = aggregate_batch_summary(st.session_state.batch_predictions)
        
        if batch_summary["action_distribution"]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Action counts table
                action_df = pd.DataFrame(
                    list(batch_summary["action_distribution"].items()),
                    columns=["Action", "Count"]
                )
                st.dataframe(action_df, use_container_width=True)
            
            with col2:
                # Action donut chart
                fig = create_action_donut(batch_summary["action_distribution"])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No action distribution data available.")
    
    else:
        st.info("Upload a batch CSV to see action distribution. Go to **Batch Scoring** page.")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # QUICK STATS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Quick Stats")
    
    if st.session_state.batch_predictions:
        batch_summary = aggregate_batch_summary(st.session_state.batch_predictions)
        batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk = len(batch_df[batch_df["churn_probability"] > 0.65])
            st.metric("High Risk Customers", high_risk)
        
        with col2:
            medium_risk = len(batch_df[(batch_df["churn_probability"] >= 0.35) & (batch_df["churn_probability"] <= 0.65)])
            st.metric("Medium Risk Customers", medium_risk)
        
        with col3:
            low_risk = len(batch_df[batch_df["churn_probability"] < 0.35])
            st.metric("Low Risk Customers", low_risk)
        
        with col4:
            failed_rows = batch_summary["rows_failed"]
            st.metric("Failed Predictions", failed_rows)
    
    else:
        st.info("No data loaded yet.")
