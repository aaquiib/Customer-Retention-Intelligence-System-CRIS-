"""Page 04: Segment Intelligence - Per-segment analytics and distributions."""

import streamlit as st
import pandas as pd
from utils.data_processors import prepare_batch_result_df, build_segment_stats
from utils.chart_builders import (
    create_tenure_distribution,
    create_monthly_charges_distribution,
    create_service_adoption_bar,
    create_contract_pie,
    create_cdf_curve
)
from config import SEGMENT_LABELS, SEGMENT_COLORS


def render():
    """Render the Segment Intelligence page."""
    
    st.title("🔍 Segment Intelligence")
    st.markdown("Deep-dive analysis for each customer segment.")
    
    if not st.session_state.batch_predictions:
        st.info("📁 Upload a batch CSV to view segment analysis. Go to **Batch Scoring** page.")
        return
    
    batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
    
    # Check if batch data is empty
    if batch_df.empty:
        st.error("❌ No valid predictions in batch data. Unable to display segment analysis.")
        return
    
    # ─────────────────────────────────────────────────────────────
    # SEGMENT SELECTOR
    # ─────────────────────────────────────────────────────────────
    
    tabs = st.tabs([f"📊 {SEGMENT_LABELS[i]}" for i in range(4)])
    
    for seg_id, tab in enumerate(tabs):
        with tab:
            seg_df = batch_df[batch_df["segment"] == seg_id]
            
            if len(seg_df) == 0:
                st.info(f"No data for {SEGMENT_LABELS[seg_id]} segment.")
                continue
            
            # ─────────────────────────────────────────────────────────────
            # SEGMENT STATS CARDS
            # ─────────────────────────────────────────────────────────────
            
            stats = build_segment_stats(batch_df, seg_id)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Customer Count", stats["size"])
            
            with col2:
                st.metric("Churn Rate", f"{stats['churn_rate']:.2%}")
            
            with col3:
                st.metric("Avg Tenure", f"{stats['avg_tenure']:.1f} mo")
            
            with col4:
                st.metric("Avg Charges", f"${stats['avg_monthly_charges']:.2f}")
            
            with col5:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")
            
            st.divider()
            
            # ─────────────────────────────────────────────────────────────
            # DISTRIBUTIONS
            # ─────────────────────────────────────────────────────────────
            
            st.subheader("Customer Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Tenure distribution
                tenure_data = seg_df["tenure"].dropna().tolist()
                if tenure_data:
                    fig = create_tenure_distribution(tenure_data, SEGMENT_LABELS[seg_id])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly charges distribution
                charges_data = seg_df["MonthlyCharges"].dropna().tolist()
                if charges_data:
                    fig = create_monthly_charges_distribution(charges_data, SEGMENT_LABELS[seg_id])
                    st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # ─────────────────────────────────────────────────────────────
            # SERVICE & CONTRACT ANALYSIS
            # ─────────────────────────────────────────────────────────────
            
            st.subheader("Service & Contract Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Service adoption
                services = [
                    "PhoneService", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"
                ]
                
                service_adoption = {}
                for service in services:
                    if service in seg_df.columns:
                        count = (seg_df[service] == "Yes").sum()
                        service_adoption[service] = count
                
                if service_adoption:
                    fig = create_service_adoption_bar(service_adoption, SEGMENT_LABELS[seg_id])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Contract type breakdown
                if "Contract" in seg_df.columns:
                    contracts = seg_df["Contract"].value_counts().to_dict()
                    fig = create_contract_pie(contracts, SEGMENT_LABELS[seg_id])
                    st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # ─────────────────────────────────────────────────────────────
            # CHURN PROBABILITY CDF
            # ─────────────────────────────────────────────────────────────
            
            st.subheader("Churn Probability Distribution")
            
            churn_data = seg_df["churn_probability"].dropna().tolist()
            if churn_data:
                fig = create_cdf_curve(churn_data, f"CDF - {SEGMENT_LABELS[seg_id]}")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # ─────────────────────────────────────────────────────────────
            # DETAILED TABLE
            # ─────────────────────────────────────────────────────────────
            
            st.subheader("Detailed Customer List")
            
            display_cols = [
                "customerID",
                "segment_confidence",
                "churn_probability",
                "is_churner",
                "recommended_action",
                "MonthlyCharges",
                "tenure",
                "Contract",
                "InternetService"
            ]
            
            display_cols = [c for c in display_cols if c in seg_df.columns]
            
            st.dataframe(
                seg_df[display_cols],
                use_container_width=True,
                height=400
            )
