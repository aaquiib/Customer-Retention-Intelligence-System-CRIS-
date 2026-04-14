"""Page 03: Batch Scoring - CSV upload and bulk predictions."""

import streamlit as st
import pandas as pd
import io
from utils.api_client import APIClient
from utils.validators import parse_csv_file
from utils.data_processors import aggregate_batch_summary, prepare_batch_result_df, get_top_customers_by_risk
from utils.chart_builders import create_segment_donut, create_action_donut
from config import SEGMENT_LABELS


def render():
    """Render the Batch Scoring page."""
    
    st.title("📁 Batch Customer Scoring")
    st.markdown("Upload a CSV file to score multiple customers at once.")
    
    api_client = st.session_state.api_client
    
    # ─────────────────────────────────────────────────────────────
    # CSV UPLOAD SECTION
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Upload CSV File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with 19 customer columns",
            type="csv",
            key="batch_uploader"
        )
    
    with col2:
        if st.button("📥 Download Template", key="download_template"):
            with st.spinner("Downloading template..."):
                success, template_bytes, error = api_client.get_batch_template()
                
                if success:
                    st.download_button(
                        label="Click to download",
                        data=template_bytes,
                        file_name="batch_template.csv",
                        mime="text/csv",
                        key="template_download"
                    )
                else:
                    st.error(f"❌ Download failed: {error}")
    
    if uploaded_file:
        # Parse CSV
        success, df, error = parse_csv_file(uploaded_file)
        
        if not success:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            with st.expander("Preview CSV Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            st.divider()
            
            # ─────────────────────────────────────────────────────────────
            # SCORE BATCH BUTTON
            # ─────────────────────────────────────────────────────────────
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                score_button = st.button(
                    "⚡ Score Batch",
                    key="score_batch_button",
                    use_container_width=True
                )
            
            if score_button:
                with st.spinner(f"Scoring {len(df)} customers..."):
                    # Convert DataFrame to CSV bytes
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    
                    # Call API
                    success, predictions, error = api_client.predict_batch(csv_bytes)
                    
                    if success:
                        # Store in session
                        st.session_state.batch_predictions = predictions
                        st.session_state.batch_data = df
                        
                        st.success(f"✅ Batch scoring complete!")
                        
                        # ─────────────────────────────────────────────────────────────
                        # BATCH SUMMARY
                        # ─────────────────────────────────────────────────────────────
                        
                        st.divider()
                        st.subheader("Batch Summary")
                        
                        batch_summary = aggregate_batch_summary(predictions)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Rows",
                                batch_summary["total_rows"]
                            )
                        
                        with col2:
                            st.metric(
                                "Rows Processed",
                                batch_summary["rows_processed"]
                            )
                        
                        with col3:
                            st.metric(
                                "Rows Failed",
                                batch_summary["rows_failed"]
                            )
                        
                        with col4:
                            st.metric(
                                "Churn Rate",
                                f"{batch_summary['churn_rate']:.2%}"
                            )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Avg Churn Probability",
                                f"{batch_summary['avg_churn_probability']:.4f}"
                            )
                        
                        with col2:
                            batch_df = prepare_batch_result_df(predictions)
                            # Safe revenue calculation (MonthlyCharges might be string)
                            churners = batch_df[batch_df["is_churner"] == True]
                            if len(churners) > 0 and "MonthlyCharges" in churners.columns:
                                try:
                                    revenue_at_risk = pd.to_numeric(churners["MonthlyCharges"], errors="coerce").sum()
                                except:
                                    revenue_at_risk = 0.0
                            else:
                                revenue_at_risk = 0.0
                            
                            st.metric(
                                "Total Revenue at Risk",
                                f"${revenue_at_risk:,.0f}"
                            )
                        
                        st.divider()
                        
                        # ─────────────────────────────────────────────────────────────
                        # SEGMENT & ACTION DISTRIBUTION
                        # ─────────────────────────────────────────────────────────────
                        
                        st.subheader("Distribution Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Segment distribution
                            fig = create_segment_donut(batch_summary["segment_distribution"])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Action distribution
                            if batch_summary["action_distribution"]:
                                fig = create_action_donut(batch_summary["action_distribution"])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No action distribution data.")
                        
                        st.divider()
                        
                        # ─────────────────────────────────────────────────────────────
                        # RESULTS TABLE
                        # ─────────────────────────────────────────────────────────────
                        
                        st.subheader("Detailed Results")
                        
                        batch_df = prepare_batch_result_df(predictions)
                        
                        # Filters
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            selected_segments = st.multiselect(
                                "Filter by Segment",
                                options=[0, 1, 2, 3],
                                format_func=lambda x: SEGMENT_LABELS[x],
                                default=[0, 1, 2, 3],
                                key="batch_segment_filter"
                            )
                        
                        with col2:
                            selected_risk = st.multiselect(
                                "Filter by Risk Band",
                                options=["Low", "Medium", "High"],
                                default=["Low", "Medium", "High"],
                                key="batch_risk_filter"
                            )
                        
                        with col3:
                            selected_churners = st.checkbox(
                                "Churners Only",
                                value=False,
                                key="batch_churners_filter"
                            )
                        
                        # Apply filters
                        filtered_df = batch_df.copy()
                        
                        if selected_segments:
                            filtered_df = filtered_df[filtered_df["segment"].isin(selected_segments)]
                        
                        # Risk band filter
                        if selected_risk:
                            risk_mask = pd.Series([False] * len(filtered_df))
                            if "Low" in selected_risk:
                                risk_mask |= filtered_df["churn_probability"] < 0.35
                            if "Medium" in selected_risk:
                                risk_mask |= (filtered_df["churn_probability"] >= 0.35) & (filtered_df["churn_probability"] < 0.65)
                            if "High" in selected_risk:
                                risk_mask |= filtered_df["churn_probability"] >= 0.65
                            filtered_df = filtered_df[risk_mask]
                        
                        if selected_churners:
                            filtered_df = filtered_df[filtered_df["is_churner"] == True]
                        
                        st.markdown(f"Showing {len(filtered_df)} of {len(batch_df)} customers")
                        
                        # Display table
                        display_cols = [
                            "customerID",
                            "segment_label",
                            "segment_confidence",
                            "churn_probability",
                            "is_churner",
                            "recommended_action",
                            "MonthlyCharges",
                            "tenure",
                            "Contract",
                            "top_feature_1",
                            "top_feature_2",
                            "top_feature_3"
                        ]
                        
                        # Only show columns that exist
                        display_cols = [c for c in display_cols if c in filtered_df.columns]
                        
                        st.dataframe(
                            filtered_df[display_cols].head(100),
                            use_container_width=True,
                            height=400
                        )
                        
                        st.divider()
                        
                        # ─────────────────────────────────────────────────────────────
                        # TOP RISK CUSTOMERS
                        # ─────────────────────────────────────────────────────────────
                        
                        with st.expander("🔴 Top 20 Highest-Risk Customers", expanded=False):
                            top_risk_df = get_top_customers_by_risk(batch_df, n=20)
                            st.dataframe(
                                top_risk_df[display_cols],
                                use_container_width=True
                            )
                        
                        st.divider()
                        
                        # ─────────────────────────────────────────────────────────────
                        # DOWNLOAD ENRICHED CSV
                        # ─────────────────────────────────────────────────────────────
                        
                        st.subheader("Download Results")
                        
                        # Prepare CSV for download
                        csv_buffer = io.StringIO()
                        batch_df.to_csv(csv_buffer, index=False)
                        csv_bytes = csv_buffer.getvalue().encode("utf-8")
                        
                        st.download_button(
                            label="📥 Download Enriched CSV",
                            data=csv_bytes,
                            file_name="batch_predictions_enriched.csv",
                            mime="text/csv",
                            key="download_enriched_csv"
                        )
                    
                    else:
                        st.error(f"❌ Batch scoring failed: {error}")
    
    else:
        st.info("📤 Upload a CSV file to get started.")
