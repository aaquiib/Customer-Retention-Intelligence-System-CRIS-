"""Page 06: Action Planning - Filterable customer table with targeted actions."""

import streamlit as st
import pandas as pd
import io
from utils.data_processors import prepare_batch_result_df, get_top_customers_by_risk
from config import SEGMENT_LABELS


def render():
    """Render the Action Planning page."""
    
    st.title("📋 Action Planning & Customer Lookup")
    st.markdown("Filter and analyze customers by risk, segment, and recommended actions.")
    
    if not st.session_state.batch_predictions:
        st.info("📁 Upload a batch CSV to view action planning. Go to **Batch Scoring** page.")
        return
    
    batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
    
    # ─────────────────────────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_segments = st.multiselect(
            "Segments",
            options=[0, 1, 2, 3],
            format_func=lambda x: SEGMENT_LABELS[x],
            default=[0, 1, 2, 3],
            key="action_segment_filter"
        )
    
    with col2:
        unique_actions = batch_df["recommended_action"].unique()
        selected_actions = st.multiselect(
            "Actions",
            options=unique_actions,
            default=list(unique_actions),
            key="action_type_filter"
        )
    
    with col3:
        selected_risk = st.multiselect(
            "Risk Bands",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="action_risk_filter"
        )
    
    with col4:
        unique_contracts = batch_df["Contract"].unique()
        selected_contracts = st.multiselect(
            "Contract Types",
            options=unique_contracts,
            default=list(unique_contracts),
            key="action_contract_filter"
        )
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # APPLY FILTERS
    # ─────────────────────────────────────────────────────────────
    
    filtered_df = batch_df.copy()
    
    # Segment filter
    if selected_segments:
        filtered_df = filtered_df[filtered_df["segment"].isin(selected_segments)]
    
    # Action filter
    if selected_actions:
        filtered_df = filtered_df[filtered_df["recommended_action"].isin(selected_actions)]
    
    # Risk band filter
    risk_mask = pd.Series([False] * len(filtered_df))
    if "Low" in selected_risk:
        risk_mask |= filtered_df["churn_probability"] < 0.35
    if "Medium" in selected_risk:
        risk_mask |= (filtered_df["churn_probability"] >= 0.35) & (filtered_df["churn_probability"] < 0.65)
    if "High" in selected_risk:
        risk_mask |= filtered_df["churn_probability"] >= 0.65
    filtered_df = filtered_df[risk_mask]
    
    # Contract filter
    if selected_contracts:
        filtered_df = filtered_df[filtered_df["Contract"].isin(selected_contracts)]
    
    # ─────────────────────────────────────────────────────────────
    # MAIN TABLE
    # ─────────────────────────────────────────────────────────────
    
    st.subheader(f"Customer List ({len(filtered_df)} of {len(batch_df)})")
    
    # Sort options
    col1, col2 = st.columns([2, 2])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=[
                "churn_probability",
                "MonthlyCharges",
                "tenure",
                "segment_confidence"
            ],
            format_func=lambda x: {
                "churn_probability": "Churn Probability (High to Low)",
                "MonthlyCharges": "Monthly Charges (High to Low)",
                "tenure": "Tenure (Low to High)",
                "segment_confidence": "Segment Confidence (High to Low)"
            }[x],
            key="action_sort"
        )
    
    with col2:
        search_id = st.text_input(
            "Search by Customer ID",
            key="action_search"
        )
    
    # Apply sort
    if sort_by == "churn_probability":
        filtered_df = filtered_df.sort_values("churn_probability", ascending=False)
    elif sort_by == "MonthlyCharges":
        filtered_df = filtered_df.sort_values("MonthlyCharges", ascending=False)
    elif sort_by == "tenure":
        filtered_df = filtered_df.sort_values("tenure", ascending=True)
    elif sort_by == "segment_confidence":
        filtered_df = filtered_df.sort_values("segment_confidence", ascending=False)
    
    # Apply search
    if search_id:
        try:
            search_id = int(search_id)
            filtered_df = filtered_df[filtered_df["customerID"] == search_id]
        except:
            st.warning("Invalid customer ID format")
    
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
    
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        height=500
    )
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # TOP 20 HIGHEST-RISK CUSTOMERS
    # ─────────────────────────────────────────────────────────────
    
    with st.expander("🔴 Top 20 Highest-Risk Customers", expanded=False):
        top_risk_df = get_top_customers_by_risk(batch_df, n=20)
        st.dataframe(
            top_risk_df[display_cols],
            use_container_width=True
        )
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # CUSTOMER LOOKUP MODAL
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Customer Lookup")
    
    lookup_id = st.number_input(
        "Enter Customer ID",
        min_value=0,
        step=1,
        key="customer_lookup_id"
    )
    
    if st.button("🔍 Lookup Customer", key="customer_lookup_button"):
        customer_row = batch_df[batch_df["customerID"] == lookup_id]
        
        if len(customer_row) > 0:
            customer = customer_row.iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Segment badge
                segment_id = int(customer["segment"])
                segment_color = {
                    0: "#2ecc71",
                    1: "#e74c3c",
                    2: "#f39c12",
                    3: "#3498db"
                }.get(segment_id, "#95a5a6")
                
                st.markdown(f"""
                <div style="background-color: {segment_color}; padding: 15px; border-radius: 8px; text-align: center;">
                    <p style="color: white; font-size: 14px; margin: 0;">SEGMENT</p>
                    <p style="color: white; font-size: 20px; font-weight: bold; margin: 5px 0;">{customer.get('segment_label', 'Unknown')}</p>
                    <p style="color: white; font-size: 12px; margin: 0;">Conf: {customer.get('segment_confidence', 0):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Key metrics
                st.markdown(f"""
                **Customer Overview**
                
                - **Churn Probability**: {customer.get('churn_probability', 0):.4f}
                - **Is Churner**: {'✅ Yes' if customer.get('is_churner') else '❌ No'}
                - **Recommended Action**: {customer.get('recommended_action', 'N/A')}
                - **Monthly Charges**: ${customer.get('MonthlyCharges', 0):.2f}
                - **Tenure**: {customer.get('tenure', 0):.0f} months
                - **Contract**: {customer.get('Contract', 'N/A')}
                """)
            
            st.divider()
            
            # Full profile
            st.markdown("**Full Customer Profile**")
            
            profile_data = {
                "Field": [],
                "Value": []
            }
            
            for col in customer.index:
                if col != "customerID":
                    profile_data["Field"].append(col)
                    profile_data["Value"].append(str(customer[col]))
            
            profile_df = pd.DataFrame(profile_data)
            st.dataframe(profile_df, use_container_width=True)
        
        else:
            st.warning(f"❌ Customer ID {lookup_id} not found")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # DOWNLOAD FILTERED RESULTS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Download Results")
    
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    
    st.download_button(
        label="📥 Download Filtered Results (CSV)",
        data=csv_bytes,
        file_name="action_plan_filtered.csv",
        mime="text/csv",
        key="download_action_plan"
    )
