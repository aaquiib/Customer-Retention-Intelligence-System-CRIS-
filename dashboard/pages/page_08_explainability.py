"""Page 08: SHAP Explainability - Global and instance-level feature importance."""

import streamlit as st
import pandas as pd
from utils.api_client import APIClient
from utils.validators import convert_numpy_to_python
from utils.chart_builders import create_feature_importance_bar, create_waterfall_chart
from utils.data_processors import prepare_batch_result_df


def render():
    """Render the Explainability page."""
    
    st.title("🧠 SHAP Feature Explanations")
    st.markdown("Understand which features drive churn predictions globally and per customer.")
    
    api_client = st.session_state.api_client
    
    # ─────────────────────────────────────────────────────────────
    # GLOBAL FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Global Feature Importance")
    st.markdown("Average SHAP values across all customers in the dataset.")
    
    with st.spinner("Computing global importance..."):
        success, global_importance, error = api_client.get_global_importance(top_n=10)
        
        if success:
            top_features = global_importance.get("top_features", [])
            
            if top_features:
                # Extract data
                feature_names = [f["feature_name"] for f in top_features]
                importances = [f["importance"] for f in top_features]
                signs = [f["sign"] for f in top_features]
                
                # Create chart
                fig = create_feature_importance_bar(
                    feature_names,
                    importances,
                    signs,
                    top_n=10
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature table
                st.markdown("**Top 10 Features with Sign Direction**")
                
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": [f"{v:.6f}" for v in importances],
                    "Direction": signs,
                    "Impact": ["🔴 Increases Churn" if s == "positive" else "🟢 Decreases Churn" for s in signs]
                })
                
                st.dataframe(importance_df, use_container_width=True)
                
                # Legend
                st.markdown("""
                **How to interpret:**
                - 🔴 **Positive** (increases churn probability): High values of this feature increase likelihood of churn
                - 🟢 **Negative** (decreases churn probability): High values of this feature decrease likelihood of churn
                """)
                
                # Model details
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Sample Size**: {global_importance.get('sample_size', 'N/A')}")
                with col2:
                    st.markdown(f"**Explainer Type**: {global_importance.get('explainer_type', 'N/A')}")
            
            else:
                st.info("No feature importance data available.")
        
        else:
            st.error(f"❌ Error fetching global importance: {error}")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # INSTANCE-LEVEL SHAP EXPLANATIONS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Per-Customer SHAP Waterfall")
    st.markdown("Select a customer to see which features influenced their prediction.")
    
    # Customer selector
    customer_source = st.radio(
        "Select customer from:",
        options=["Batch Data", "Single Prediction"],
        key="explain_customer_source"
    )
    
    customer_data = None
    
    if customer_source == "Batch Data":
        if st.session_state.batch_predictions:
            batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
            
            customer_id = st.selectbox(
                "Select Customer ID",
                options=batch_df["customerID"].tolist(),
                key="explain_customer_id"
            )
            
            selected_row = batch_df[batch_df["customerID"] == customer_id].iloc[0]
            
            # Extract customer features - convert numpy types to Python native types
            customer_data = {}
            for col in batch_df.columns:
                if col not in ["segment", "segment_label", "segment_confidence", "churn_probability", 
                              "is_churner", "recommended_action", "top_feature_1", "top_feature_2", 
                              "top_feature_3", "customerID"]:
                    customer_data[col] = convert_numpy_to_python(selected_row[col])
        else:
            st.warning("No batch data available. Upload CSV in Batch Scoring page.")
    
    else:
        if st.session_state.selected_customer:
            customer_data = st.session_state.selected_customer.get("data", {})
            st.info("Using customer from Single Prediction page")
        else:
            st.warning("No customer from single prediction available. Go to Single Prediction page first.")
    
    if customer_data:
        with st.spinner("Computing SHAP values..."):
            success, shap_data, error = api_client.get_instance_importance(customer_data, top_n=5)
            
            if success:
                top_features = shap_data.get("top_features", [])
                base_value = shap_data.get("base_value", 0)
                prediction_value = shap_data.get("prediction_value", 0)
                
                if top_features:
                    st.markdown("**SHAP Force Plot (Waterfall Chart)**")
                    
                    # Build waterfall
                    feature_names = [f["feature_name"] for f in top_features]
                    shap_values = [f["shap_value"] for f in top_features]
                    
                    fig = create_waterfall_chart(
                        feature_names,
                        shap_values,
                        base_value,
                        prediction_value
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature contributions table
                    st.markdown("**Feature Contributions**")
                    
                    contributions_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": [f"{v:.6f}" for v in shap_values],
                        "Feature Value": [f.get("feature_value", "N/A") for f in top_features],
                        "Direction": [f.get("impact_direction", "N/A") for f in top_features],
                        "Impact": ["↑ Increases Churn" if v > 0 else "↓ Decreases Churn" for v in shap_values]
                    })
                    
                    st.dataframe(contributions_df, use_container_width=True)
                    
                    # Explanation text
                    st.markdown(f"""
                    **Interpretation:**
                    
                    - **Base Value** (starting point): {base_value:.4f}
                    - **Final Prediction** (model output): {prediction_value:.4f}
                    - **Total Change**: {prediction_value - base_value:+.4f}
                    
                    Each feature either pushes the prediction higher (red) or lower (green) based on the customer's value for that feature.
                    """)
                
                else:
                    st.info("No feature importance data available for this customer.")
            
            else:
                st.error(f"❌ Error computing SHAP values: {error}")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # EXPLAINABILITY METHODS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Explainability Methods")
    
    with st.expander("Available SHAP Methods", expanded=False):
        success, methods, error = api_client.get_explanation_methods()
        
        if success:
            methods_list = methods.get("available_methods", [])
            
            if methods_list:
                st.markdown("**Available Explainability Methods:**")
                for method in methods_list:
                    st.markdown(f"- {method}")
            else:
                st.info("No methods data available.")
        else:
            st.info("Could not fetch methods information.")
    
    # ─────────────────────────────────────────────────────────────
    # GENERAL INFORMATION
    # ─────────────────────────────────────────────────────────────
    
    st.divider()
    st.subheader("About SHAP Explanations")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** provides model-agnostic local and global explanations:
    
    - **Global Importance**: Shows which features matter most across the entire dataset
    - **Local Importance**: Shows which features influenced a specific prediction
    - **Sign Direction**: Indicates whether high values increase or decrease churn probability
    
    **Interpreting SHAP Values:**
    - Positive values push the prediction toward churn (higher probability)
    - Negative values push the prediction toward retention (lower probability)
    - Larger magnitudes indicate stronger influence on the prediction
    """)
