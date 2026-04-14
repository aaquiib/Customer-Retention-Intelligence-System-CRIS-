"""Page 02: Single Prediction - 19-field form with SHAP explanation."""

import streamlit as st
import pandas as pd
from utils.api_client import APIClient
from utils.validators import validate_customer_fields
from utils.chart_builders import create_waterfall_chart, create_gauge_chart
from config import (
    CATEGORICAL_VALUES,
    CUSTOMER_FIELDS,
    SEGMENT_LABELS,
    SEGMENT_COLORS,
    CHURN_THRESHOLD,
    RISK_BANDS
)


def render():
    """Render the Single Prediction page."""
    
    st.title("👤 Single Customer Prediction")
    st.markdown("Enter customer details to get churn prediction and SHAP explanation.")
    
    api_client = st.session_state.api_client
    
    # ─────────────────────────────────────────────────────────────
    # FORM INPUT (Left Panel)
    # ─────────────────────────────────────────────────────────────
    
    col_form, col_output = st.columns([1, 1])
    
    with col_form:
        st.subheader("Customer Details")
        
        customer_data = {}
        
        # Demographic Section
        with st.container():
            st.markdown("**Demographic Information**")
            col1, col2 = st.columns(2)
            
            with col1:
                customer_data["gender"] = st.selectbox(
                    "Gender",
                    CATEGORICAL_VALUES["gender"],
                    key="form_gender"
                )
            
            with col2:
                customer_data["SeniorCitizen"] = st.selectbox(
                    "Senior Citizen",
                    [0, 1],
                    key="form_senior"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                customer_data["Partner"] = st.selectbox(
                    "Has Partner",
                    CATEGORICAL_VALUES["Partner"],
                    key="form_partner"
                )
            
            with col2:
                customer_data["Dependents"] = st.selectbox(
                    "Has Dependents",
                    CATEGORICAL_VALUES["Dependents"],
                    key="form_dependents"
                )
        
        st.divider()
        
        # Account Section
        with st.container():
            st.markdown("**Account Information**")
            col1, col2 = st.columns(2)
            
            with col1:
                customer_data["tenure"] = st.number_input(
                    "Tenure (months)",
                    min_value=0,
                    max_value=72,
                    value=12,
                    step=1,
                    key="form_tenure"
                )
            
            with col2:
                customer_data["MonthlyCharges"] = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=200.0,
                    value=65.0,
                    step=0.5,
                    key="form_monthly"
                )
            
            customer_data["TotalCharges"] = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=780.0,
                step=10.0,
                key="form_total"
            )
        
        st.divider()
        
        # Services Section
        with st.container():
            st.markdown("**Services**")
            
            col1, col2 = st.columns(2)
            with col1:
                customer_data["PhoneService"] = st.selectbox(
                    "Phone Service",
                    CATEGORICAL_VALUES["PhoneService"],
                    key="form_phone"
                )
                customer_data["MultipleLines"] = st.selectbox(
                    "Multiple Lines",
                    CATEGORICAL_VALUES["MultipleLines"],
                    key="form_lines"
                )
                customer_data["InternetService"] = st.selectbox(
                    "Internet Service",
                    CATEGORICAL_VALUES["InternetService"],
                    key="form_internet"
                )
                customer_data["OnlineSecurity"] = st.selectbox(
                    "Online Security",
                    CATEGORICAL_VALUES["OnlineSecurity"],
                    key="form_security"
                )
                customer_data["OnlineBackup"] = st.selectbox(
                    "Online Backup",
                    CATEGORICAL_VALUES["OnlineBackup"],
                    key="form_backup"
                )
            
            with col2:
                customer_data["DeviceProtection"] = st.selectbox(
                    "Device Protection",
                    CATEGORICAL_VALUES["DeviceProtection"],
                    key="form_device"
                )
                customer_data["TechSupport"] = st.selectbox(
                    "Tech Support",
                    CATEGORICAL_VALUES["TechSupport"],
                    key="form_tech"
                )
                customer_data["StreamingTV"] = st.selectbox(
                    "Streaming TV",
                    CATEGORICAL_VALUES["StreamingTV"],
                    key="form_tv"
                )
                customer_data["StreamingMovies"] = st.selectbox(
                    "Streaming Movies",
                    CATEGORICAL_VALUES["StreamingMovies"],
                    key="form_movies"
                )
        
        st.divider()
        
        # Contract & Billing Section
        with st.container():
            st.markdown("**Contract & Billing**")
            col1, col2 = st.columns(2)
            
            with col1:
                customer_data["Contract"] = st.selectbox(
                    "Contract",
                    CATEGORICAL_VALUES["Contract"],
                    key="form_contract"
                )
                customer_data["PaperlessBilling"] = st.selectbox(
                    "Paperless Billing",
                    CATEGORICAL_VALUES["PaperlessBilling"],
                    key="form_paperless"
                )
            
            with col2:
                customer_data["PaymentMethod"] = st.selectbox(
                    "Payment Method",
                    CATEGORICAL_VALUES["PaymentMethod"],
                    key="form_payment"
                )
        
        st.divider()
        
        # Predict Button
        predict_button = st.button("🔮 Predict Churn", key="predict_button", use_container_width=True)
    
    # ─────────────────────────────────────────────────────────────
    # PREDICTION OUTPUT (Right Panel)
    # ─────────────────────────────────────────────────────────────
    
    with col_output:
        st.subheader("Prediction Results")
        
        if predict_button:
            with st.spinner("Getting prediction..."):
                # Validate input
                is_valid, errors = validate_customer_fields(customer_data)
                
                if not is_valid:
                    st.error("❌ Validation failed:")
                    for error in errors:
                        st.error(f"  - {error}")
                else:
                    # Call API
                    success, prediction, error = api_client.predict_single(
                        customer_data,
                        return_features=True
                    )
                    
                    if success:
                        # Store in session for later reference
                        st.session_state.selected_customer = {
                            "data": customer_data,
                            "prediction": prediction
                        }
                        
                        # Display results
                        segment_id = prediction.get("segment", -1)
                        segment_label = prediction.get("segment_label", "Unknown")
                        segment_confidence = prediction.get("segment_confidence", 0)
                        churn_prob = prediction.get("churn_probability", 0)
                        is_churner = prediction.get("is_churner", False)
                        recommended_action = prediction.get("recommended_action", {})
                        
                        # Segment badge
                        segment_color = SEGMENT_COLORS.get(segment_id, "#95a5a6")
                        st.markdown(f"""
                        <div style="background-color: {segment_color}; padding: 15px; border-radius: 8px; text-align: center;">
                            <p style="color: white; font-size: 12px; margin: 0;">SEGMENT</p>
                            <p style="color: white; font-size: 24px; font-weight: bold; margin: 5px 0;">{segment_label}</p>
                            <p style="color: white; font-size: 10px; margin: 0;">Confidence: {segment_confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Churn gauge
                        fig = create_gauge_chart(
                            churn_prob,
                            "Churn Probability",
                            threshold=CHURN_THRESHOLD
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Status badge
                        if churn_prob > 0.65:
                            risk_level = "🔴 HIGH RISK"
                            risk_color = "#e74c3c"
                        elif churn_prob >= 0.35:
                            risk_level = "🟡 MEDIUM RISK"
                            risk_color = "#f39c12"
                        else:
                            risk_level = "🟢 LOW RISK"
                            risk_color = "#2ecc71"
                        
                        st.markdown(f"""
                        <div style="background-color: {risk_color}; padding: 10px; border-radius: 6px; text-align: center; color: white; font-weight: bold;">
                            {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Churner status
                        churner_status = "✅ WILL CHURN" if is_churner else "❌ UNLIKELY TO CHURN"
                        st.markdown(f"**Decision**: {churner_status}")
                        st.markdown(f"**Threshold**: {CHURN_THRESHOLD}")
                        
                        # Recommended action
                        if recommended_action:
                            st.markdown(f"""
                            **Recommended Action**: {recommended_action.get('action_label', 'N/A')}
                            
                            *Reason*: {recommended_action.get('reason', 'N/A')}
                            """)
                    
                    else:
                        st.error(f"❌ Prediction failed: {error}")
    
    # ─────────────────────────────────────────────────────────────
    # SHAP EXPLANATION (Full Width)
    # ─────────────────────────────────────────────────────────────
    
    if predict_button and st.session_state.selected_customer:
        st.divider()
        st.subheader("SHAP Feature Explanation")
        
        with st.spinner("Computing SHAP values..."):
            success, shap_data, error = api_client.get_instance_importance(
                customer_data,
                top_n=5
            )
            
            if success:
                top_features = shap_data.get("top_features", [])
                base_value = shap_data.get("base_value", 0)
                prediction_value = shap_data.get("prediction_value", 0)
                
                if top_features:
                    # Build data for waterfall
                    feature_names = [f["feature_name"] for f in top_features]
                    shap_values = [f["shap_value"] for f in top_features]
                    
                    # Waterfall chart
                    fig = create_waterfall_chart(
                        feature_names,
                        shap_values,
                        base_value,
                        prediction_value
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature table
                    st.markdown("**Top 5 Contributing Features**")
                    
                    feature_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": [f"{v:.6f}" for v in shap_values],
                        "Direction": ["↑ Increases" if v > 0 else "↓ Decreases" for v in shap_values],
                        "Impact": ["🔴 Positive" if v > 0 else "🟢 Negative" for v in shap_values]
                    })
                    
                    st.dataframe(feature_df, use_container_width=True)
                    
                    st.markdown("""
                    **How to read:**
                    - 🔴 **Positive SHAP values** increase churn probability
                    - 🟢 **Negative SHAP values** decrease churn probability
                    - Base value: {:.4f}
                    - Final prediction: {:.4f}
                    """.format(base_value, prediction_value))
                else:
                    st.info("No feature importance data available.")
            
            else:
                st.error(f"❌ SHAP computation failed: {error}")
