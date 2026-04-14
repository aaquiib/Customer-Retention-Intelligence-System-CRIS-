"""Page 07: What-If Simulator - Test retention strategies."""

import streamlit as st
from utils.api_client import APIClient
from utils.validators import validate_customer_fields, convert_numpy_to_python
from utils.chart_builders import create_gauge_chart
from config import CATEGORICAL_VALUES, CHURN_THRESHOLD, SEGMENT_LABELS, SEGMENT_COLORS


def render():
    """Render the What-If Simulator page."""
    
    st.title("🔮 What-If Simulator")
    st.markdown("Test different retention strategies and see their impact on churn probability.")
    
    api_client = st.session_state.api_client
    
    # ─────────────────────────────────────────────────────────────
    # CUSTOMER SELECTOR
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Select Customer or Enter Manual Data")
    
    input_mode = st.radio(
        "Input Mode",
        options=["Manual Entry", "Load from Batch Data"],
        key="whatif_input_mode"
    )
    
    customer_data = {}
    
    if input_mode == "Manual Entry":
        st.markdown("**Enter customer details manually**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            customer_data["gender"] = st.selectbox("Gender", CATEGORICAL_VALUES["gender"], key="whatif_gender")
            customer_data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1], key="whatif_senior")
            customer_data["Partner"] = st.selectbox("Partner", CATEGORICAL_VALUES["Partner"], key="whatif_partner")
            customer_data["Dependents"] = st.selectbox("Dependents", CATEGORICAL_VALUES["Dependents"], key="whatif_dependents")
        
        with col2:
            customer_data["tenure"] = st.number_input("Tenure (months)", 0, 72, 12, key="whatif_tenure")
            customer_data["MonthlyCharges"] = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, key="whatif_monthly")
            customer_data["TotalCharges"] = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, key="whatif_total")
        
        # Services in columns
        col1, col2 = st.columns(2)
        with col1:
            customer_data["PhoneService"] = st.selectbox("Phone Service", CATEGORICAL_VALUES["PhoneService"], key="whatif_phone")
            customer_data["MultipleLines"] = st.selectbox("Multiple Lines", CATEGORICAL_VALUES["MultipleLines"], key="whatif_lines")
            customer_data["InternetService"] = st.selectbox("Internet Service", CATEGORICAL_VALUES["InternetService"], key="whatif_internet")
            customer_data["OnlineSecurity"] = st.selectbox("Online Security", CATEGORICAL_VALUES["OnlineSecurity"], key="whatif_security")
            customer_data["OnlineBackup"] = st.selectbox("Online Backup", CATEGORICAL_VALUES["OnlineBackup"], key="whatif_backup")
        
        with col2:
            customer_data["DeviceProtection"] = st.selectbox("Device Protection", CATEGORICAL_VALUES["DeviceProtection"], key="whatif_device")
            customer_data["TechSupport"] = st.selectbox("Tech Support", CATEGORICAL_VALUES["TechSupport"], key="whatif_tech")
            customer_data["StreamingTV"] = st.selectbox("Streaming TV", CATEGORICAL_VALUES["StreamingTV"], key="whatif_tv")
            customer_data["StreamingMovies"] = st.selectbox("Streaming Movies", CATEGORICAL_VALUES["StreamingMovies"], key="whatif_movies")
        
        # Contract section
        col1, col2 = st.columns(2)
        with col1:
            customer_data["Contract"] = st.selectbox("Contract", CATEGORICAL_VALUES["Contract"], key="whatif_contract")
        with col2:
            customer_data["PaperlessBilling"] = st.selectbox("Paperless Billing", CATEGORICAL_VALUES["PaperlessBilling"], key="whatif_paperless")
        
        customer_data["PaymentMethod"] = st.selectbox("Payment Method", CATEGORICAL_VALUES["PaymentMethod"], key="whatif_payment")
    
    else:
        # Load from batch data
        if st.session_state.batch_predictions:
            from utils.data_processors import prepare_batch_result_df
            batch_df = prepare_batch_result_df(st.session_state.batch_predictions)
            
            if batch_df.empty:
                st.error("❌ Batch data is empty. No valid predictions available.")
                st.stop()
            
            customer_id = st.selectbox(
                "Select Customer",
                options=batch_df["customerID"].tolist(),
                key="whatif_customer_select"
            )
            
            # Filter to selected customer - safer than iloc
            selected_rows = batch_df[batch_df["customerID"] == customer_id]
            if selected_rows.empty:
                st.error("❌ Selected customer not found in batch data.")
                st.stop()
            
            selected_customer = selected_rows.iloc[0]
            
            # Load customer data from batch - convert numpy types to Python native types
            for field in ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "MonthlyCharges", 
                         "TotalCharges", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                         "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                         "Contract", "PaperlessBilling", "PaymentMethod"]:
                if field in selected_customer.index:
                    # Convert numpy types (int64, float64) to Python native types for JSON serialization
                    customer_data[field] = convert_numpy_to_python(selected_customer[field])
            
            st.info(f"✅ Loaded customer {int(customer_id)} from batch data")
        else:
            st.warning("❌ No batch data available. Use Manual Entry mode.")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # MODIFICATION FORM
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Scenario Modifications")
    st.markdown("Select which fields to modify. Leave others as original values.")
    
    modifications = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Modify Contract", key="mod_contract"):
            modifications["Contract"] = st.selectbox(
                "New Contract",
                CATEGORICAL_VALUES["Contract"],
                key="mod_contract_value"
            )
        
        if st.checkbox("Modify Tenure", key="mod_tenure"):
            modifications["tenure"] = st.number_input(
                "New Tenure (months)",
                0, 72, int(customer_data.get("tenure", 12)),
                key="mod_tenure_value"
            )
        
        if st.checkbox("Modify Monthly Charges", key="mod_monthly"):
            modifications["MonthlyCharges"] = st.number_input(
                "New Monthly Charges ($)",
                0.0, 200.0,
                float(customer_data.get("MonthlyCharges", 65.0)),
                key="mod_monthly_value"
            )
        
        if st.checkbox("Modify Internet Service", key="mod_internet"):
            modifications["InternetService"] = st.selectbox(
                "New Internet Service",
                CATEGORICAL_VALUES["InternetService"],
                key="mod_internet_value"
            )
    
    with col2:
        if st.checkbox("Modify Online Security", key="mod_security"):
            modifications["OnlineSecurity"] = st.selectbox(
                "New Online Security",
                CATEGORICAL_VALUES["OnlineSecurity"],
                key="mod_security_value"
            )
        
        if st.checkbox("Modify Tech Support", key="mod_tech"):
            modifications["TechSupport"] = st.selectbox(
                "New Tech Support",
                CATEGORICAL_VALUES["TechSupport"],
                key="mod_tech_value"
            )
        
        if st.checkbox("Modify Payment Method", key="mod_payment"):
            modifications["PaymentMethod"] = st.selectbox(
                "New Payment Method",
                CATEGORICAL_VALUES["PaymentMethod"],
                key="mod_payment_value"
            )
        
        if st.checkbox("Modify Paperless Billing", key="mod_paperless"):
            modifications["PaperlessBilling"] = st.selectbox(
                "New Paperless Billing",
                CATEGORICAL_VALUES["PaperlessBilling"],
                key="mod_paperless_value"
            )
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # SIMULATE BUTTON
    # ─────────────────────────────────────────────────────────────
    
    simulate_button = st.button("⚡ Simulate Scenario", key="simulate_button", use_container_width=True)
    
    if simulate_button:
        # Validate customer data
        is_valid, errors = validate_customer_fields(customer_data)
        
        if not is_valid:
            st.error("❌ Customer data validation failed:")
            for error in errors:
                st.error(f"  - {error}")
        else:
            with st.spinner("Running simulation..."):
                # Call what-if API
                success, scenario, error = api_client.what_if_single(customer_data, modifications)
                
                if success:
                    # ─────────────────────────────────────────────────────────────
                    # SIMULATION RESULTS
                    # ─────────────────────────────────────────────────────────────
                    
                    st.divider()
                    st.subheader("Scenario Comparison")
                    
                    original_pred = scenario.get("original_prediction", {})
                    modified_pred = scenario.get("modified_prediction", {})
                    delta = scenario.get("delta", {})
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    # ORIGINAL PREDICTION
                    with col1:
                        st.markdown("**ORIGINAL**")
                        
                        orig_seg_id = original_pred.get("segment", -1)
                        orig_seg_color = SEGMENT_COLORS.get(orig_seg_id, "#95a5a6")
                        
                        st.markdown(f"""
                        <div style="background-color: {orig_seg_color}; padding: 10px; border-radius: 6px; color: white; text-align: center; font-size: 12px;">
                            {SEGMENT_LABELS.get(orig_seg_id, 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Churn Probability",
                            f"{original_pred.get('churn_probability', 0):.4f}"
                        )
                        
                        st.metric(
                            "Status",
                            "🔴 WILL CHURN" if original_pred.get("is_churner") else "✅ SAFE"
                        )
                    
                    # DELTA
                    with col2:
                        st.markdown("**DELTA (CHANGE)**")
                        
                        delta_prob = delta.get("churn_probability_delta", 0)
                        
                        if delta_prob < 0:
                            delta_color = "#2ecc71"
                            delta_icon = "↓"
                            delta_label = "IMPROVEMENT"
                        else:
                            delta_color = "#e74c3c"
                            delta_icon = "↑"
                            delta_label = "WORSENING"
                        
                        st.markdown(f"""
                        <div style="background-color: {delta_color}; padding: 10px; border-radius: 6px; color: white; text-align: center;">
                            {delta_icon} {delta_label}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Probability Delta",
                            f"{delta_prob:+.4f}"
                        )
                        
                        # Segment change indicator
                        if delta.get("segment_changed"):
                            st.warning(f"⚠️ Segment changed to {SEGMENT_LABELS.get(modified_pred.get('segment'), 'Unknown')}")
                        else:
                            st.success("✅ Segment unchanged")
                    
                    # MODIFIED PREDICTION
                    with col3:
                        st.markdown("**MODIFIED**")
                        
                        mod_seg_id = modified_pred.get("segment", -1)
                        mod_seg_color = SEGMENT_COLORS.get(mod_seg_id, "#95a5a6")
                        
                        st.markdown(f"""
                        <div style="background-color: {mod_seg_color}; padding: 10px; border-radius: 6px; color: white; text-align: center; font-size: 12px;">
                            {SEGMENT_LABELS.get(mod_seg_id, 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Churn Probability",
                            f"{modified_pred.get('churn_probability', 0):.4f}"
                        )
                        
                        st.metric(
                            "Status",
                            "🔴 WILL CHURN" if modified_pred.get("is_churner") else "✅ SAFE"
                        )
                    
                    st.divider()
                    
                    # Threshold crossing indicator
                    orig_churner = original_pred.get("is_churner", False)
                    mod_churner = modified_pred.get("is_churner", False)
                    
                    if orig_churner and not mod_churner:
                        st.success("✅ **SUCCESS**: Customer converted from churner to non-churner!")
                    elif not orig_churner and mod_churner:
                        st.warning("⚠️ **WARNING**: Modification worsens customer status!")
                    
                    # Detailed comparison
                    st.subheader("Modified Fields")
                    
                    if modifications:
                        for field, new_value in modifications.items():
                            original_value = customer_data.get(field, "N/A")
                            st.markdown(f"**{field}**: {original_value} → **{new_value}**")
                    else:
                        st.info("No modifications specified")
                
                else:
                    st.error(f"❌ Simulation failed: {error}")
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────
    # PRE-DEFINED POLICY SCENARIOS
    # ─────────────────────────────────────────────────────────────
    
    st.subheader("Pre-defined Policy Scenarios")
    
    with st.spinner("Loading policy scenarios..."):
        success, policy_scenarios, error = api_client.get_policy_scenarios()
        
        if success and policy_scenarios:
            scenario_cols = st.columns(min(4, len(policy_scenarios)))
            
            for idx, scenario in enumerate(policy_scenarios[:4]):
                with scenario_cols[idx]:
                    scenario_name = scenario.get("name", "Unknown")
                    scenario_desc = scenario.get("description", "")
                    
                    if st.button(f"📌 {scenario_name}", key=f"policy_scenario_{idx}", use_container_width=True):
                        st.info(f"**{scenario_name}**\n\n{scenario_desc}")
                        
                        # Auto-fill modifications from scenario
                        with st.spinner("Processing scenario..."):
                            scenario_mods = scenario.get("modifications", {})
                            success, result, error = api_client.what_if_single(customer_data, scenario_mods)
                            
                            if success:
                                st.success("✅ Scenario processed!")
                                # Display result
                            else:
                                st.error(f"❌ {error}")
        else:
            st.info("No pre-defined policy scenarios available.")
