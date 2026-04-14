"""Page 09: Model Health - Model metrics, performance, and deployment info."""

import streamlit as st
from utils.api_client import APIClient
from config import SEGMENT_LABELS


def render():
    """Render the Model Health page."""
    
    st.title("🏥 Model Health & Metadata")
    st.markdown("Model performance metrics, architecture, and system information.")
    
    api_client = st.session_state.api_client
    
    with st.spinner("Fetching model information..."):
        success, model_info, error = api_client.get_model_info()
        
        if not success:
            st.error(f"❌ Error fetching model info: {error}")
            return
        
        # ─────────────────────────────────────────────────────────────
        # CHURN MODEL SECTION
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Churn Prediction Model")
        
        churn_model = model_info.get("churn_model", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Type",
                churn_model.get("model_name", "Unknown")
            )
        
        with col2:
            threshold = churn_model.get("decision_threshold", 0.4356)
            st.metric(
                "Decision Threshold",
                f"{threshold:.4f}"
            )
        
        with col3:
            st.metric(
                "Training Samples",
                churn_model.get("training_data_size", "N/A")
            )
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # PERFORMANCE METRICS
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Model Performance Metrics")
        
        metrics = churn_model.get("performance_metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auc = metrics.get("roc_auc", 0)
            # AUC color coding
            if auc > 0.85:
                metric_color = "green"
                emoji = "✅"
            elif auc > 0.75:
                metric_color = "blue"
                emoji = "✓"
            else:
                metric_color = "orange"
                emoji = "⚠️"
            
            st.metric(
                f"{emoji} AUC-ROC",
                f"{auc:.4f}",
                delta="Excellent" if auc > 0.85 else ("Good" if auc > 0.75 else "Fair")
            )
        
        with col2:
            accuracy = metrics.get("accuracy", 0)
            st.metric(
                "Accuracy",
                f"{accuracy:.4f}",
                delta=f"{accuracy*100:.1f}%"
            )
        
        with col3:
            precision = metrics.get("precision", 0)
            st.metric(
                "Precision",
                f"{precision:.4f}",
                delta=f"{precision*100:.1f}%"
            )
        
        with col4:
            recall = metrics.get("recall", 0)
            st.metric(
                "Recall (Sensitivity)",
                f"{recall:.4f}",
                delta=f"{recall*100:.1f}%"
            )
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # MODEL DETAILS TABLE
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Churn Model Architecture")
        
        with st.expander("Detailed Model Information", expanded=True):
            model_details = [
                ("Model Name", churn_model.get("model_name", "N/A")),
                ("Model Framework", churn_model.get("framework", "N/A")),
                ("Input Features", churn_model.get("num_features", "N/A")),
                ("Decision Threshold", f"{churn_model.get('decision_threshold', 0.4356):.4f}"),
                ("Training Data Size", churn_model.get("training_data_size", "N/A")),
                ("Feature Set", churn_model.get("feature_set_version", "N/A")),
                ("Last Updated", churn_model.get("training_date", "N/A")),
            ]
            
            for label, value in model_details:
                st.markdown(f"**{label}**: {value}")
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # SEGMENTATION MODEL SECTION
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Customer Segmentation Model")
        
        seg_model = model_info.get("segmentation_model", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Type",
                seg_model.get("model_name", "Unknown")
            )
        
        with col2:
            st.metric(
                "Number of Clusters",
                seg_model.get("num_clusters", 4)
            )
        
        with col3:
            st.metric(
                "Training Samples",
                seg_model.get("training_data_size", "N/A")
            )
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # SEGMENT DEFINITIONS
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Segment Definitions")
        
        segments = seg_model.get("segments", {})
        
        segment_cols = st.columns(4)
        
        for seg_id in range(4):
            with segment_cols[seg_id]:
                seg_label = SEGMENT_LABELS.get(seg_id, f"Segment {seg_id}")
                seg_info = segments.get(str(seg_id), {})
                
                st.markdown(f"""
                **{seg_label}**
                
                {seg_info.get('description', 'No description available')}
                """)
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # EXPLAINER INFO
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Explainability Engine")
        
        explainer = model_info.get("explainer", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Explainer Type",
                explainer.get("type", "N/A")
            )
        
        with col2:
            st.metric(
                "Background Samples",
                explainer.get("background_samples", "N/A")
            )
        
        with col3:
            st.metric(
                "Computation Type",
                explainer.get("computation_type", "N/A")
            )
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # API HEALTH
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("System Health")
        
        success, health_status, error = api_client.get_health()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            api_status = "🟢 Running" if success else "🔴 Offline"
            st.metric(
                "API Status",
                api_status
            )
        
        with col2:
            st.metric(
                "Models Loaded",
                "✅ Yes" if success else "❌ No"
            )
        
        with col3:
            st.metric(
                "Predictions Ready",
                "✅ Yes" if success else "❌ No"
            )
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # PERFORMANCE INTERPRETATION
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Performance Interpretation")
        
        with st.expander("What do these metrics mean?", expanded=False):
            st.markdown("""
            **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
            - Measures model's ability to distinguish between churners and non-churners
            - 0.5 = random, 1.0 = perfect
            - >0.8 = Excellent, 0.7-0.8 = Good, <0.7 = Fair
            
            **Accuracy**: Percentage of correct predictions overall
            - (TP + TN) / (TP + TN + FP + FN)
            - Can be misleading with imbalanced data
            
            **Precision**: Of predicted churners, how many actually churn?
            - TP / (TP + FP)
            - Important for reducing false alarms in retention campaigns
            
            **Recall**: Of actual churners, how many did we identify?
            - TP / (TP + FN)
            - Important for catching high-risk customers
            
            **Decision Threshold**: Probability cutoff for classifying as churner
            - Default 0.4356 balances precision and recall
            - Can be adjusted based on business priorities
            """)
        
        st.divider()
        
        # ─────────────────────────────────────────────────────────────
        # MODEL INFORMATION EXPORT
        # ─────────────────────────────────────────────────────────────
        
        st.subheader("Model Information Export")
        
        # Format as readable text
        model_info_text = f"""
# Churn Segmentation Decision System - Model Report

## Churn Prediction Model
- **Name**: {churn_model.get('model_name', 'N/A')}
- **Framework**: {churn_model.get('framework', 'N/A')}
- **Features**: {churn_model.get('num_features', 'N/A')}
- **Training Samples**: {churn_model.get('training_data_size', 'N/A')}

### Performance Metrics
- **AUC-ROC**: {metrics.get('roc_auc', 'N/A')}
- **Accuracy**: {metrics.get('accuracy', 'N/A')}
- **Precision**: {metrics.get('precision', 'N/A')}
- **Recall**: {metrics.get('recall', 'N/A')}

### Configuration
- **Decision Threshold**: {churn_model.get('decision_threshold', '0.4356')}
- **Last Updated**: {churn_model.get('training_date', 'N/A')}

## Customer Segmentation Model
- **Name**: {seg_model.get('model_name', 'N/A')}
- **Clusters**: {seg_model.get('num_clusters', '4')}
- **Training Samples**: {seg_model.get('training_data_size', 'N/A')}

## Explainability
- **Type**: {explainer.get('type', 'N/A')}
- **Background Samples**: {explainer.get('background_samples', 'N/A')}
"""
        
        st.download_button(
            label="📥 Download Model Report (TXT)",
            data=model_info_text,
            file_name="model_health_report.txt",
            mime="text/plain",
            key="download_model_report"
        )
