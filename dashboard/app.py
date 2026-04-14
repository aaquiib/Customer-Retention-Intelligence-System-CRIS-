"""Main Streamlit app entry point with sidebar navigation."""

import streamlit as st
import logging
from datetime import datetime
from utils.api_client import APIClient
from config import API_BASE_URL, SEGMENT_LABELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="CRIS - Customer Retention Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #34495e;
        margin-top: 20px;
    }
    
    /* KPI cards */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Success/Error messages */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 4px;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 4px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()

if "batch_data" not in st.session_state:
    st.session_state.batch_data = None

if "batch_predictions" not in st.session_state:
    st.session_state.batch_predictions = []

if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎯 CRIS")
    st.markdown("**Customer Retention Intelligence System**")
    st.divider()
    
    # Health status
    st.subheader("Status")
    st.info("✅ System Ready | API Connected")
    
    st.divider()
    
    # Model info badge (with timeout handling)
    st.subheader("Model Info")
    st.markdown("""
    - **Type**: LightGBM Classifier
    - **Features**: 33 customer attributes
    - **Estimators**: 650
    - **Threshold**: 0.4356
    """)
    
    st.divider()
    
    # Navigation
    st.subheader("Navigation")
    
    pages = [
        ("📊 Overview", "Overview"),
        ("👤 Single Prediction", "Single Prediction"),
        ("📁 Batch Scoring", "Batch Scoring"),
        ("🔍 Segment Intelligence", "Segment Intelligence"),
        ("⚠️ Churn Risk Analysis", "Churn Risk Analysis"),
        ("📋 Action Planning", "Action Planning"),
        ("🔮 What-If Simulator", "What-If Simulator"),
        ("🧠 Explainability", "Explainability"),
        ("🏥 Model Health", "Model Health"),
    ]
    
    selected_page = st.radio(
        "Select Page:",
        options=[p[0] for p in pages],
        label_visibility="collapsed"
    )
    
    # Update session state with selected page
    for label, page_name in pages:
        if label == selected_page:
            st.session_state.current_page = page_name
            break
    
    st.divider()
    
    # Help section
    st.subheader("Help")
    with st.expander("About this dashboard"):
        st.markdown("""
        **CRIS** helps you:
        - 🎯 Identify customers at risk of churning
        - 📊 Understand key drivers of churn (SHAP)
        - 🔮 Test retention strategies (What-If)
        - 📋 Plan targeted retention actions
        
        **Features:**
        - Single & batch customer scoring
        - Segment-based analysis
        - Interactive what-if scenarios
        - SHAP feature explanations
        """)
    
    st.markdown(f"""
    ---
    **API**: {API_BASE_URL}  
    **Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)


# ─────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────

page_name = st.session_state.current_page

if page_name == "Overview":
    from pages import page_01_overview
    page_01_overview.render()

elif page_name == "Single Prediction":
    from pages import page_02_single_prediction
    page_02_single_prediction.render()

elif page_name == "Batch Scoring":
    from pages import page_03_batch_scoring
    page_03_batch_scoring.render()

elif page_name == "Segment Intelligence":
    from pages import page_04_segment_intelligence
    page_04_segment_intelligence.render()

elif page_name == "Churn Risk Analysis":
    from pages import page_05_churn_risk
    page_05_churn_risk.render()

elif page_name == "Action Planning":
    from pages import page_06_action_planning
    page_06_action_planning.render()

elif page_name == "What-If Simulator":
    from pages import page_07_what_if_simulator
    page_07_what_if_simulator.render()

elif page_name == "Explainability":
    from pages import page_08_explainability
    page_08_explainability.render()

elif page_name == "Model Health":
    from pages import page_09_model_health
    page_09_model_health.render()

else:
    st.error(f"Unknown page: {page_name}")
