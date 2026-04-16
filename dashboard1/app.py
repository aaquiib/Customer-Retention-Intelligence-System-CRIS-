"""
CRIS - Customer Retention Intelligence System
Professional Streamlit Dashboard

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_option_menu import option_menu

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRIS — Retention Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = st.session_state.get("api_base", "http://localhost:8000")

# ─────────────────────────────────────────────────────────────────
# THEME — sophisticated, dark, muted
# ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Base palette — graphite + muted teal/amber accents */
    :root {
        --bg-0: #0b0f14;
        --bg-1: #111820;
        --bg-2: #161f2a;
        --bg-3: #1e2935;
        --line: #243242;
        --text-0: #e6edf3;
        --text-1: #a6b3c0;
        --text-2: #6c7a89;
        --accent: #7dd3c0;      /* muted teal */
        --accent-dim: #3e6e66;
        --warn: #d4a574;         /* soft amber */
        --danger: #c97a7a;       /* muted rose */
        --ok: #8fbc8f;           /* sage */
    }

    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background: var(--bg-0) !important;
        color: var(--text-0) !important;
    }
    .block-container { padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1500px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e13 0%, #0f141c 100%) !important;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text-0) !important; }

    /* Headings */
    h1, h2, h3, h4 { color: var(--text-0) !important; font-weight: 500; letter-spacing: -0.01em; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.05rem !important; color: var(--text-1) !important; }

    /* Cards */
    .cris-card {
        background: linear-gradient(145deg, var(--bg-1), var(--bg-2));
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 0 rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.25);
    }
    .cris-card-tight { padding: 14px 16px; }
    .cris-label {
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--text-2); margin-bottom: 6px; font-weight: 500;
    }
    .cris-value {
        font-size: 1.9rem; font-weight: 300; color: var(--text-0);
        font-variant-numeric: tabular-nums;
    }
    .cris-delta { font-size: 0.85rem; color: var(--text-1); margin-top: 4px; }
    .cris-delta.up   { color: var(--ok); }
    .cris-delta.down { color: var(--danger); }

    /* Chips */
    .cris-chip {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 500; letter-spacing: 0.05em;
        border: 1px solid var(--line); background: var(--bg-2); color: var(--text-1);
        margin-right: 6px;
    }
    .cris-chip.accent { color: var(--accent); border-color: var(--accent-dim); background: rgba(125,211,192,0.08); }
    .cris-chip.warn   { color: var(--warn);   border-color: #6b5336;         background: rgba(212,165,116,0.08); }
    .cris-chip.danger { color: var(--danger); border-color: #6b3939;         background: rgba(201,122,122,0.08); }
    .cris-chip.ok     { color: var(--ok);     border-color: #4a6a4a;         background: rgba(143,188,143,0.08); }

    /* Inputs */
    input, textarea, select, .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb] {
        background: var(--bg-1) !important;
        border: 1px solid var(--line) !important;
        color: var(--text-0) !important;
        border-radius: 8px !important;
    }
    .stSelectbox div[data-baseweb] > div { background: var(--bg-1) !important; }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(180deg, #1b2532, #141b25) !important;
        color: var(--text-0) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent-dim) !important;
        color: var(--accent) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, #2a5b53, #1e4a43) !important;
        border-color: var(--accent-dim) !important;
        color: #e6f7f2 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-2) !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 10px 18px !important;
        font-size: 0.9rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* Dataframe */
    .stDataFrame, .stTable { background: var(--bg-1) !important; border: 1px solid var(--line); border-radius: 8px; }

    /* Progress/meters */
    .stProgress > div > div > div > div { background: var(--accent) !important; }

    /* Divider accent */
    .cris-hr { height: 1px; background: linear-gradient(90deg, transparent, var(--line), transparent); margin: 18px 0; }

    /* Status dot */
    .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; box-shadow: 0 0 8px currentColor; }
    .dot.ok { background: var(--ok); color: var(--ok); }
    .dot.bad { background: var(--danger); color: var(--danger); }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Section header (brand) */
    .brand {
        display: flex; align-items: center; gap: 12px; margin-bottom: 18px;
        padding-bottom: 14px; border-bottom: 1px solid var(--line);
    }
    .brand .mark {
        width: 36px; height: 36px; border-radius: 8px;
        background: linear-gradient(135deg, #2a5b53, #1e4a43);
        display: flex; align-items: center; justify-content: center;
        color: #c7eee5; font-weight: 600; font-size: 1.1rem;
        border: 1px solid var(--accent-dim);
    }
    .brand .title { font-size: 1.05rem; font-weight: 500; letter-spacing: 0.02em; }
    .brand .sub   { font-size: 0.72rem; color: var(--text-2); letter-spacing: 0.1em; text-transform: uppercase; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# API CLIENT
# ─────────────────────────────────────────────────────────────────

def _raw_req(base: str, method: str, path: str, timeout: int = 60, **kw):
    url = f"{base.rstrip('/')}{path}"
    try:
        r = requests.request(method, url, timeout=timeout, **kw)
        if r.status_code >= 400:
            try: detail = r.json()
            except: detail = r.text
            return {"__error__": f"HTTP {r.status_code}", "detail": detail}
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"__error__": "Cannot reach API", "detail": f"Is the API running at {base}?"}
    except Exception as e:
        return {"__error__": type(e).__name__, "detail": str(e)}

# ── Cached GETs: prevent re-fetching on every widget interaction ──
@st.cache_data(ttl=15, show_spinner=False)
def _cached_health(base: str):            return _raw_req(base, "GET", "/health", timeout=5)

@st.cache_data(ttl=300, show_spinner=False)
def _cached_model_info(base: str):        return _raw_req(base, "GET", "/api/explanations/model-info")

@st.cache_data(ttl=300, show_spinner=False)
def _cached_global_importance(base: str, n: int):
    return _raw_req(base, "GET", f"/api/feature-importance/global?top_n={n}")

@st.cache_data(ttl=600, show_spinner=False)
def _cached_batch_template(base: str):    return _raw_req(base, "GET", "/api/predict-batch/template")

@st.cache_data(ttl=600, show_spinner=False)
def _cached_policy_scenarios(base: str):  return _raw_req(base, "GET", "/api/what-if/policy-changes")

@st.cache_data(ttl=600, show_spinner=False)
def _cached_explanation_methods(base: str): return _raw_req(base, "GET", "/api/explanations/methods")


class CRISClient:
    def __init__(self, base: str, timeout: int = 60):
        self.base = base.rstrip("/")
        self.timeout = timeout

    # GETs → cached (no network on widget-only reruns)
    def health(self):                    return _cached_health(self.base)
    def model_info(self):                return _cached_model_info(self.base)
    def global_importance(self, n=10):   return _cached_global_importance(self.base, n)
    def batch_template(self):            return _cached_batch_template(self.base)
    def policy_scenarios(self):          return _cached_policy_scenarios(self.base)
    def explanation_methods(self):       return _cached_explanation_methods(self.base)

    # POSTs → only invoked explicitly by action buttons
    def instance_importance(self, c, n=5):
        return _raw_req(self.base, "POST", f"/api/feature-importance/instance?top_n={n}", json=c)
    def predict(self, customer, return_features=False):
        return _raw_req(self.base, "POST", "/api/predict",
                        json={"customer": customer, "return_features": return_features})
    def what_if(self, customer, modifications):
        return _raw_req(self.base, "POST", "/api/what-if",
                        json={"customer": customer, "modifications": modifications})
    def predict_batch(self, file_bytes, filename):
        try:
            r = requests.post(f"{self.base}/api/predict-batch",
                              files={"file": (filename, file_bytes, "text/csv")},
                              timeout=600)
            return r.json()
        except Exception as e:
            return {"__error__": type(e).__name__, "detail": str(e)}


# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "api_base": "http://localhost:8000",
        "last_prediction": None,
        "last_batch": None,
        "last_whatif": None,
        "customer_form": {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 840.0,
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",  # valid trained category
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()
client = CRISClient(st.session_state.api_base)


# ─────────────────────────────────────────────────────────────────
# HELPERS — UI COMPONENTS
# ─────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a6b3c0", family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=30, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#243242", zerolinecolor="#243242"),
    yaxis=dict(gridcolor="#243242", zerolinecolor="#243242"),
    colorway=["#7dd3c0", "#d4a574", "#8fbc8f", "#c97a7a", "#a0a8c0", "#9b87c4"],
)

def metric_card(label: str, value: str, delta: str = "", tone: str = ""):
    delta_html = f'<div class="cris-delta {tone}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="cris-card cris-card-tight">'
        f'<div class="cris-label">{label}</div>'
        f'<div class="cris-value">{value}</div>{delta_html}'
        f'</div>', unsafe_allow_html=True)

def chip(text, tone="accent"):
    return f'<span class="cris-chip {tone}">{text}</span>'

def gauge(value: float, title: str = "Churn Probability", threshold: float = 0.4356):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 42, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6c7a89", "tickwidth": 1},
            "bar": {"color": "#7dd3c0", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(143,188,143,0.15)"},
                {"range": [threshold * 100, 70], "color": "rgba(212,165,116,0.15)"},
                {"range": [70, 100], "color": "rgba(201,122,122,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#d4a574", "width": 2},
                "thickness": 0.75,
                "value": threshold * 100,
            },
        },
    ))
    layout = {**PLOTLY_LAYOUT, "margin": dict(l=20, r=20, t=20, b=20)}
    fig.update_layout(**layout, height=240)
    fig.update_layout(title=dict(text=title, x=0.5, y=0.05, font=dict(color="#6c7a89", size=11)))
    return fig

def shap_bar(features: List[Dict], title="Top Drivers"):
    if not features:
        return None
    df = pd.DataFrame(features)
    vcol = "shap_value" if "shap_value" in df.columns else "importance"
    df = df.sort_values(vcol)
    colors = ["#c97a7a" if v > 0 else "#8fbc8f" for v in df[vcol]]
    fig = go.Figure(go.Bar(
        x=df[vcol], y=df.get("feature_name", df.get("feature")),
        orientation="h", marker=dict(color=colors, line=dict(color="#243242", width=1)),
        text=[f"{v:+.3f}" for v in df[vcol]], textposition="outside",
        textfont=dict(color="#a6b3c0", size=10),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=max(220, 32 * len(df) + 60),
                      title=dict(text=title, font=dict(color="#a6b3c0", size=12)))
    fig.add_vline(x=0, line=dict(color="#243242", width=1))
    return fig


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="mark">◆</div>'
        '<div><div class="title">CRIS</div>'
        '<div class="sub">Retention Intelligence</div></div>'
        '</div>', unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Overview", "Customer Analysis", "Batch Scoring", "Action Planning",
                 "What-If Lab", "Explainability", "Model Intelligence",
                 "Formulas", "Settings"],
        icons=["speedometer2", "person-lines-fill", "file-earmark-spreadsheet",
               "clipboard-check", "sliders2", "diagram-3", "cpu",
               "calculator", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#6c7a89", "font-size": "15px"},
            "nav-link": {
                "font-size": "0.9rem", "color": "#a6b3c0", "padding": "10px 14px",
                "border-radius": "6px", "margin": "2px 0", "--hover-color": "#1e2935",
            },
            "nav-link-selected": {
                "background-color": "#1e2935", "color": "#7dd3c0",
                "font-weight": "500", "border-left": "2px solid #7dd3c0",
            },
        },
    )

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Live health
    h = client.health()
    if "__error__" in h:
        st.markdown('<div><span class="dot bad"></span><small>API offline</small></div>', unsafe_allow_html=True)
        st.caption(f"{h.get('detail','')}")
    else:
        ok = h.get("models_loaded", False)
        dot = "ok" if ok else "bad"
        txt = "Models loaded" if ok else h.get("status", "initializing")
        st.markdown(f'<div><span class="dot {dot}"></span><small>{txt}</small></div>', unsafe_allow_html=True)
        st.caption(f"{st.session_state.api_base}")

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
    st.caption(f"Session · {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ─────────────────────────────────────────────────────────────────
# CUSTOMER INPUT FORM (reusable)
# ─────────────────────────────────────────────────────────────────
OPTIONS = {
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

def render_customer_form(key_prefix: str = "form") -> Dict[str, Any]:
    form = st.session_state.customer_form.copy()
    with st.container():
        tabs = st.tabs(["Profile", "Contract & Services", "Billing"])

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            form["gender"]         = c1.selectbox("Gender", OPTIONS["gender"], index=OPTIONS["gender"].index(form["gender"]), key=f"{key_prefix}_gender")
            form["SeniorCitizen"]  = c2.selectbox("Senior citizen", OPTIONS["SeniorCitizen"], index=OPTIONS["SeniorCitizen"].index(form["SeniorCitizen"]), key=f"{key_prefix}_senior")
            form["Partner"]        = c3.selectbox("Partner", OPTIONS["Partner"], index=OPTIONS["Partner"].index(form["Partner"]), key=f"{key_prefix}_partner")
            form["Dependents"]     = c4.selectbox("Dependents", OPTIONS["Dependents"], index=OPTIONS["Dependents"].index(form["Dependents"]), key=f"{key_prefix}_deps")

            c1, c2, c3 = st.columns(3)
            form["tenure"]         = c1.number_input("Tenure (months)", 0, 120, int(form["tenure"]), key=f"{key_prefix}_ten")
            form["MonthlyCharges"] = c2.number_input("Monthly charges", 0.0, 500.0, float(form["MonthlyCharges"]), step=1.0, key=f"{key_prefix}_mc")
            form["TotalCharges"]   = c3.number_input("Total charges", 0.0, 50000.0, float(form["TotalCharges"]), step=10.0, key=f"{key_prefix}_tc")

        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            form["Contract"]       = c1.selectbox("Contract", OPTIONS["Contract"], index=OPTIONS["Contract"].index(form["Contract"]), key=f"{key_prefix}_con")
            form["InternetService"]= c2.selectbox("Internet service", OPTIONS["InternetService"], index=OPTIONS["InternetService"].index(form["InternetService"]), key=f"{key_prefix}_net")
            form["PhoneService"]   = c3.selectbox("Phone service", OPTIONS["PhoneService"], index=OPTIONS["PhoneService"].index(form["PhoneService"]), key=f"{key_prefix}_phone")

            c1, c2, c3 = st.columns(3)
            form["MultipleLines"]  = c1.selectbox("Multiple lines", OPTIONS["MultipleLines"], index=OPTIONS["MultipleLines"].index(form["MultipleLines"]), key=f"{key_prefix}_ml")
            form["OnlineSecurity"] = c2.selectbox("Online security", OPTIONS["OnlineSecurity"], index=OPTIONS["OnlineSecurity"].index(form["OnlineSecurity"]), key=f"{key_prefix}_os")
            form["OnlineBackup"]   = c3.selectbox("Online backup", OPTIONS["OnlineBackup"], index=OPTIONS["OnlineBackup"].index(form["OnlineBackup"]), key=f"{key_prefix}_ob")

            c1, c2, c3 = st.columns(3)
            form["DeviceProtection"]= c1.selectbox("Device protection", OPTIONS["DeviceProtection"], index=OPTIONS["DeviceProtection"].index(form["DeviceProtection"]), key=f"{key_prefix}_dp")
            form["TechSupport"]    = c2.selectbox("Tech support", OPTIONS["TechSupport"], index=OPTIONS["TechSupport"].index(form["TechSupport"]), key=f"{key_prefix}_ts")
            form["StreamingTV"]    = c3.selectbox("Streaming TV", OPTIONS["StreamingTV"], index=OPTIONS["StreamingTV"].index(form["StreamingTV"]), key=f"{key_prefix}_stv")

            form["StreamingMovies"]= st.selectbox("Streaming movies", OPTIONS["StreamingMovies"], index=OPTIONS["StreamingMovies"].index(form["StreamingMovies"]), key=f"{key_prefix}_sm")

        with tabs[2]:
            c1, c2 = st.columns(2)
            form["PaperlessBilling"]= c1.selectbox("Paperless billing", OPTIONS["PaperlessBilling"], index=OPTIONS["PaperlessBilling"].index(form["PaperlessBilling"]), key=f"{key_prefix}_pb")
            form["PaymentMethod"]  = c2.selectbox("Payment method", OPTIONS["PaymentMethod"], index=OPTIONS["PaymentMethod"].index(form["PaymentMethod"]), key=f"{key_prefix}_pm")

    st.session_state.customer_form = form
    return form


# ═════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════
def page_overview():
    st.markdown("# Retention Intelligence Overview")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">A unified view of model health, customer segments, and global churn drivers.</p>', unsafe_allow_html=True)

    info = client.model_info()
    gi = client.global_importance(10)

    has_info = "model_info" in info if isinstance(info, dict) else False

    # Top strip — KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        auc = info.get("model_info", {}).get("churn_model", {}).get("performance_metrics", {}).get("roc_auc", 0) if has_info else 0
        metric_card("Model ROC-AUC", f"{auc:.3f}" if auc else "—", "LightGBM · Test split", "up")
    with c2:
        thr = info.get("model_info", {}).get("churn_model", {}).get("decision_threshold", 0) if has_info else 0
        metric_card("Decision Threshold", f"{thr:.3f}" if thr else "—", "Optimised for F1")
    with c3:
        segs = info.get("model_info", {}).get("segmentation_model", {}).get("num_clusters", 0) if has_info else 0
        metric_card("Customer Segments", f"{segs}", "K-Means clustering")
    with c4:
        n = info.get("model_info", {}).get("churn_model", {}).get("training_data_size", 0) if has_info else 0
        metric_card("Training Cohort", f"{n:,}" if n else "—", "historical customers")

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Main grid — segments + drivers
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Global Churn Drivers")
        if isinstance(gi, dict) and gi.get("success") and gi.get("explanation"):
            feats = gi["explanation"]["top_features"]
            df = pd.DataFrame(feats).head(10)
            df = df.sort_values("importance")
            fig = go.Figure(go.Bar(
                x=df["importance"], y=df["feature_name"],
                orientation="h",
                marker=dict(color="#7dd3c0", line=dict(color="#243242", width=1)),
                text=[f"{v:.3f}" for v in df["importance"]],
                textposition="outside", textfont=dict(color="#a6b3c0", size=10),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Method: {gi['explanation'].get('explainer_type','shap')} · sample = {gi['explanation'].get('sample_size',0)}")
        else:
            st.info("Global feature importance unavailable — start the API or check SHAP initialisation.")

    with right:
        st.markdown("### Segment Library")
        if has_info:
            segs = info["model_info"].get("segments", {})
            for sid, meta in segs.items():
                st.markdown(
                    f'<div class="cris-card cris-card-tight" style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><div class="cris-label">Segment {sid}</div>'
                    f'<div style="color:#e6edf3;font-size:1rem;margin-top:2px">{meta.get("name","")}</div></div>'
                    f'{chip("cluster", "accent")}</div>'
                    f'<div style="color:#6c7a89;font-size:0.85rem;margin-top:8px">{meta.get("description","")}</div>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.info("Model info unavailable")

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Performance panel
    if has_info:
        pm = info["model_info"]["churn_model"].get("performance_metrics", {})
        st.markdown("### Model Performance")
        cols = st.columns(5)
        for i, (k, label) in enumerate([("roc_auc","ROC-AUC"),("accuracy","Accuracy"),("precision","Precision"),("recall","Recall"),("f1","F1 Score")]):
            with cols[i]:
                v = pm.get(k, 0)
                metric_card(label, f"{v:.3f}" if v else "—")


# ═════════════════════════════════════════════════════════════════
# PAGE: FORMULAS & DEFINITIONS
# ═════════════════════════════════════════════════════════════════
def page_formulas():
    st.markdown("# Formulas & Definitions")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Every calculation used across the CRIS dashboard, in one place. Useful as a reference for analysts, PMs, and reviewers.</p>', unsafe_allow_html=True)

    def formula_card(title: str, expr: str, notes: str = ""):
        notes_html = f'<div style="color:#6c7a89;font-size:0.82rem;margin-top:8px;line-height:1.5">{notes}</div>' if notes else ""
        st.markdown(
            f'<div class="cris-card" style="margin-bottom:10px">'
            f'<div class="cris-label">{title}</div>'
            f'<div style="color:#e6edf3;font-family:ui-monospace,Menlo,Consolas,monospace;'
            f'font-size:0.92rem;margin-top:6px;line-height:1.65;white-space:pre-wrap">{expr}</div>'
            f'{notes_html}'
            f'</div>', unsafe_allow_html=True)

    # Quick navigation
    st.markdown(
        '<div style="margin:8px 0 18px 0">'
        + chip("Churn & Risk", "accent")
        + chip("Explainability (SHAP)", "accent")
        + chip("Priority & Action", "warn")
        + chip("Revenue Economics", "ok")
        + chip("Model Performance", "accent")
        + '</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)

    with fc1:
        st.markdown("#### Churn & Risk")
        formula_card(
            "Churn probability",
            "p = σ(f_LGBM(x))   ∈ [0, 1]",
            "Output of the LightGBM churn classifier. x = 19 customer features + segment label, after feature engineering and preprocessing."
        )
        formula_card(
            "Churner flag",
            "is_churner = (p ≥ τ)\nτ = 0.4356   (F1-optimised threshold)",
            "τ is chosen on validation data to maximise F1. Returned per request as `threshold`."
        )
        formula_card(
            "Risk band",
            "High   if p ≥ 0.65\nMedium if 0.35 ≤ p < 0.65\nLow    if p < 0.35",
            "Used in Action Planning for fast filtering. Independent of the churner flag."
        )
        formula_card(
            "Segment confidence (K-Means)",
            "confidence = 1 − d₁ / d₂",
            "d₁ = distance to assigned cluster centroid, d₂ = distance to second-closest centroid. 1 ≈ unambiguous; 0 ≈ tied between two clusters."
        )
        formula_card(
            "Decision delta (What-If)",
            "Δp = p_modified − p_original",
            "Δp < 0 ⇒ intervention reduces churn risk (improvement). Δp > 0 ⇒ worsening. Segment / churner-flag transitions are reported separately."
        )

        st.markdown("#### Explainability (SHAP)")
        formula_card(
            "Per-instance contribution",
            "p̂(x) = φ₀ + Σⱼ φⱼ(x)",
            "φ₀ = base value (expected model output over background). φⱼ = SHAP value of feature j. Positive φⱼ pushes the prediction toward churn; negative φⱼ pulls it toward retention."
        )
        formula_card(
            "Impact direction",
            "direction = 'increases_churn' if φⱼ > 0\n            'decreases_churn' if φⱼ < 0\n            'neutral'          otherwise",
        )
        formula_card(
            "Global feature importance",
            "I_j = (1/N) · Σᵢ |φⱼ(xᵢ)|",
            "Mean absolute SHAP value across N background samples. Ranked descending for the Global Drivers chart."
        )
        formula_card(
            "Batch global importance",
            "Ī_j = (1/N) · Σᵢ |φⱼ(xᵢ)|\nfreq_j = #{ i : j ∈ top-5 of xᵢ }",
            "Computed only over rows processed in the current batch. Frequency shows how often a feature is pivotal per customer."
        )

    with fc2:
        st.markdown("#### Priority & Action")
        formula_card(
            "Composite priority score",
            "Priority = 0.40 · p\n         + 0.30 · segment_risk\n         + 0.30 · customer_value\nPriority ∈ [0, 1]   (clamped)",
            "Used by the Action Planning page to rank outreach. Replaces the API's rule-based priority for the in-dashboard ranking."
        )
        formula_card(
            "Segment risk (batch-derived)",
            "segment_risk_s = mean( p_i  :  segment_i = s )",
            "Average churn probability among customers assigned to segment s within the currently scored batch."
        )
        formula_card(
            "Customer value (batch-normalised)",
            "customer_value = (MC − MC_min) / (MC_max − MC_min)",
            "Min-max normalisation of MonthlyCharges across the batch → 0..1."
        )
        formula_card(
            "Priority bands",
            "High   if Priority ≥ 0.67\nMedium if 0.34 ≤ Priority < 0.67\nLow    if Priority < 0.34",
        )

        st.markdown("#### Revenue Economics")
        formula_card(
            "Portfolio MRR",
            "MRR = Σᵢ MonthlyChargesᵢ",
            "Sum of monthly recurring revenue across all scored customers."
        )
        formula_card(
            "MRR at risk",
            "MRR_risk = Σᵢ MonthlyChargesᵢ · 𝟙(is_churnerᵢ)",
            "Monthly revenue tied to customers predicted as churners (hard threshold)."
        )
        formula_card(
            "Expected monthly loss",
            "E[loss]_monthly = Σᵢ pᵢ · MonthlyChargesᵢ",
            "Probability-weighted monthly revenue exposure — smoother than MRR at risk; captures medium-risk customers too."
        )
        formula_card(
            "ARR at risk",
            "ARR_risk = MRR_risk · 12",
            "Annualised exposure, assuming monthly charges remain stable."
        )
        formula_card(
            "Risk share (per segment)",
            "share_s = MRR_risk(s) / MRR_risk(total)",
            "Proportion of the at-risk revenue concentrated in segment s — drives the donut on the Batch page."
        )
        formula_card(
            "Segment churn rate",
            "churn_rate_s = #{churners in s} / #{customers in s}",
        )

        st.markdown("#### Model Performance")
        formula_card(
            "Classification metrics",
            "Precision = TP / (TP + FP)\nRecall    = TP / (TP + FN)\nF1        = 2 · P · R / (P + R)\nAccuracy  = (TP + TN) / N\nROC-AUC   = P( p_pos > p_neg )",
            "Reported on the hold-out test split. ROC-AUC is threshold-independent; the others use τ = 0.4356."
        )


# ═════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER ANALYSIS
# ═════════════════════════════════════════════════════════════════
def page_customer():
    st.markdown("# Customer Analysis")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Score an individual customer, inspect SHAP drivers, and review the business-rule recommendation.</p>', unsafe_allow_html=True)

    with st.form("single_customer_form", clear_on_submit=False):
        customer = render_customer_form("single")
        c1, c2, _ = st.columns([1, 1, 3])
        run = c1.form_submit_button("Score customer", type="primary", use_container_width=True)
        clear = c2.form_submit_button("Reset", use_container_width=True)

    if clear:
        st.session_state.last_prediction = None

    if run:
        with st.spinner("Scoring…"):
            resp = client.predict(customer, return_features=True)
        st.session_state.last_prediction = resp

    resp = st.session_state.last_prediction
    if not resp:
        st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
        st.info("Submit a customer profile to view predictions.")
        return

    if "__error__" in resp:
        st.error(f"{resp['__error__']}: {resp.get('detail','')}"); return
    if not resp.get("success"):
        st.error(resp.get("error", "Prediction failed")); return

    pred = resp["prediction"]
    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Headline row
    left, mid, right = st.columns([1.1, 1, 1.3])
    with left:
        st.plotly_chart(gauge(pred["churn_probability"], "Churn Probability", pred["threshold"]),
                        use_container_width=True)
        tone = "danger" if pred["is_churner"] else "ok"
        label = "AT RISK" if pred["is_churner"] else "RETAINED"
        thr_chip = chip(f"threshold · {pred['threshold']:.3f}", "accent")
        st.markdown(
            f'<div style="text-align:center;margin-top:-10px">{chip(label, tone)}{thr_chip}</div>',
            unsafe_allow_html=True)

    with mid:
        st.markdown(
            f'<div class="cris-card">'
            f'<div class="cris-label">Segment</div>'
            f'<div class="cris-value" style="font-size:1.4rem">{pred["segment_label"]}</div>'
            f'<div class="cris-delta">Cluster #{pred["segment"]} · confidence {pred["segment_confidence"]:.2%}</div>'
            f'</div>', unsafe_allow_html=True)

        act = pred.get("recommended_action") or {}
        pr = act.get("priority_score", 0)
        tone = "danger" if pr > 0.7 else ("warn" if pr > 0.4 else "ok")
        st.markdown(
            f'<div class="cris-card" style="margin-top:10px">'
            f'<div class="cris-label">Recommended Action</div>'
            f'<div class="cris-value" style="font-size:1.15rem">{act.get("action_label","—")}</div>'
            f'<div class="cris-delta">Priority {chip(f"{pr:.2f}", tone)}</div>'
            f'<div style="color:#6c7a89;font-size:0.82rem;margin-top:8px;line-height:1.4">{act.get("reason","")}</div>'
            f'</div>', unsafe_allow_html=True)

    with right:
        st.markdown("### Top SHAP drivers")
        feats = pred.get("top_features") or []
        if feats:
            fig = shap_bar(feats, title=None)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            df = pd.DataFrame(feats)[["feature_name","feature_value","shap_value","impact_direction"]]
            df.columns = ["Feature", "Value", "SHAP", "Impact"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("SHAP explanation unavailable for this prediction.")

    # Engineered features expander
    with st.expander("Engineered features & raw response"):
        ef = pred.get("engineered_features")
        if ef:
            st.json(ef, expanded=False)
        st.caption("Full API response")
        st.json(resp, expanded=False)


# ═════════════════════════════════════════════════════════════════
# PAGE: BATCH SCORING
# ═════════════════════════════════════════════════════════════════
def page_batch():
    st.markdown("# Batch Scoring")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Upload a CSV of up to 50,000 customers. CRIS returns predictions, segment mix, action plan, and global drivers.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Upload customer CSV", type=["csv"], label_visibility="collapsed")
    with c2:
        tpl = client.batch_template()
        if isinstance(tpl, dict) and "template" in tpl:
            st.download_button("Download CSV template", data=tpl["template"],
                               file_name="cris_template.csv", mime="text/csv",
                               use_container_width=True)

    col_run, _ = st.columns([1, 4])
    if up and col_run.button("Run batch scoring", type="primary", use_container_width=True):
        with st.spinner(f"Scoring {up.name}…"):
            t0 = time.time()
            resp = client.predict_batch(up.getvalue(), up.name)
            elapsed = time.time() - t0
        resp["_elapsed"] = elapsed
        st.session_state.last_batch = resp

    resp = st.session_state.last_batch
    if not resp:
        st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
        st.info("Upload a CSV to begin.")
        return

    if "__error__" in resp:
        st.error(f"{resp['__error__']}: {resp.get('detail','')}"); return
    if not resp.get("success"):
        st.error(resp.get("message", "Batch failed")); return

    status = resp["status"]
    preds  = resp["predictions"]

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Rows processed", f"{status['rows_processed']:,}", f"of {status['total_rows']:,}")
    with c2: metric_card("Churn rate", f"{status['churn_rate']*100:.1f}%", "predicted churners")
    with c3: metric_card("Avg churn prob.", f"{status['avg_churn_probability']*100:.1f}%")
    with c4: metric_card("Avg segment conf.", f"{status['avg_segment_confidence']*100:.1f}%")
    with c5: metric_card("Latency", f"{resp.get('_elapsed',0):.1f}s", "end-to-end")

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Distributions
    left, right = st.columns(2)
    with left:
        st.markdown("### Segment distribution")
        seg_dist = status.get("segment_distribution", {})
        if seg_dist:
            df = pd.DataFrame([{"segment": f"Segment {k}", "count": v} for k, v in seg_dist.items()])
            fig = px.pie(df, names="segment", values="count", hole=0.65)
            fig.update_traces(marker=dict(line=dict(color="#0b0f14", width=2)),
                              textfont=dict(color="#e6edf3"))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True,
                              legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Recommended actions")
        act_dist = status.get("action_distribution", {})
        if act_dist:
            df = pd.DataFrame([{"action": k, "count": v} for k, v in act_dist.items()]).sort_values("count")
            fig = go.Figure(go.Bar(x=df["count"], y=df["action"], orientation="h",
                                   marker=dict(color="#d4a574", line=dict(color="#243242", width=1)),
                                   text=df["count"], textposition="outside",
                                   textfont=dict(color="#a6b3c0")))
            fig.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)

    # Global drivers
    tf = resp.get("top_features_global")
    if tf:
        st.markdown("### Global drivers across batch")
        df = pd.DataFrame(tf).sort_values("avg_shap_value")
        fig = go.Figure(go.Bar(
            x=df["avg_shap_value"], y=df["feature_name"], orientation="h",
            marker=dict(color="#7dd3c0", line=dict(color="#243242", width=1)),
            text=[f"n={int(f)}" for f in df["frequency"]], textposition="outside",
            textfont=dict(color="#a6b3c0"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Revenue at Risk & Segment Intelligence ─────────────────
    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
    st.markdown("### Revenue at Risk")

    # Build a per-row analytical frame
    ri = []
    for p in preds:
        f = p.get("input_features", {}) or {}
        mc = float(f.get("MonthlyCharges") or 0)
        tc = float(f.get("TotalCharges") or 0)
        tenure = int(f.get("tenure") or 0)
        prob = float(p.get("churn_probability", 0))
        ri.append({
            "segment": p.get("segment_label", "—"),
            "segment_id": p.get("segment", -1),
            "is_churner": bool(p.get("is_churner")),
            "churn_prob": prob,
            "monthly": mc,
            "total": tc,
            "tenure": tenure,
            "action": (p.get("recommended_action") or {}).get("action_label", "—"),
            "priority": float((p.get("recommended_action") or {}).get("priority_score", 0)),
            "expected_loss": prob * mc,                  # risk-weighted monthly revenue
            "arr_at_risk": mc * 12 if p.get("is_churner") else 0.0,  # ARR exposed if churner
        })
    rdf = pd.DataFrame(ri)

    total_monthly = rdf["monthly"].sum()
    exp_monthly_loss = rdf["expected_loss"].sum()
    mrr_at_risk = rdf.loc[rdf["is_churner"], "monthly"].sum()
    arr_at_risk = mrr_at_risk * 12
    avg_customer_value = rdf["monthly"].mean() if len(rdf) else 0
    pct_mrr_risk = (mrr_at_risk / total_monthly * 100) if total_monthly else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Portfolio MRR", f"${total_monthly:,.0f}", "all customers")
    with c2: metric_card("MRR at risk", f"${mrr_at_risk:,.0f}",
                         f"{pct_mrr_risk:.1f}% of portfolio",
                         "down" if pct_mrr_risk > 15 else "")
    with c3: metric_card("Expected monthly loss", f"${exp_monthly_loss:,.0f}",
                         "probability-weighted", "down")
    with c4: metric_card("ARR at risk", f"${arr_at_risk:,.0f}", "annualised exposure", "down")
    with c5: metric_card("Avg customer value", f"${avg_customer_value:,.2f}", "MRR per customer")

    # Segment-wise revenue at risk
    st.markdown("### Segment-wise Revenue at Risk")
    seg_agg = rdf.groupby("segment").agg(
        customers=("segment", "size"),
        churners=("is_churner", "sum"),
        mrr=("monthly", "sum"),
        mrr_at_risk=("monthly", lambda s: s[rdf.loc[s.index, "is_churner"]].sum()),
        expected_loss=("expected_loss", "sum"),
        avg_prob=("churn_prob", "mean"),
        avg_tenure=("tenure", "mean"),
        avg_monthly=("monthly", "mean"),
        avg_priority=("priority", "mean"),
    ).reset_index()
    seg_agg["churn_rate"] = seg_agg["churners"] / seg_agg["customers"].replace(0, 1)
    seg_agg["arr_at_risk"] = seg_agg["mrr_at_risk"] * 12
    seg_agg["risk_share"] = (seg_agg["mrr_at_risk"] /
                             seg_agg["mrr_at_risk"].sum()) if seg_agg["mrr_at_risk"].sum() else 0

    left, right = st.columns([1.2, 1])
    with left:
        # Horizontal stacked bar: safe vs at-risk MRR per segment
        safe_mrr = seg_agg["mrr"] - seg_agg["mrr_at_risk"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=seg_agg["segment"], x=safe_mrr, orientation="h",
            name="Safe MRR", marker=dict(color="#3e6e66",
                                         line=dict(color="#243242", width=1)),
            text=[f"${v:,.0f}" for v in safe_mrr], textposition="inside",
            textfont=dict(color="#c7eee5", size=10),
        ))
        fig.add_trace(go.Bar(
            y=seg_agg["segment"], x=seg_agg["mrr_at_risk"], orientation="h",
            name="MRR at risk", marker=dict(color="#c97a7a",
                                            line=dict(color="#243242", width=1)),
            text=[f"${v:,.0f}" for v in seg_agg["mrr_at_risk"]],
            textposition="inside", textfont=dict(color="#fbe6e6", size=10),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode="stack",
                          legend=dict(orientation="h", y=-0.15),
                          title=dict(text="MRR composition by segment",
                                     font=dict(color="#a6b3c0", size=12)))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        # Risk-share donut
        fig = go.Figure(go.Pie(
            labels=seg_agg["segment"], values=seg_agg["mrr_at_risk"],
            hole=0.65,
            marker=dict(line=dict(color="#0b0f14", width=2)),
            textinfo="label+percent",
            textfont=dict(color="#e6edf3", size=11),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text="Share of MRR at risk",
                                     font=dict(color="#a6b3c0", size=12)),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Segment-wise Intelligence table
    st.markdown("### Segment-wise Intelligence")

    # Top action per segment
    top_action = (rdf.groupby(["segment", "action"])
                     .size().reset_index(name="n")
                     .sort_values(["segment", "n"], ascending=[True, False])
                     .drop_duplicates("segment"))
    top_action_map = dict(zip(top_action["segment"], top_action["action"]))

    intel = pd.DataFrame({
        "Segment":          seg_agg["segment"],
        "Customers":        seg_agg["customers"].astype(int),
        "Churners":         seg_agg["churners"].astype(int),
        "Churn rate":       [f"{v*100:.1f}%" for v in seg_agg["churn_rate"]],
        "Avg prob.":        [f"{v*100:.1f}%" for v in seg_agg["avg_prob"]],
        "Avg tenure":       [f"{v:.1f} mo" for v in seg_agg["avg_tenure"]],
        "Avg MRR":          [f"${v:,.2f}" for v in seg_agg["avg_monthly"]],
        "Segment MRR":      [f"${v:,.0f}" for v in seg_agg["mrr"]],
        "MRR at risk":      [f"${v:,.0f}" for v in seg_agg["mrr_at_risk"]],
        "ARR at risk":      [f"${v:,.0f}" for v in seg_agg["arr_at_risk"]],
        "Risk share":       [f"{v*100:.1f}%" for v in seg_agg["risk_share"]],
        "Avg priority":     [f"{v:.2f}" for v in seg_agg["avg_priority"]],
        "Top action":       [top_action_map.get(s, "—") for s in seg_agg["segment"]],
    })
    st.dataframe(intel, use_container_width=True, hide_index=True)

    # Segment cards — narrative intelligence
    st.markdown("#### Segment highlights")
    seg_rows = seg_agg.sort_values("mrr_at_risk", ascending=False)
    cards_per_row = 2
    seg_list = seg_rows.to_dict("records")
    for i in range(0, len(seg_list), cards_per_row):
        cols = st.columns(cards_per_row)
        for j, s in enumerate(seg_list[i:i + cards_per_row]):
            with cols[j]:
                churn_rate = s["churn_rate"]
                tone = "danger" if churn_rate > 0.4 else ("warn" if churn_rate > 0.2 else "ok")
                action = top_action_map.get(s["segment"], "—")
                st.markdown(
                    f'<div class="cris-card" style="height:100%">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div><div class="cris-label">Segment</div>'
                    f'<div style="color:#e6edf3;font-size:1.05rem">{s["segment"]}</div></div>'
                    f'<div>{chip(f"{churn_rate*100:.0f}% churn", tone)}</div></div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px">'
                    f'<div><div class="cris-label">Customers</div><div style="color:#e6edf3">{int(s["customers"]):,}</div></div>'
                    f'<div><div class="cris-label">Churners</div><div style="color:#e6edf3">{int(s["churners"]):,}</div></div>'
                    f'<div><div class="cris-label">MRR</div><div style="color:#e6edf3">${s["mrr"]:,.0f}</div></div>'
                    f'<div><div class="cris-label">MRR at risk</div><div style="color:#c97a7a">${s["mrr_at_risk"]:,.0f}</div></div>'
                    f'<div><div class="cris-label">Avg tenure</div><div style="color:#e6edf3">{s["avg_tenure"]:.1f} mo</div></div>'
                    f'<div><div class="cris-label">Avg priority</div><div style="color:#e6edf3">{s["avg_priority"]:.2f}</div></div>'
                    f'</div>'
                    f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid #243242">'
                    f'<div class="cris-label">Recommended focus</div>'
                    f'<div style="color:#a6b3c0;font-size:0.9rem;margin-top:4px">{action}</div>'
                    f'</div></div>', unsafe_allow_html=True)

    # Per-row table
    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
    st.markdown("### Per-customer predictions")
    rows = []
    for i, p in enumerate(preds):
        rows.append({
            "#": i + 1,
            "Segment": p.get("segment_label", ""),
            "Churn prob.": round(p.get("churn_probability", 0), 4),
            "Churner": "●" if p.get("is_churner") else "○",
            "Action": (p.get("recommended_action") or {}).get("action_label", ""),
            "Priority": round((p.get("recommended_action") or {}).get("priority_score", 0), 3),
            "Tenure": p.get("input_features", {}).get("tenure"),
            "Monthly": p.get("input_features", {}).get("MonthlyCharges"),
            "Contract": p.get("input_features", {}).get("Contract"),
        })
    df = pd.DataFrame(rows)

    f1, f2 = st.columns([1, 1])
    churner_only = f1.checkbox("Show only predicted churners", value=False)
    min_prio = f2.slider("Min. priority", 0.0, 1.0, 0.0, 0.05)

    view = df.copy()
    if churner_only: view = view[view["Churner"] == "●"]
    view = view[view["Priority"] >= min_prio]

    st.dataframe(view, use_container_width=True, hide_index=True, height=400)
    csv = df.to_csv(index=False).encode()
    st.download_button("Export predictions as CSV", csv, "cris_predictions.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════
# PAGE: ACTION PLANNING
# ═════════════════════════════════════════════════════════════════
def page_actions():
    st.markdown("# Action Planning")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Filter, rank, and export targeted retention actions across the current batch.</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cris-card cris-card-tight" style="margin-bottom:6px">'
        '<div class="cris-label">Priority formula</div>'
        '<div style="color:#a6b3c0;font-size:0.9rem;margin-top:4px">'
        'Priority = <span style="color:#7dd3c0">0.40 · churn probability</span> '
        '+ <span style="color:#d4a574">0.30 · segment risk</span> '
        '+ <span style="color:#8fbc8f">0.30 · customer value</span>'
        '</div>'
        '<div style="color:#6c7a89;font-size:0.78rem;margin-top:4px">'
        'Segment risk = average churn probability within each segment in this batch · '
        'Customer value = MonthlyCharges min-max normalised across the batch'
        '</div></div>', unsafe_allow_html=True)

    batch = st.session_state.last_batch
    if not batch or not batch.get("success") or not batch.get("predictions"):
        st.info("No batch scored yet. Run **Batch Scoring** first to populate the action planner.")
        return

    preds = batch["predictions"]

    # ── Composite priority inputs ───────────────────────────────
    # Priority = 0.4 · churn_probability
    #         + 0.3 · segment_risk (segment's avg churn prob in this batch)
    #         + 0.3 · customer_value (MonthlyCharges, min-max normalised)
    seg_risk_map: Dict[int, float] = {}
    seg_groups: Dict[int, List[float]] = {}
    for p in preds:
        seg_groups.setdefault(int(p.get("segment", -1)), []).append(
            float(p.get("churn_probability", 0)))
    for sid, vals in seg_groups.items():
        seg_risk_map[sid] = sum(vals) / len(vals) if vals else 0.0

    mc_vals = [float((p.get("input_features") or {}).get("MonthlyCharges") or 0) for p in preds]
    mc_min, mc_max = (min(mc_vals), max(mc_vals)) if mc_vals else (0.0, 0.0)
    mc_range = (mc_max - mc_min) or 1.0

    # Build enriched frame
    rows = []
    for i, p in enumerate(preds):
        f = p.get("input_features", {}) or {}
        act = p.get("recommended_action") or {}
        tf = p.get("top_features") or []
        prob = float(p.get("churn_probability", 0))
        mc = float(f.get("MonthlyCharges") or 0)
        sid = int(p.get("segment", -1))

        seg_risk = seg_risk_map.get(sid, 0.0)
        cust_value = (mc - mc_min) / mc_range           # 0..1
        prio = 0.40 * prob + 0.30 * seg_risk + 0.30 * cust_value
        prio = max(0.0, min(1.0, prio))

        if prob >= 0.65: risk = "High"
        elif prob >= 0.35: risk = "Medium"
        else: risk = "Low"
        if prio >= 0.67: plevel = "High"
        elif prio >= 0.34: plevel = "Medium"
        else: plevel = "Low"
        rows.append({
            "#": i + 1,
            "Segment": p.get("segment_label", "—"),
            "SegmentId": int(p.get("segment", -1)),
            "Priority %": int(round(prio * 100)),
            "Priority": plevel,
            "Risk": risk,
            "Churn prob.": round(prob, 4),
            "Churner": bool(p.get("is_churner")),
            "Action": act.get("action_label", "—"),
            "Reason": act.get("reason", ""),
            "Seg. risk": round(seg_risk, 3),
            "Value norm.": round(cust_value, 3),
            "Monthly": mc,
            "MRR at risk": round(mc if p.get("is_churner") else 0, 2),
            "Expected loss": round(prob * mc, 2),
            "Tenure": int(f.get("tenure") or 0),
            "Contract": f.get("Contract", "—"),
            "Payment": f.get("PaymentMethod", "—"),
            "Internet": f.get("InternetService", "—"),
            "Seg. conf.": round(float(p.get("segment_confidence", 0)), 3),
            "Driver 1": tf[0]["feature_name"] if len(tf) > 0 else "",
            "Driver 2": tf[1]["feature_name"] if len(tf) > 1 else "",
            "Driver 3": tf[2]["feature_name"] if len(tf) > 2 else "",
        })
    df = pd.DataFrame(rows)

    # Quick preset strip
    st.markdown("### Focus presets")
    if "ap_preset" not in st.session_state: st.session_state.ap_preset = None
    preset_defs = [
        ("All customers",       {}),
        ("High-priority churners", {"priority": ["High"], "churner_only": True}),
        ("High-risk · monthly",    {"risk": ["High"], "contract": ["Month-to-month"]}),
        ("Silent defectors",       {"risk": ["Medium", "High"], "churner_only": False}),
        ("High-value at risk",     {"risk": ["High"], "min_monthly": 80.0}),
    ]
    pcols = st.columns(len(preset_defs))
    for i, (name, _) in enumerate(preset_defs):
        is_active = st.session_state.ap_preset == name
        label = f"● {name}" if is_active else name
        if pcols[i].button(label, key=f"ap_p_{i}", use_container_width=True):
            st.session_state.ap_preset = None if is_active else name
    preset = next((p for n, p in preset_defs if n == st.session_state.ap_preset), {})

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Filters
    st.markdown("### Filters")
    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        seg_options = sorted(df["Segment"].unique().tolist())
        sel_segments = st.multiselect("Segments", seg_options, default=seg_options, key="ap_seg")
    with f2:
        act_options = sorted(df["Action"].unique().tolist())
        sel_actions = st.multiselect("Actions", act_options, default=act_options, key="ap_act")
    with f3:
        sel_risk = st.multiselect("Risk band", ["Low", "Medium", "High"],
                                  default=preset.get("risk", ["Low", "Medium", "High"]), key="ap_risk")
    with f4:
        con_options = sorted(df["Contract"].unique().tolist())
        sel_contracts = st.multiselect("Contract", con_options,
                                       default=preset.get("contract", con_options), key="ap_con")
    with f5:
        sel_priority = st.multiselect("Priority level", ["High", "Medium", "Low"],
                                      default=preset.get("priority", ["High", "Medium", "Low"]),
                                      key="ap_prio")

    g1, g2, g3, g4 = st.columns([1, 1, 1, 1])
    with g1:
        min_prob = st.slider("Min. churn prob.", 0.0, 1.0, 0.0, 0.01, key="ap_minp")
    with g2:
        min_monthly = st.number_input("Min. MRR", 0.0, 1000.0,
                                      float(preset.get("min_monthly", 0.0)), 5.0, key="ap_minmrr")
    with g3:
        churner_default = preset.get("churner_only", False)
        churner_only = st.checkbox("Churners only", value=churner_default, key="ap_chonly")
    with g4:
        sort_by = st.selectbox("Sort by", [
            "Priority %", "Churn prob.", "Monthly", "Expected loss",
            "MRR at risk", "Tenure", "Seg. conf."
        ], key="ap_sort")

    # Apply filters
    view = df.copy()
    if sel_segments: view = view[view["Segment"].isin(sel_segments)]
    if sel_actions:  view = view[view["Action"].isin(sel_actions)]
    if sel_risk:     view = view[view["Risk"].isin(sel_risk)]
    if sel_contracts:view = view[view["Contract"].isin(sel_contracts)]
    if sel_priority: view = view[view["Priority"].isin(sel_priority)]
    if churner_only: view = view[view["Churner"]]
    view = view[view["Churn prob."] >= min_prob]
    view = view[view["Monthly"] >= min_monthly]
    ascending = sort_by == "Tenure"
    view = view.sort_values(sort_by, ascending=ascending)

    # Search
    s1, s2 = st.columns([1, 4])
    with s1:
        search_id = st.text_input("Customer # search", "", key="ap_search")
    if search_id.strip().isdigit():
        view = view[view["#"] == int(search_id)]

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # KPI strip (filtered)
    n_shown = len(view); n_total = len(df)
    n_high = int((view["Priority"] == "High").sum())
    n_churn = int(view["Churner"].sum())
    filt_mrr = view["Monthly"].sum()
    filt_risk_mrr = view.loc[view["Churner"], "Monthly"].sum()
    filt_exp = view["Expected loss"].sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: metric_card("In view", f"{n_shown:,}", f"of {n_total:,} customers")
    with k2: metric_card("High priority", f"{n_high:,}", "priority ≥ 0.67")
    with k3: metric_card("Predicted churners", f"{n_churn:,}")
    with k4: metric_card("MRR at risk", f"${filt_risk_mrr:,.0f}", f"filtered MRR ${filt_mrr:,.0f}")
    with k5: metric_card("Expected loss", f"${filt_exp:,.0f}", "prob-weighted monthly")

    # Priority × Risk breakdown
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("### Priority × Risk heatmap")
        if len(view):
            pr = (view.groupby(["Priority", "Risk"]).size()
                      .unstack(fill_value=0)
                      .reindex(index=["High", "Medium", "Low"],
                               columns=["High", "Medium", "Low"], fill_value=0))
            fig = go.Figure(go.Heatmap(
                z=pr.values, x=pr.columns, y=pr.index,
                text=pr.values, texttemplate="%{text}",
                textfont=dict(color="#0b0f14", size=13),
                colorscale=[[0, "#1e2935"], [0.5, "#3e6e66"], [1, "#7dd3c0"]],
                showscale=False,
            ))
            hm_layout = {**PLOTLY_LAYOUT,
                         "xaxis": dict(title="Risk band", gridcolor="#243242"),
                         "yaxis": dict(title="Priority", gridcolor="#243242")}
            fig.update_layout(**hm_layout, height=260)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Action mix")
        if len(view):
            mix = view["Action"].value_counts().reset_index()
            mix.columns = ["Action", "Count"]
            mix = mix.sort_values("Count")
            fig = go.Figure(go.Bar(
                x=mix["Count"], y=mix["Action"], orientation="h",
                marker=dict(color="#d4a574", line=dict(color="#243242", width=1)),
                text=mix["Count"], textposition="outside",
                textfont=dict(color="#a6b3c0"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig, use_container_width=True)

    # Main table
    st.markdown("### Customer list")
    show_cols = ["#", "Priority %", "Priority", "Risk", "Churn prob.", "Seg. risk",
                 "Value norm.", "Churner", "Segment", "Action", "Monthly",
                 "MRR at risk", "Expected loss", "Tenure", "Contract", "Payment",
                 "Internet", "Seg. conf.", "Driver 1", "Driver 2", "Driver 3"]
    show_cols = [c for c in show_cols if c in view.columns]

    disp = view[show_cols].copy()
    disp["Churner"] = disp["Churner"].map({True: "●", False: "○"})
    disp["Priority %"] = disp["Priority %"].astype(str) + "%"
    disp["Churn prob."] = (disp["Churn prob."] * 100).round(2).astype(str) + "%"
    disp["Monthly"] = disp["Monthly"].apply(lambda v: f"${v:,.2f}")
    disp["MRR at risk"] = disp["MRR at risk"].apply(lambda v: f"${v:,.2f}")
    disp["Expected loss"] = disp["Expected loss"].apply(lambda v: f"${v:,.2f}")

    st.dataframe(disp, use_container_width=True, hide_index=True, height=520)

    # Top-20 high-risk
    with st.expander("Top 20 highest-risk customers (entire batch)", expanded=False):
        top20 = df.sort_values(["Churner", "Priority %", "Churn prob."],
                               ascending=[False, False, False]).head(20)
        top20_disp = top20[show_cols].copy()
        top20_disp["Churner"] = top20_disp["Churner"].map({True: "●", False: "○"})
        top20_disp["Priority %"] = top20_disp["Priority %"].astype(str) + "%"
        top20_disp["Churn prob."] = (top20_disp["Churn prob."] * 100).round(2).astype(str) + "%"
        for c in ["Monthly", "MRR at risk", "Expected loss"]:
            top20_disp[c] = top20_disp[c].apply(lambda v: f"${v:,.2f}" if isinstance(v, (int, float)) else v)
        st.dataframe(top20_disp, use_container_width=True, hide_index=True)

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Customer lookup
    st.markdown("### Customer lookup")
    l1, l2 = st.columns([1, 4])
    with l1:
        lid = st.number_input("Customer #", 1, int(df["#"].max()) if len(df) else 1, 1, key="ap_lookup")
    with l2:
        do_lookup = st.button("Open profile", key="ap_lookup_btn")

    if do_lookup:
        row = df[df["#"] == lid]
        if row.empty:
            st.warning(f"Customer #{lid} not found.")
        else:
            r = row.iloc[0]
            c_left, c_right = st.columns([1, 2])
            with c_left:
                cid = int(r["#"])
                seg = r["Segment"]
                is_churner = bool(r["Churner"])
                tone = "danger" if is_churner else "ok"
                status_chip = chip("WILL CHURN" if is_churner else "RETAINED", tone)
                churn_pct = float(r["Churn prob."]) * 100
                seg_conf_pct = float(r["Seg. conf."]) * 100
                st.markdown(
                    f'<div class="cris-card" style="text-align:center">'
                    f'<div class="cris-label">Customer #{cid}</div>'
                    f'<div style="color:#e6edf3;font-size:1.15rem;margin:8px 0">{seg}</div>'
                    f'<div style="margin:8px 0">{status_chip}</div>'
                    f'<div class="cris-value" style="font-size:1.8rem">{churn_pct:.1f}%</div>'
                    f'<div class="cris-delta">churn probability · seg conf {seg_conf_pct:.1f}%</div>'
                    f'</div>', unsafe_allow_html=True)
            with c_right:
                plevel = r["Priority"]
                prio_tone = "danger" if plevel == "High" else ("warn" if plevel == "Medium" else "ok")
                risk_val = r["Risk"]
                risk_tone = "warn" if risk_val != "Low" else "accent"
                prio_chip = chip(f"priority · {plevel} ({r['Priority %']}%)", prio_tone)
                risk_chip = chip(f"risk · {risk_val}", risk_tone)
                action = r["Action"]; reason = r["Reason"]
                st.markdown(
                    f'<div class="cris-card">'
                    f'<div class="cris-label">Recommended action</div>'
                    f'<div style="color:#e6edf3;font-size:1.15rem;margin:6px 0">{action}</div>'
                    f'<div>{prio_chip}{risk_chip}</div>'
                    f'<div style="color:#6c7a89;font-size:0.9rem;margin-top:10px;line-height:1.5">{reason}</div>'
                    f'</div>', unsafe_allow_html=True)

                monthly = float(r["Monthly"]); tenure = int(r["Tenure"])
                contract = r["Contract"]; payment = r["Payment"]
                internet = r["Internet"]; exp_loss = float(r["Expected loss"])
                st.markdown(
                    f'<div class="cris-card" style="margin-top:10px">'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">'
                    f'<div><div class="cris-label">Monthly</div><div style="color:#e6edf3">${monthly:,.2f}</div></div>'
                    f'<div><div class="cris-label">Tenure</div><div style="color:#e6edf3">{tenure} mo</div></div>'
                    f'<div><div class="cris-label">Contract</div><div style="color:#e6edf3">{contract}</div></div>'
                    f'<div><div class="cris-label">Payment</div><div style="color:#e6edf3">{payment}</div></div>'
                    f'<div><div class="cris-label">Internet</div><div style="color:#e6edf3">{internet}</div></div>'
                    f'<div><div class="cris-label">Exp. loss</div><div style="color:#c97a7a">${exp_loss:,.2f}</div></div>'
                    f'</div></div>', unsafe_allow_html=True)

                drivers = [r.get("Driver 1", ""), r.get("Driver 2", ""), r.get("Driver 3", "")]
                drivers = [d for d in drivers if d]
                if drivers:
                    chips_html = "".join(chip(d, "accent") for d in drivers)
                    st.markdown(
                        f'<div class="cris-card" style="margin-top:10px">'
                        f'<div class="cris-label">Top SHAP drivers</div>'
                        f'<div style="margin-top:8px">{chips_html}</div>'
                        f'</div>', unsafe_allow_html=True)

            # Full raw profile
            raw = preds[int(r["#"]) - 1].get("input_features", {}) or {}
            prof = pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in raw.items()])
            with st.expander("Full customer profile"):
                st.dataframe(prof, use_container_width=True, hide_index=True)

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # Export
    st.markdown("### Export")
    e1, e2, _ = st.columns([1, 1, 3])
    with e1:
        csv_full = view.to_csv(index=False).encode()
        st.download_button("Filtered view (CSV)", csv_full,
                           "cris_action_plan_filtered.csv", "text/csv",
                           use_container_width=True)
    with e2:
        high_prio = df[df["Priority"] == "High"].to_csv(index=False).encode()
        st.download_button("All high-priority (CSV)", high_prio,
                           "cris_high_priority.csv", "text/csv",
                           use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF
# ═════════════════════════════════════════════════════════════════
def page_whatif():
    st.markdown("# What-If Simulation Lab")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Perturb customer attributes to quantify the retention impact of each decision.</p>', unsafe_allow_html=True)

    # ── 1. CUSTOMER SOURCE ──────────────────────────────────────
    st.markdown("### 1 · Customer source")
    mode = st.radio(
        "Input mode",
        ["Manual entry", "Load from last batch"],
        horizontal=True,
        label_visibility="collapsed",
        key="wif_mode",
    )

    customer: Dict[str, Any] = {}

    if mode == "Manual entry":
        customer = render_customer_form("wif")
    else:
        batch = st.session_state.last_batch
        if not batch or not batch.get("success") or not batch.get("predictions"):
            st.info("No batch data loaded. Run **Batch Scoring** first, or switch to Manual entry.")
            return

        preds = batch["predictions"]
        rows = []
        for i, p in enumerate(preds):
            f = p.get("input_features", {})
            rows.append({
                "#": i + 1,
                "Segment": p.get("segment_label", ""),
                "Churn prob.": round(p.get("churn_probability", 0), 4),
                "Tenure": f.get("tenure"),
                "Monthly": f.get("MonthlyCharges"),
                "Contract": f.get("Contract"),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=260)

        sel = st.number_input("Select customer #", 1, len(preds), 1, key="wif_sel")
        customer = {k: v for k, v in preds[sel - 1]["input_features"].items() if v is not None}
        sel_pred = preds[sel - 1]
        seg_chip = chip(sel_pred.get("segment_label", ""), "accent")
        churn_pct = sel_pred.get("churn_probability", 0) * 100
        prob_chip = chip(f"churn · {churn_pct:.1f}%", "warn")
        st.markdown(
            f'<div class="cris-card cris-card-tight">'
            f'<div class="cris-label">Selected customer</div>'
            f'<div style="color:#e6edf3">#{sel} · {seg_chip}{prob_chip}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # ── 2. POLICY TEMPLATES ─────────────────────────────────────
    st.markdown("### 2 · Policy templates")
    pol = client.policy_scenarios()
    scenarios = pol.get("scenarios", []) if isinstance(pol, dict) else []
    # Normalise known-bad category values coming back from the API's policy template
    _payment_fix = {"Bank transfer": "Bank transfer (automatic)",
                    "Credit card": "Credit card (automatic)"}
    for sc in scenarios:
        mods = sc.get("modifications") or {}
        if mods.get("PaymentMethod") in _payment_fix:
            mods["PaymentMethod"] = _payment_fix[mods["PaymentMethod"]]
    scenario_mods: Dict[str, Any] = {}

    if scenarios:
        tcols = st.columns(min(len(scenarios), 5))
        if "wif_active_scenario" not in st.session_state:
            st.session_state.wif_active_scenario = None
        for i, sc in enumerate(scenarios[:5]):
            with tcols[i]:
                is_active = st.session_state.wif_active_scenario == sc["name"]
                label = f"● {sc['name']}" if is_active else sc["name"]
                if st.button(label, key=f"wif_scen_{i}", use_container_width=True):
                    st.session_state.wif_active_scenario = None if is_active else sc["name"]
        if st.session_state.wif_active_scenario:
            active = next(s for s in scenarios if s["name"] == st.session_state.wif_active_scenario)
            scenario_mods = active["modifications"]
            st.markdown(
                f'<div class="cris-card cris-card-tight" style="margin-top:10px">'
                f'<div class="cris-label">{active["name"]}</div>'
                f'<div style="color:#a6b3c0;font-size:0.9rem">{active["description"]}</div>'
                f'<div style="margin-top:8px">'
                + "".join(chip(f"{k} → {v}", "warn") for k, v in scenario_mods.items())
                + '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # ── 3. MODIFICATIONS ────────────────────────────────────────
    st.markdown("### 3 · Scenario modifications")
    st.caption("Check a field to override it. Unchecked fields keep their original value.")

    mod_fields = [
        "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
        "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies",
        "DeviceProtection", "OnlineBackup", "tenure", "MonthlyCharges",
    ]
    modifications: Dict[str, Any] = {}

    cols_l, cols_r = st.columns(2)
    columns_map = [cols_l, cols_r]
    for i, f in enumerate(mod_fields):
        target = columns_map[i % 2]
        with target:
            row = st.container()
            c_chk, c_val = row.columns([1, 2])
            default_on = f in scenario_mods
            with c_chk:
                on = st.checkbox(f, value=default_on, key=f"wif_on_{f}")
            with c_val:
                if on:
                    if f in OPTIONS:
                        opts = OPTIONS[f]
                        default = scenario_mods.get(f, customer.get(f, opts[0]))
                        try: idx = opts.index(default)
                        except: idx = 0
                        modifications[f] = st.selectbox(
                            f"{f}_new", opts, index=idx,
                            key=f"wif_v_{f}", label_visibility="collapsed")
                    elif f == "tenure":
                        modifications[f] = st.number_input(
                            f"{f}_new", 0, 120,
                            int(scenario_mods.get(f, customer.get(f, 12))),
                            key=f"wif_v_{f}", label_visibility="collapsed")
                    elif f == "MonthlyCharges":
                        modifications[f] = st.number_input(
                            f"{f}_new", 0.0, 500.0,
                            float(scenario_mods.get(f, customer.get(f, 70))),
                            key=f"wif_v_{f}", label_visibility="collapsed")

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)

    # ── 4. SIMULATE ─────────────────────────────────────────────
    c_run, c_clr, _ = st.columns([1, 1, 3])
    if c_run.button("Simulate scenario", type="primary", use_container_width=True):
        if not customer:
            st.warning("Provide customer data first.")
        elif not modifications:
            st.warning("Select at least one modification.")
        else:
            with st.spinner("Running simulation…"):
                resp = client.what_if(customer, modifications)
            st.session_state.last_whatif = {"resp": resp, "customer": customer}
    if c_clr.button("Clear result", use_container_width=True):
        st.session_state.last_whatif = None

    state = st.session_state.last_whatif
    if not state:
        return
    resp = state["resp"]
    original_customer = state["customer"]

    if "__error__" in resp:
        st.error(f"{resp['__error__']}: {resp.get('detail','')}"); return
    if not resp.get("success"):
        st.error(resp.get("error", "Simulation failed")); return

    o = resp["original_prediction"]
    m = resp["modified_prediction"]
    d = resp["delta"]

    st.markdown('<div class="cris-hr"></div>', unsafe_allow_html=True)
    st.markdown("### Scenario comparison")

    # Three-column side-by-side: Original / Delta / Modified
    c1, c2, c3 = st.columns([1, 1, 1])

    orig_tone = "danger" if o["is_churner"] else "ok"
    mod_tone  = "danger" if m["is_churner"] else "ok"
    orig_status = "WILL CHURN" if o["is_churner"] else "RETAINED"
    mod_status  = "WILL CHURN" if m["is_churner"] else "RETAINED"

    with c1:
        st.markdown('<div class="cris-label" style="text-align:center;letter-spacing:0.15em">ORIGINAL</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge(o["churn_probability"], "", o["threshold"]), use_container_width=True)
        st.markdown(
            f'<div style="text-align:center">'
            f'{chip(o["segment_label"], "accent")}'
            f'{chip(orig_status, orig_tone)}'
            f'</div>'
            f'<div class="cris-card cris-card-tight" style="margin-top:10px">'
            f'<div class="cris-label">Churn probability</div>'
            f'<div class="cris-value" style="font-size:1.5rem">{o["churn_probability"]*100:.2f}%</div>'
            f'<div class="cris-delta">confidence {o["segment_confidence"]*100:.1f}%</div>'
            f'</div>', unsafe_allow_html=True)

    with c2:
        delta = d["churn_probability_delta"]
        arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "•")
        dcolor = "#8fbc8f" if delta < 0 else ("#c97a7a" if delta > 0 else "#a6b3c0")
        dlabel = "IMPROVEMENT" if delta < 0 else ("WORSENING" if delta > 0 else "NO CHANGE")
        dtone = "ok" if delta < 0 else ("danger" if delta > 0 else "accent")

        st.markdown('<div class="cris-label" style="text-align:center;letter-spacing:0.15em">DELTA</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="cris-card" style="text-align:center;margin-top:8px">'
            f'<div style="font-size:0.72rem;letter-spacing:0.12em;color:{dcolor}">{dlabel}</div>'
            f'<div style="font-size:3rem;font-weight:300;color:{dcolor};margin:6px 0">{arrow} {abs(delta)*100:.2f}%</div>'
            f'<div class="cris-delta">absolute Δ probability</div>'
            f'</div>', unsafe_allow_html=True)

        seg_note = ("⚠ segment changed to " + m["segment_label"]) if d["segment_changed"] else "• segment unchanged"
        chu_note = ("⚠ churner flag flipped") if d["is_churner_changed"] else "• churner flag stable"
        st.markdown(
            f'<div class="cris-card cris-card-tight" style="margin-top:10px">'
            f'<div style="color:{"#d4a574" if d["segment_changed"] else "#6c7a89"};font-size:0.85rem">{seg_note}</div>'
            f'<div style="color:{"#d4a574" if d["is_churner_changed"] else "#6c7a89"};font-size:0.85rem;margin-top:4px">{chu_note}</div>'
            f'</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="cris-label" style="text-align:center;letter-spacing:0.15em">MODIFIED</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge(m["churn_probability"], "", m["threshold"]), use_container_width=True)
        st.markdown(
            f'<div style="text-align:center">'
            f'{chip(m["segment_label"], "accent")}'
            f'{chip(mod_status, mod_tone)}'
            f'</div>'
            f'<div class="cris-card cris-card-tight" style="margin-top:10px">'
            f'<div class="cris-label">Churn probability</div>'
            f'<div class="cris-value" style="font-size:1.5rem">{m["churn_probability"]*100:.2f}%</div>'
            f'<div class="cris-delta">confidence {m["segment_confidence"]*100:.1f}%</div>'
            f'</div>', unsafe_allow_html=True)

    # Threshold crossing banner
    if o["is_churner"] and not m["is_churner"]:
        st.markdown(
            '<div class="cris-card" style="margin-top:16px;border-color:#4a6a4a;background:rgba(143,188,143,0.08)">'
            '<div style="color:#8fbc8f;font-weight:500;letter-spacing:0.05em">✓ SUCCESS · Customer converted from churner to retained</div>'
            '<div style="color:#a6b3c0;font-size:0.85rem;margin-top:4px">This intervention is predicted to save the account.</div>'
            '</div>', unsafe_allow_html=True)
    elif not o["is_churner"] and m["is_churner"]:
        st.markdown(
            '<div class="cris-card" style="margin-top:16px;border-color:#6b3939;background:rgba(201,122,122,0.08)">'
            '<div style="color:#c97a7a;font-weight:500;letter-spacing:0.05em">⚠ WARNING · Modification pushes customer into churner segment</div>'
            '<div style="color:#a6b3c0;font-size:0.85rem;margin-top:4px">Reconsider — this change worsens predicted retention.</div>'
            '</div>', unsafe_allow_html=True)

    # Trajectory
    st.markdown("### Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=["Original", "Modified"],
        y=[o["churn_probability"], m["churn_probability"]],
        mode="lines+markers+text",
        text=[f"{o['churn_probability']*100:.1f}%", f"{m['churn_probability']*100:.1f}%"],
        textposition="top center",
        line=dict(color="#7dd3c0", width=3),
        marker=dict(size=14, color="#7dd3c0", line=dict(color="#0b0f14", width=2)),
        textfont=dict(color="#e6edf3")))
    fig.add_hline(y=o["threshold"], line=dict(color="#d4a574", dash="dash", width=1),
                  annotation_text=f"threshold {o['threshold']:.3f}", annotation_font_color="#d4a574")
    traj_layout = {**PLOTLY_LAYOUT, "yaxis": dict(range=[0, 1], gridcolor="#243242", tickformat=".0%")}
    fig.update_layout(**traj_layout, height=280)
    st.plotly_chart(fig, use_container_width=True)

    # Modified-field ledger
    st.markdown("### Applied modifications")
    mods = resp.get("modified_features", {})
    if mods:
        ledger = pd.DataFrame([
            {"Field": k,
             "Original": str(original_customer.get(k, "—")),
             "→": "→",
             "Modified": str(v)}
            for k, v in mods.items()
        ])
        st.dataframe(ledger, use_container_width=True, hide_index=True)
    else:
        st.info("No modifications recorded.")


# ═════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════
def page_explain():
    st.markdown("# Explainability Studio")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">SHAP-based global and per-customer explanations.</p>', unsafe_allow_html=True)

    tabs = st.tabs(["Global Importance", "Instance Explanation", "Methods"])

    with tabs[0]:
        n = st.slider("Top N features", 5, 30, 12)
        resp = client.global_importance(n)
        if isinstance(resp, dict) and resp.get("success") and resp.get("explanation"):
            e = resp["explanation"]
            feats = e["top_features"]
            df = pd.DataFrame(feats).sort_values("importance")
            colors = ["#c97a7a" if f.get("sign") == "positive" else "#8fbc8f" for f in feats][::-1]
            fig = go.Figure(go.Bar(
                x=df["importance"], y=df["feature_name"], orientation="h",
                marker=dict(color=colors, line=dict(color="#243242", width=1)),
                text=[f"{v:.3f}" for v in df["importance"]], textposition="outside",
                textfont=dict(color="#a6b3c0"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(300, 30 * n + 60))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Features analysed", str(len(feats)))
            with c2: metric_card("Explainer", e.get("explainer_type", "shap"))
            with c3: metric_card("Background sample", f"{e.get('sample_size',0):,}")
        else:
            st.error(resp.get("error","Unavailable") if isinstance(resp, dict) else "Unavailable")

    with tabs[1]:
        with st.form("instance_explain_form"):
            customer = render_customer_form("exp")
            n = st.slider("Top features to explain", 3, 15, 6, key="exp_n")
            submitted = st.form_submit_button("Explain prediction", type="primary")
        if submitted:
            with st.spinner("Computing SHAP…"):
                resp = client.instance_importance(customer, n)
            if isinstance(resp, dict) and resp.get("success") and resp.get("explanation"):
                e = resp["explanation"]
                pred = e["prediction"]
                c1, c2 = st.columns([1, 1.3])
                with c1:
                    st.plotly_chart(gauge(pred["churn_probability"], "Churn probability", pred["threshold"]),
                                    use_container_width=True)
                    st.markdown(f'<div style="text-align:center">{chip(pred["segment_label"], "accent")}</div>', unsafe_allow_html=True)
                with c2:
                    fig = shap_bar(e["top_features"], title="Per-instance contributions")
                    if fig: st.plotly_chart(fig, use_container_width=True)
                base = e.get("base_value")
                if base is not None:
                    st.caption(f"Base value (expected model output): **{base:.4f}**")
            else:
                st.error(resp.get("error", "Explanation failed") if isinstance(resp, dict) else "Error")

    with tabs[2]:
        m = client.explanation_methods()
        if isinstance(m, dict) and "available_methods" in m:
            for method in m["available_methods"]:
                st.markdown(
                    f'<div class="cris-card" style="margin-bottom:10px">'
                    f'<div class="cris-label">{method["type"]}</div>'
                    f'<div class="cris-value" style="font-size:1.1rem">{method["name"]}</div>'
                    f'<div class="cris-delta">{method["description"]}</div>'
                    f'<div style="margin-top:8px">'
                    f'{chip("speed · " + method["speed"], "warn")}{chip("accuracy · " + method["accuracy"], "ok")}'
                    f'</div></div>', unsafe_allow_html=True)
            st.caption(f"Default: **{m.get('default_method','')}** · {m.get('note','')}")


# ═════════════════════════════════════════════════════════════════
# PAGE: MODEL INTELLIGENCE
# ═════════════════════════════════════════════════════════════════
def page_model():
    st.markdown("# Model Intelligence")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">Architecture, configuration, and registered segments.</p>', unsafe_allow_html=True)

    info = client.model_info()
    if not isinstance(info, dict) or "model_info" not in info:
        st.error("Model info unavailable."); return

    mi = info["model_info"]
    churn = mi.get("churn_model", {})
    seg = mi.get("segmentation_model", {})
    expl = mi.get("explainer", {})

    # Churn model
    st.markdown("### Churn model")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Framework</div>'
                f'<div class="cris-value" style="font-size:1.1rem">{churn.get("framework","")}</div>'
                f'<div class="cris-delta">{churn.get("model_name","")}</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Features</div>'
                f'<div class="cris-value">{churn.get("num_features","—")}</div>'
                f'<div class="cris-delta">inputs · {churn.get("input_features","")}</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Estimators</div>'
                f'<div class="cris-value">{churn.get("n_estimators","—")}</div>'
                f'<div class="cris-delta">max depth · {churn.get("max_depth","")}</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Threshold</div>'
                f'<div class="cris-value">{churn.get("decision_threshold",0):.3f}</div>'
                f'<div class="cris-delta">optimised for F1</div></div>', unsafe_allow_html=True)

    pm = churn.get("performance_metrics", {})
    if pm:
        st.markdown("### Performance")
        fig = go.Figure(go.Scatterpolar(
            r=[pm.get("roc_auc",0), pm.get("accuracy",0), pm.get("precision",0), pm.get("recall",0), pm.get("f1",0)],
            theta=["ROC-AUC","Accuracy","Precision","Recall","F1"],
            fill="toself", fillcolor="rgba(125,211,192,0.18)",
            line=dict(color="#7dd3c0", width=2),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          polar=dict(bgcolor="rgba(0,0,0,0)",
                                     radialaxis=dict(range=[0,1], gridcolor="#243242", tickfont=dict(color="#6c7a89")),
                                     angularaxis=dict(gridcolor="#243242", tickfont=dict(color="#a6b3c0"))))
        st.plotly_chart(fig, use_container_width=True)

    # Segmentation
    st.markdown("### Segmentation model")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Algorithm</div>'
                f'<div class="cris-value" style="font-size:1.1rem">{seg.get("algorithm","")}</div>'
                f'<div class="cris-delta">{seg.get("framework","")}</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Clusters</div>'
                f'<div class="cris-value">{seg.get("num_clusters","—")}</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="cris-card cris-card-tight"><div class="cris-label">Training size</div>'
                f'<div class="cris-value">{seg.get("training_data_size","—"):,}</div></div>' if isinstance(seg.get("training_data_size"), int)
                else '<div class="cris-card cris-card-tight"><div class="cris-label">Training size</div><div class="cris-value">—</div></div>', unsafe_allow_html=True)

    # Segments library
    segs = mi.get("segments", {})
    if segs:
        st.markdown("### Segment catalogue")
        cols = st.columns(len(segs))
        for i, (sid, meta) in enumerate(segs.items()):
            with cols[i]:
                st.markdown(
                    f'<div class="cris-card" style="height:100%">'
                    f'<div class="cris-label">Segment {sid}</div>'
                    f'<div style="font-size:1.05rem;color:#e6edf3;margin:4px 0 10px">{meta.get("name","")}</div>'
                    f'<div style="color:#6c7a89;font-size:0.85rem;line-height:1.4">{meta.get("description","")}</div>'
                    f'</div>', unsafe_allow_html=True)

    # Explainer
    if expl:
        st.markdown("### Explainer")
        st.json(expl, expanded=False)


# ═════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════
def page_settings():
    st.markdown("# Settings")
    st.markdown('<p style="color:#6c7a89;margin-top:-6px">API endpoint and diagnostics.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    new_base = c1.text_input("API base URL", value=st.session_state.api_base)
    if c2.button("Apply & reconnect", type="primary"):
        st.session_state.api_base = new_base.rstrip("/")
        st.rerun()

    st.markdown("### Health diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        h = client.health()
        st.markdown("**GET /health**"); st.json(h)
    with c2:
        info = client.model_info()
        st.markdown("**GET /api/explanations/model-info**"); st.json(info, expanded=False)

    st.markdown("### About")
    st.markdown(
        '<div class="cris-card">'
        '<div class="cris-label">CRIS · Customer Retention Intelligence System</div>'
        '<div style="color:#a6b3c0;line-height:1.6;margin-top:6px">'
        'Streamlit dashboard for a churn-prediction & customer-segmentation pipeline. '
        'Consumes FastAPI endpoints for single / batch scoring, what-if simulation, and SHAP explainability.'
        '</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────
PAGES = {
    "Overview":           page_overview,
    "Customer Analysis":  page_customer,
    "Batch Scoring":      page_batch,
    "Action Planning":    page_actions,
    "What-If Lab":        page_whatif,
    "Explainability":     page_explain,
    "Model Intelligence": page_model,
    "Formulas":           page_formulas,
    "Settings":           page_settings,
}
PAGES[page]()
