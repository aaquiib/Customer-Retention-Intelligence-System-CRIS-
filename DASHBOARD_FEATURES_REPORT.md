# CRIS Dashboard - Comprehensive Features Report

**Customer Retention Intelligence System**  
*Telco Customer Churn Prediction & Segmentation Dashboard*

---

## Executive Summary

The CRIS dashboard is a comprehensive, professionally-styled web application for customer churn prediction, segmentation, and retention strategy planning. Built with Streamlit using a sophisticated dark theme, it provides 9 integrated pages delivering end-to-end analytics, model explainability, and actionable insights for customer retention.

**Key Capabilities:**
- ✅ Real-time single customer prediction with SHAP explainability
- ✅ Batch CSV processing (bulk analysis up to 50,000 customers)
- ✅ Customer segmentation (4 segments from K-Means clustering)
- ✅ Churn risk analysis with revenue impact quantification
- ✅ What-If scenario simulation for retention strategy testing
- ✅ SHAP-based model explainability (global and per-instance)
- ✅ Actionable customer targeting with priority ranking
- ✅ Model performance monitoring and diagnostics
- ✅ Comprehensive formulas and definitions reference

**Technology Stack:**
- **Frontend**: Streamlit 1.28.1 with custom CSS (dark theme, sophisticated aesthetics)
- **Architecture**: Monolithic single-file Streamlit app (dashboard1/app.py, ~2100 lines)
- **Backend**: FastAPI with LightGBM + KMeans
- **ML Models**: 
  - Churn: LightGBM Classifier (650 estimators, 33 engineered features)
  - Segmentation: KMeans (4 clusters, k-means++ initialization)
- **Explainability**: SHAP TreeExplainer (200 background samples)
- **Data Version**: Telco Customer Churn Dataset (7,032 training samples)

---

## Application Architecture

### Design Philosophy

**Monolithic Single-File Structure**: Unlike modular Streamlit multi-page apps, CRIS uses a single `dashboard1/app.py` file (~2100 lines) with 9 integrated pages. This approach prioritizes:
- **Visual Coherence**: Unified dark theme with muted teal/amber accents throughout
- **Tight Integration**: Session state seamlessly shared across all pages
- **Responsive Performance**: No module imports delays; fast page navigation
- **Professional Styling**: Custom CSS applied globally for consistent branding

### Page Navigation

Sidebar menu using `streamlit-option-menu` with 9 integrated pages:
1. **Overview** — KPI dashboard and global insights
2. **Customer Analysis** — Single customer prediction with SHAP
3. **Batch Scoring** — CSV upload, bulk predictions, segment/risk analysis
4. **Action Planning** — Prioritized customer targeting and interventions
5. **What-If Lab** — Scenario simulation for strategy testing
6. **Explainability** — Global and instance-level SHAP explanations
7. **Model Intelligence** — Model architecture, metrics, and segmentation library
8. **Formulas** — Complete reference of all calculations and definitions
9. **Settings** — API diagnostics and configuration

### Session State Management

Global `st.session_state` persists across all 9 pages within a single browser session:

```python
Session state keys:
  - api_base: API endpoint (default: http://localhost:8000)
  - last_prediction: Last single customer prediction result
  - last_batch: Last batch scoring result + predictions list
  - last_whatif: Last what-if simulation result + customer context
  - customer_form: Manual customer input form state (19 fields)
  - wif_*: What-If Lab-specific state (mode, active scenario, etc.)
  - ap_*: Action Planning-specific state (filters, preset, etc.)
```

**Data Flow**: 
- Upload CSV on Page 3 (Batch Scoring) → stored in `st.session_state.last_batch`
- All subsequent pages (Overview, Action Planning, What-If, Explainability) read from shared state
- Manual customer form on Page 2 (Customer Analysis) persists across page switches
- Clearing session happens on browser refresh or explicit reset buttons

---

## Color Palette & Design System

### Theme Variables (CSS)
```css
--bg-0: #0b0f14       /* Main background (darkest) */
--bg-1: #111820       /* Card background */
--bg-2: #161f2a       /* Hover/highlight states */
--bg-3: #1e2935       /* Sidebar active */
--line: #243242       /* Borders & dividers */
--text-0: #e6edf3     /* Primary text */
--text-1: #a6b3c0     /* Secondary text */
--text-2: #6c7a89     /* Tertiary/labels */
--accent: #7dd3c0     /* Muted teal (primary accent) */
--accent-dim: #3e6e66 /* Muted teal (dark) */
--warn: #d4a574       /* Soft amber (warning) */
--danger: #c97a7a     /* Muted rose (danger/at-risk) */
--ok: #8fbc8f         /* Sage green (positive/retained) */
```

### Component Library

**metric_card(label, value, delta="", tone="")**
- Displays KPI metrics in 2-3 line card format
- `tone` controls delta color: "up" (green), "down" (red), "" (neutral)

**chip(text, tone="accent")**
- Inline pill-shaped badge for tags/status
- Tones: "accent", "warn", "danger", "ok"

**gauge(value, title, threshold)**
- Plotly-rendered gauge chart showing churn probability
- Color-coded zones: green (safe), amber (warning), red (danger)

**shap_bar(features, title)**
- Horizontal bar chart for SHAP feature contributions
- Red bars = increase churn, green bars = decrease churn

---

## Page-by-Page Feature Guide

### 📊 Page 01: Overview Dashboard

**Purpose**: High-level KPI summary and business intelligence

**Key Metrics Displayed** (Top Row):
| Metric | Source | Updates | Purpose |
|--------|--------|---------|---------|
| Model ROC-AUC | API `/explanations/model-info` | On page load (cached 300s) | Model quality indicator |
| Decision Threshold | API model-info | Static (0.4356) | F1-optimised cutoff |
| Customer Segments | API model-info | Static (4 clusters) | Segmentation count |
| Training Cohort | API model-info | Static (7,032 rows) | Training data volume |

**Visualizations**:

1. **Global Churn Drivers** (Left column)
   - Horizontal bar chart of top 10 SHAP features
   - X-axis: Feature importance (mean absolute SHAP value)
   - Color: Teal accent with dark borders
   - Source: `/api/feature-importance/global?top_n=10`
   - Updates: Every 5 minutes (cached)

2. **Segment Library** (Right column)
   - 4 stacked cards (one per segment)
   - Each card shows: Segment #, Name, Description
   - Color-coded accent chips for visual distinction
   - Example: "Segment 0 · Loyal High-Value · Long tenure, high spend, low churn"

3. **Model Performance** (Bottom)
   - 5 metric cards: ROC-AUC, Accuracy, Precision, Recall, F1
   - All metrics displayed at 3 decimal places
   - Color-coded status: ✓ Good (>0.75), ⚠ Fair (0.6-0.75), ✗ Poor (<0.6)

4. **Current Batch Insights** (Bottom, if batch data loaded)
   - Batch size (customer count)
   - Portfolio churn rate (%)
   - Average churn probability (%)
   - Dominant segment (which segment has most customers)

**Features**:
- ✅ Real-time API health indicator in sidebar
- ✅ Dynamic KPI updates as batch data changes
- ✅ Click-through to detailed segment profiles from Segment Library cards
- ✅ Model performance interpretation guide (color-coded status badges)

---

### 👤 Page 02: Customer Analysis

**Purpose**: Individual customer churn prediction with detailed explanation and business context

**Input Form** (19 fields, organized in 3 tabs):

**Tab 1: Profile** (Demographics)
- Gender: [Male / Female]
- Senior citizen: [0 / 1]
- Partner: [Yes / No]
- Dependents: [Yes / No]
- Tenure (months): [0-120] slider
- Monthly charges ($): [0-500] numeric input
- Total charges ($): [0-50,000] numeric input

**Tab 2: Contract & Services**
- Contract: [Month-to-month / One year / Two year]
- Internet service: [Fiber optic / DSL / No]
- Phone service: [Yes / No]
- Multiple lines: [Yes / No / No phone service]
- Online security: [Yes / No / No internet service]
- Online backup: [Yes / No / No internet service]
- Device protection: [Yes / No / No internet service]
- Tech support: [Yes / No / No internet service]
- Streaming TV: [Yes / No / No internet service]
- Streaming movies: [Yes / No / No internet service]

**Tab 3: Billing & Payment**
- Paperless billing: [Yes / No]
- Payment method: [Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)]

**Output Section** (3-column layout):

**Column 1: Churn Gauge**
- Large gauge visualization: 0-100% scale
- Color zones: Green (safe) → Amber (warning) → Red (danger)
- Threshold line at 0.4356 (F1-optimised)
- Status chips: "AT RISK" (red) or "RETAINED" (green)

**Column 2: Segment & Recommended Action**
- Segment name with cluster ID and confidence % (e.g., "Loyal High-Value · Cluster #0 · 89.2% confidence")
- Recommended action card (from business rules engine):
  - Action label: "Retain", "Review", "At Risk", "Monitor", "Upgrade offer"
  - Priority score (0-1) with color-coded chip
  - Detailed reason text (e.g., "Long-tenure, high-value customer showing early signs of service dissatisfaction")

**Column 3: Top SHAP Drivers**
- Horizontal bar chart: Top 5 SHAP features
- Table below with columns: Feature, Value, SHAP, Impact direction
- Color: Red (increases churn) / Green (decreases churn)

**Additional Sections**:
- "Engineered features & raw response" expander showing:
  - Full engineered feature vector (33 features post-processing)
  - Complete API JSON response for debugging

**Features**:
- ✅ Form state persists across page navigation
- ✅ Manual input or load from last batch (dropdown on Batch page)
- ✅ Real-time prediction <500ms response time
- ✅ Confidence badges (✅ if model AUC > 0.80)
- ✅ SHAP explanation always included (or "unavailable" message with API diagnostics)

---

### 📤 Page 03: Batch Scoring

**Purpose**: Bulk customer analysis via CSV upload with comprehensive segment and risk breakdown

**Upload Interface**:
- Drag-and-drop CSV or file picker
- Accepted format: CSV with 19 required columns
- Max file size: 200MB
- Downloadable CSV template and sample file link

**Processing Pipeline**:
1. Validate CSV (correct column count, data types)
2. Normalize column names (case-insensitive matching)
3. Call `/api/predict-batch` endpoint
4. Parse results into prediction dataframe with:
   - customerID, churn_probability, is_churner, segment, segment_label
   - All 19 input features + 8 engineered features
   - Recommended action + priority score

**Output Results** (3 major sections):

### Section 1: Batch Summary Metrics (Top KPI strip)
- Rows processed: "1,234 of 5,000" (rows_processed / total_rows)
- Churn rate: "23.4%" (sum of is_churner / total)
- Avg churn prob.: "42.3%" (mean churn_probability)
- Avg segment confidence: "78.5%" (mean segment_confidence)
- Latency: "12.3s" (end-to-end processing time)

### Section 2: Segment & Churn Risk Analysis

**Visualizations**:

1. **Segment Distribution** (Donut chart)
   - 4 segments shown as proportional slices
   - Color-coded by segment
   - Clickable legend for filtering
   - Percentages and counts displayed

2. **Recommended Actions** (Horizontal bar chart)
   - X-axis: Count of customers per action type
   - Y-axis: Action label (e.g., "Retain", "Review", "At Risk")
   - Color: Amber gradient (softer than red/green for neutral context)

3. **Churn Risk Distribution** (Histogram + Risk band pie)
   - Histogram: Churn probability bins (0-1) with customer counts
   - Threshold line at 0.4356 showing decision boundary
   - Adjacent pie chart: Low (<0.35) / Medium (0.35-0.65) / High (≥0.65) breakdown
   - Colors: Green / Amber / Red respectively

4. **Segment × Risk Band Heatmap**
   - X-axis: 3 risk bands (Low, Medium, High)
   - Y-axis: 4 customer segments
   - Cell values: Customer count per (segment, risk band) combination
   - Color intensity: Dark (safe) → Teal (at-risk)
   - Identifies which segments concentrate in high-risk zone

### Section 3: Revenue at Risk (Detailed breakdown)

**KPI Cards** (5-column strip):
- Portfolio MRR: Sum of all MonthlyCharges
- MRR at risk: Sum of MonthlyCharges for predicted churners
- Expected monthly loss: Sum of (churn_probability × MonthlyCharges)
- ARR at risk: MRR_at_risk × 12 (annualised)
- Avg customer value: Mean MonthlyCharges

**Segment-wise Breakdown**:
1. **MRR Composition Stacked Bar** (Segment × Safe/At-Risk)
   - Horizontal stacked bar per segment
   - Safe MRR (gray-teal) + At-Risk MRR (muted rose)
   - Labels inside bars with exact $$ amounts

2. **Risk-Share Donut**
   - Shows what proportion of total at-risk MRR belongs to each segment
   - Helps identify which segment is driving overall risk

3. **Segment Intelligence Table**
   - Columns: Segment, Customers, Churners, Churn rate, Avg prob., Avg tenure, Avg MRR, Segment MRR, MRR at risk, ARR at risk, Risk share, Avg priority, Top action
   - Sortable, color-coded by values
   - Shows recommended action per segment

4. **Segment Highlight Cards** (2-column responsive grid)
   - Per-segment narrative summaries
   - Each card: Segment name, churn rate chip, customers, churners, MRR, MRR at risk, avg tenure, avg priority, recommended focus action
   - Color: Tone varies by churn rate (green < 20%, amber 20-40%, red > 40%)

### Section 4: Per-Customer Predictions Table

**Columns** (visible on scroll):
- #: Row number (1-N)
- Segment: Segment label
- Churn prob.: Predicted churn probability (4 decimals)
- Churner: "●" (churner) or "○" (loyal) symbol
- Action: Recommended action
- Priority: Priority score (0-1)
- Tenure, Monthly, Contract: Customer attributes
- Plus all 19 input features

**Features**:
- ✅ Sortable by clicking column headers
- ✅ Filter checkboxes: "Show only predicted churners", "Min. priority slider"
- ✅ Pagination (default 50 rows per page)
- ✅ Export as CSV button
- ✅ Row highlighting by risk band (color gradient)

**Performance Considerations**:
- Batch processing: ~10-20ms per customer (650-tree LightGBM)
- SHAP explanations (global top features): ~500ms
- Full batch with 10,000 customers: ~3-5 minutes

---

### 📋 Page 04: Action Planning

**Purpose**: Filtered customer table with priority ranking and batch action export capabilities

**Quick Presets** (Button strip):
- "All customers" → No filters
- "High-priority churners" → Priority=High, Churner=Yes
- "High-risk · monthly" → Risk=High, Contract=Month-to-month
- "Silent defectors" → Risk=[Medium, High], Churner=No
- "High-value at risk" → Risk=High, MonthlyCharges ≥ $80

**Multi-Select Filters** (5-column layout):
- **Segments**: Checkboxes for all segments in batch
- **Actions**: Checkboxes for all action types
- **Risk Band**: [Low, Medium, High] multi-select
- **Contract Type**: [Month-to-month, One year, Two year] multi-select
- **Priority Level**: [High, Medium, Low] multi-select

**Additional Filters** (4-column sliders/inputs):
- Min. churn prob.: 0.0-1.0 slider (0.01 increments)
- Min. MRR: Numeric input (dollars)
- "Churners only" checkbox
- "Sort by" dropdown: Priority %, Churn prob., Monthly, Expected loss, etc.

**Search**: 
- Customer # text input for quick lookup

**Filtered Results KPI Strip**:
- In view: "234 of 5,000 customers"
- High priority: "67 customers"
- Predicted churners: "134"
- MRR at risk: "$47,230" (filtered view)
- Expected loss: "$8,945" (filtered monthly exposure)

**Visualizations** (2-column layout):
1. **Priority × Risk Heatmap**
   - 3×3 grid: Priority (High/Medium/Low) × Risk (High/Medium/Low)
   - Cell values: Customer count per (priority, risk) combination
   - Helps identify quadrant concentrations

2. **Action Mix Horizontal Bar**
   - Y-axis: Action type
   - X-axis: Count of customers per action
   - Color: Amber gradient

**Main Customer Table**:
- Shows ~20 columns (scrollable)
- Includes all predictive features, SHAP drivers, recommended actions
- Row styling: Color-coded by priority level
- 520px height (shows ~12-15 rows per page)

**Customer Lookup Panel** (Bottom):
- Input customer # → Opens detailed profile card
- Shows: Segment, churn probability gauge, recommended action, priority level
- Revenue metrics: Monthly, tenure, contract type, payment method, internet service
- Top 3 SHAP drivers as clickable chips
- Full customer profile expandable section

**Export Options**:
- "Filtered view (CSV)" → Download visible filtered rows
- "All high-priority (CSV)" → Download all Priority=High from entire batch

**Features**:
- ✅ Preset buttons for common filter combinations
- ✅ Real-time filtering on widget change (no "Apply" button)
- ✅ Search by customer # within filtered set
- ✅ Priority formula displayed at top (0.40 × churn_prob + 0.30 × seg_risk + 0.30 × cust_value)
- ✅ Supports 100K+ customers (pandas filtering is fast)

---

### 🎲 Page 05: What-If Simulation Lab

**Purpose**: Test retention strategies by modifying features and quantifying churn probability impact

**Workflow** (3-step process):

### Step 1: Customer Source
**Option A: Manual Entry**
- Pre-filled form with defaults from session state
- Full 19-field form (same as Customer Analysis page)

**Option B: Load from Last Batch**
- Preview table of batch predictions (scrollable, 5 columns)
- Select customer # from numeric input (1-N)
- Auto-populates all 19 fields
- Shows selected customer's segment and current churn probability

### Step 2: Policy Templates (Optional Quick Scenarios)
- Buttons for common interventions: "Contract upgrade", "Service bundle", "Discount", etc.
- Click to activate a pre-configured modification set
- Shows modified fields as amber chips
- Provides description: "Offer 20% discount on monthly charges"

### Step 3: Manual Modifications
**Modification interface** (2-column layout):
- Checkboxes to enable/override each field
- Unchecked fields keep original value
- Editable fields once checkbox is on:
  - Categorical: Dropdown select
  - Numeric: Number input

**Available modification fields**:
- Contract, InternetService, PaymentMethod, PaperlessBilling
- OnlineSecurity, TechSupport, StreamingTV, StreamingMovies, DeviceProtection, OnlineBackup
- tenure (months), MonthlyCharges ($)

**Simulate & Compare** (3-column card layout):

**Column 1: Original Prediction**
- Gauge showing original churn probability
- Segment chip + Status chip ("RETAINED" / "WILL CHURN")
- Card: Exact churn probability % + confidence

**Column 2: Delta (Impact)**
- Large arrow (▼ improvement / ▲ worsening / • no change)
- Percentage point change in churn probability
- Text: "IMPROVEMENT" / "WORSENING" / "NO CHANGE"
- Alerts: Segment changed? Churner flag flipped?

**Column 3: Modified Prediction**
- Gauge showing new churn probability
- Segment chip + Status chip
- Card: Exact new probability % + confidence

**Threshold Crossing Alerts**:
- ✓ Green: "Customer converted from churner to retained"
- ⚠ Orange: "Modification pushes customer into churner segment"

**Trajectory Visualization**:
- Line chart: Original → Modified probability
- X-axis: "Original" and "Modified"
- Y-axis: Probability (0-100%)
- Threshold line (dashed) showing decision boundary

**Applied Modifications Ledger** (Table):
- Columns: Field, Original value, →, Modified value
- Shows only changed fields
- Useful for audit trail

**Features**:
- ✅ Real-time simulation (<500ms response)
- ✅ Policy templates for common scenarios
- ✅ JSON serialization fix for numpy types (critical)
- ✅ Undo/redo scenario changes (session history)
- ✅ Compare multiple scenarios side-by-side
- ✅ Cost-benefit analysis (optional: intervention cost vs LTV)

---

### 🔍 Page 06: Explainability Studio

**Purpose**: SHAP-based global and per-customer explanations for model transparency

**Tab 1: Global Importance**

**Horizontal bar chart**: Top N features by mean |SHAP|
- X-axis: Feature importance score
- Y-axis: Feature names (sorted by importance)
- Color: Teal with dark borders
- Interactive: Hover for exact values
- Slider: Adjust N features (5-30, default 12)

**Summary table**:
- Columns: Feature, Importance score, Impact direction (positive/negative), Interpretation
- Example row: "Contract | 0.0842 | Negative | Month-to-month → higher churn risk"

**Metadata cards** (3-column strip):
- Features analysed: "12 of 33 engineered features"
- Explainer: "SHAP TreeExplainer (LightGBM)"
- Background sample: "200 reference customers"

**Tab 2: Instance Explanation**

**Customer Input**:
- Either manual form entry (19-field tabs)
- Or load from batch (dropdown select)

**Output**:
1. **Churn Gauge + Segment Chip** (Left column, 30% width)
   - Large gauge: Churn probability with threshold
   - Segment label chip

2. **Per-Instance SHAP Waterfall** (Right column, 70% width)
   - Horizontal bar chart: Feature contributions
   - Base value shown at left
   - Features sorted by |SHAP value|
   - Color: Red (increases churn) / Blue (decreases churn)
   - Text labels: +/- SHAP values

3. **Feature Contributions Table**
   - Columns: Feature, SHAP value, Direction (↑/↓), Impact (% contribution to final prediction)
   - Example: "Contract | +0.0847 | ↑ | Increases churn by 8.5%"

4. **Prediction Explanation (Text)**
   - "This customer has 58% churn probability (High Risk)"
   - Key drivers ranked
   - Mitigation suggestions

5. **Model Details**
   - Framework: LightGBM (Decision Tree Ensemble)
   - Trees: 650
   - Features: 33 engineered
   - Train/Val/Test AUC: 0.8791 / 0.8732 / 0.8398
   - Decision Threshold: 0.4356

**Tab 3: Methods**

**Available Explainers**:
- SHAP TreeExplainer (default, fast, ~200ms per instance)
- SHAP TreeExplainerv2 (alternate, higher accuracy)
- SHAP DeepExplainer (if model available)

**Method Cards**: Each showing:
- Name, Type, Description
- Speed: Fast / Medium / Slow
- Accuracy: Good / Excellent
- Trade-offs & use cases

**Features**:
- ✅ SHAP values computed on-demand (cached)
- ✅ Handles missing features gracefully
- ✅ Compares multiple customers side-by-side (optional)
- ✅ Export explanations as PNG/PDF

---

### 🏥 Page 07: Model Intelligence

**Purpose**: Detailed model architecture, configuration, performance metrics, and segment definitions

**Section 1: Churn Model Architecture**

4-column card layout:
- **Framework**: LightGBM Classifier
- **Features**: 33 (inputs: 19 raw + 14 categorical encodings)
- **Estimators**: 650 trees
- **Max depth**: 13
- **Threshold**: 0.4356 (F1-optimised)

**Performance Metrics** (5-card strip):
- ROC-AUC: 0.8398
- Accuracy: 75.83%
- Precision: 53.01%
- Recall: 78.57%
- F1 Score: 63.31%

**Polar chart** (Radar visualization):
- 5 axes: ROC-AUC, Accuracy, Precision, Recall, F1
- Filled area showing performance profile
- Helps identify strength/weakness patterns

**Train / Val / Test Split Comparison**:
- Table: Split name, ROC-AUC, Status
- Train: 0.8791 ✓
- Validation: 0.8732 ✓
- Test: 0.8398 ✓
- Overfitting check: Gap < 5% = ✓ Good generalization

**Section 2: Feature Engineering Pipeline**

**3-column flow diagram**:
- Column 1: "Raw input" → 19 features
- Column 2: "Preprocessing" → +14 one-hot encodings
- Column 3: "Final features" → 33 total

**Engineered features table**:
- 8 features listed with descriptions
- Examples:
  - "avg_monthly_spend": Average monthly spending across tenure
  - "charge_gap": Difference between total and monthly charges
  - "streaming_count": Number of streaming services subscribed
  - "security_count": Number of security services subscribed
  - "support_services": Tech support + online support indicators
  - etc.

**Section 3: Segmentation Model**

3-column card layout:
- **Algorithm**: K-Means (K-Means++)
- **Clusters**: 4 segments
- **Training size**: 7,032 customers
- **Random seed**: 42 (reproducible)
- **Initialization runs**: 10

**Segment Catalogue** (4-column card grid):

Each segment card shows:
- Segment #
- Name (e.g., "Loyal High-Value")
- Description (e.g., "Long tenure, high spend, low churn")
- Optional: Size (% of customers), Risk profile emoji

**Section 4: SHAP Explainer Configuration**

- **Type**: SHAP (SHapley Additive exPlanations)
- **Algorithm**: TreeExplainer (Fast, optimized for LightGBM)
- **Background samples**: 200 customers
- **Computation speed**: ~200ms per instance explanation
- **Feature importance method**: Mean |SHAP value|

**Section 5: System Health & Status**

- **API Status**: 🟢 Running / 🔴 Offline
- **Response time**: [X ms] average
- **Last sync**: [timestamp]
- **Model load time**: [X seconds at startup]

**Data Statistics**:
- Training set: 7,032 samples
- Features used: 20 raw (33 after engineering)
- Training date: [from metadata]
- Last updated: [from metrics_latest.json]

**Features**:
- ✅ Interactive metrics radar chart
- ✅ Train/val/test split comparison with overfitting detection
- ✅ Feature engineering pipeline visualization
- ✅ Full segment catalogue with descriptions
- ✅ Model reproducibility info (seed, initialization)

---

### 📖 Page 08: Formulas & Definitions

**Purpose**: Complete reference of all calculations, definitions, and formulas used across the dashboard

**Quick Navigation** (Chip bar):
- "Churn & Risk", "Explainability", "Priority & Action", "Revenue Economics", "Model Performance"

**Formula Cards** (2-column layout):

**Section: Churn & Risk**
1. **Churn probability**
   - Formula: p = σ(f_LGBM(x)) ∈ [0, 1]
   - Note: Output of LightGBM churn classifier; x = 19 customer features + segment label

2. **Churner flag**
   - Formula: is_churner = (p ≥ τ), τ = 0.4356
   - Note: F1-optimised threshold on validation data

3. **Risk band**
   - Formula: High if p ≥ 0.65, Medium if 0.35 ≤ p < 0.65, Low if p < 0.35
   - Note: Used in Action Planning for filtering

4. **Segment confidence (K-Means)**
   - Formula: confidence = 1 − d₁ / d₂
   - Note: d₁ = distance to assigned centroid, d₂ = distance to second-closest

5. **Decision delta (What-If)**
   - Formula: Δp = p_modified − p_original
   - Note: Negative = improvement, positive = worsening

**Section: Explainability (SHAP)**
1. **Per-instance contribution**
   - Formula: p̂(x) = φ₀ + Σⱼ φⱼ(x)
   - Note: φ₀ = base value, φⱼ = SHAP value of feature j

2. **Impact direction**
   - Formula: "increases_churn" if φⱼ > 0, "decreases_churn" if φⱼ < 0

3. **Global feature importance**
   - Formula: I_j = (1/N) · Σᵢ |φⱼ(xᵢ)|
   - Note: Mean absolute SHAP value across N background samples

4. **Batch global importance**
   - Formula: Ī_j = (1/N) · Σᵢ |φⱼ(xᵢ)|, freq_j = #{ i : j ∈ top-5 of xᵢ }
   - Note: Computed over current batch only

**Section: Priority & Action**
1. **Composite priority score**
   - Formula: Priority = 0.40 · p + 0.30 · segment_risk + 0.30 · customer_value
   - Note: Used in Action Planning for ranking

2. **Segment risk (batch-derived)**
   - Formula: segment_risk_s = mean( p_i : segment_i = s )

3. **Customer value (batch-normalised)**
   - Formula: customer_value = (MC − MC_min) / (MC_max − MC_min)
   - Note: Min-max normalisation of MonthlyCharges

4. **Priority bands**
   - Formula: High if Priority ≥ 0.67, Medium if 0.34 ≤ Priority < 0.67, Low if Priority < 0.34

**Section: Revenue Economics**
1. **Portfolio MRR**
   - Formula: MRR = Σᵢ MonthlyChargesᵢ

2. **MRR at risk**
   - Formula: MRR_risk = Σᵢ MonthlyChargesᵢ · 𝟙(is_churnerᵢ)
   - Note: Sum for predicted churners only

3. **Expected monthly loss**
   - Formula: E[loss]_monthly = Σᵢ pᵢ · MonthlyChargesᵢ
   - Note: Probability-weighted exposure

4. **ARR at risk**
   - Formula: ARR_risk = MRR_risk · 12

5. **Risk share (per segment)**
   - Formula: share_s = MRR_risk(s) / MRR_risk(total)

6. **Segment churn rate**
   - Formula: churn_rate_s = #{churners in s} / #{customers in s}

**Section: Model Performance**
1. **Classification metrics**
   - Formula: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2·P·R/(P+R), etc.
   - Note: Reported on test set using τ = 0.4356

---

### ⚙️ Page 09: Settings

**Purpose**: API configuration, health diagnostics, and application metadata

**API Configuration**:
- **URL Input**: Text field with current API base URL
- **"Apply & Reconnect" Button**: Updates session state and refreshes all cached data

**Health Diagnostics** (2-column layout):

**Column 1: GET /health**
- Live API health check result
- Shows:
  - Status: "ok" / "initializing" / "error"
  - models_loaded: true/false
  - Uptime, response time
  - Detailed error messages (if applicable)

**Column 2: GET /api/explanations/model-info**
- Full model metadata JSON
- Expandable sections for: churn_model, segmentation_model, explainer, segments
- Read-only JSON viewer

**About Section**:
- "CRIS — Customer Retention Intelligence System"
- Short description of dashboard purpose and architecture
- Version number, build date
- Links to documentation, support

**Features**:
- ✅ Real-time API connectivity check
- ✅ Detailed error diagnostics for troubleshooting
- ✅ Full JSON inspection of API responses
- ✅ One-click reconnect to different API endpoint

---

## Cross-Page Features & Integration

### 🔄 Session State & Data Persistence

All pages share a unified session state that persists across page navigation within a single browser session:

**Data Flow Architecture**:
```
CSV Upload (Page 3)
    ↓
    POST /api/predict-batch
    ↓
    st.session_state.last_batch = {predictions: [...], status: {...}, ...}
    ↓
    Accessed by: Pages 1, 2, 4, 5, 6, 7
    ↓
    Persists until: page refresh, explicit reset, or new upload
```

**Manual Form State**:
```
Customer form inputs (Page 2 / Page 5 / Page 6)
    ↓
    st.session_state.customer_form = {19 fields...}
    ↓
    Auto-save on every field change
    ↓
    Accessed by: All pages with customer input
```

### 🔌 API Integration Pattern

**Consistent across all pages**:
1. **GET endpoints** (cached, no network on widget-only reruns):
   - `/health` (5s TTL)
   - `/api/explanations/model-info` (300s TTL)
   - `/api/feature-importance/global?top_n=N` (300s TTL)
   - `/api/predict-batch/template` (600s TTL)
   - `/api/what-if/policy-changes` (600s TTL)

2. **POST endpoints** (called only on explicit action button):
   - `/api/predict` - Single customer prediction
   - `/api/predict-batch` - Batch CSV processing
   - `/api/what-if` - Scenario simulation
   - `/api/feature-importance/instance` - Per-customer SHAP

3. **Error handling**:
   - All responses wrapped in `{"__error__": "...", "detail": "..."}` on failure
   - User-friendly error messages displayed (no raw JSON dumps)
   - Fallback to demo/cached data when API unavailable

### 🎨 UI/UX Patterns (Consistent Across All Pages)

1. **Section dividers**: `st.markdown('<div class="cris-hr"></div>')` for visual flow
2. **Color coding**:
   - 🔴 Red (danger): Churn risk, high-risk customers
   - 🟡 Amber (warning): Medium risk, caution
   - 🟢 Green (positive): Low risk, retained, good metrics
   - 🔵 Teal accent (primary): Actions, emphasis, highlights
3. **Expandable sections**: `st.expander(...)` for detailed drill-downs
4. **Responsive columns**: `st.columns([w1, w2, ...])` for adaptive layouts
5. **Metric cards**: `metric_card()` helper for KPI display
6. **Chip badges**: `chip()` helper for status/tags
7. **Progress feedback**: `st.spinner()` for long-running operations
8. **Data tables**: `st.dataframe()` with sorting, filtering, export

---

## Technical Specifications

| Aspect | Specification |
|--------|---------------|
| **Frontend** | Streamlit 1.28.1 |
| **Architecture** | Monolithic single-file (~2100 lines) |
| **Custom styling** | CSS-in-HTML via `st.markdown()` |
| **FastAPI version** | 0.68+ (async, CORS enabled) |
| **Python version** | 3.10+ |
| **ML Framework** | LightGBM + scikit-learn (KMeans) |
| **Explainability** | SHAP 0.49+ (TreeExplainer) |
| **Single prediction latency** | 200-500ms (including SHAP) |
| **Batch processing** | 10-20ms per customer (50,000 customers ≈ 10-20 minutes) |
| **SHAP per-instance latency** | 200-300ms |
| **Session state retention** | Entire browser session (cleared on page refresh) |
| **Deployment** | Streamlit Cloud, Docker, or local `streamlit run` |

---

## Deployment Checklist

- [ ] API running on port 8000 (`uvicorn api.app:app --reload`)
- [ ] Streamlit dashboard running on port 8501 (`streamlit run dashboard1/app.py`)
- [ ] `.env` file configured with `API_BASE_URL=http://localhost:8000`
- [ ] Models loaded: churn (LightGBM) + segmentation (KMeans)
- [ ] SHAP explainer initialized with 200 background samples
- [ ] Metrics JSON readable (`models/churn/metrics_latest.json`)
- [ ] Segment labels JSON readable (`models/segmentation/segment_labels.json`)
- [ ] API `/health` endpoint returning 200 OK
- [ ] All 9 pages rendering without errors
- [ ] Type conversion working (numpy → Python types for JSON)
- [ ] Dark theme CSS applied correctly across all pages
- [ ] Session state persists across page navigation

---

## Troubleshooting Guide

| Issue | Resolution |
|-------|-----------|
| "Models not loaded" error | Ensure API is running first: `uvicorn api.app:app` |
| "Object of type int64 is not JSON serializable" | Type conversion should handle; if persists, check validators.py |
| "Cannot reach API" | Verify API_BASE_URL in .env; check firewall/network |
| "Unknown model type" in Model Intelligence | Verify metrics_latest.json exists and is readable |
| Slow batch predictions | May indicate SHAP background sampling overhead; check API logs |
| CSV upload fails | Verify 19 columns present with correct names (case-insensitive matching) |
| SHAP explanations "unavailable" | API may be computing; refresh page or check `/health` |
| Session data lost on page refresh | Expected behavior; upload CSV again or use form persistence |
| Dark theme not applied | Clear browser cache; may be CSS loading order issue |

---

## Support & Maintenance

**Code Quality**:
- ✅ Type hints throughout (for IDE support)
- ✅ Docstrings on all helper functions
- ✅ Clear section markers for page navigation (`# ═══ PAGE: [NAME] ═══`)
- ✅ Consistent naming conventions across 9 pages

**Future Enhancements**:
- Refactor to modular page structure (if file grows >2500 lines)
- Add A/B testing framework for intervention simulations
- Implement batch action exports to CRM systems
- Add real-time streaming of customer events (when new data arrives)
- Multi-language support for global deployments
- Dark/light theme toggle (currently dark only)

---

**Report Generated**: April 18, 2026  
**Dashboard Version**: 2.0 (dashboard1 as primary)  
**Status**: ✅ Production Ready  
**Architecture**: Monolithic single-file Streamlit (9 pages, ~2100 lines)  
**Last Updated**: Comprehensive rewrite documenting dashboard1 as primary CRIS dashboard

---

## Page-by-Page Feature Guide

### 📊 Page 01: Overview Dashboard

**Purpose:** High-level KPI summary and business intelligence

**Key Metrics Displayed:**
| Metric | Source | Updates | Purpose |
|--------|--------|---------|---------|
| Total Customers | Batch data | Per upload | Volume tracking |
| Overall Churn Rate | Batch predictions | Per upload | Risk overview |
| Avg Churn Probability | Model inference | Per upload | Aggregate risk |
| Revenue at Risk | Churn customers × Monthly charges | Dynamic | Financial impact |
| Model AUC | metrics_latest.json | Static (0.8398) | Model quality badge |
| API Status | Health check | Real-time | System availability |

**Visualizations:**
1. **Segment Distribution Donut Chart**
   - 4 segments: Loyal High-Value, Low Engagement, Stable Mid-Value, At-risk High-Value
   - Color-coded by risk level
   - Clickable for detail navigation

2. **Churn Risk Heatmap** (Segment × Risk Band)
   - X-axis: 4 customer segments
   - Y-axis: 3 risk bands (Low <0.35, Medium 0.35-0.65, High ≥0.65)
   - Heatmap values: Customer count per cell
   - Risk banding based on churn probability

3. **Action Distribution Donut**
   - Retention actions needed (Retain, Review, At Risk, Monitor)
   - Proportional sizing by customer count
   - Supports decision prioritization

**Data Requirements:**
- CSV file upload OR previous session data retention
- Columns: 19 customer features (demographics, account, services)

**Features:**
- Auto-loads batch data from session state
- Shows "Please upload CSV" guidance if no data
- Real-time KPI calculations
- Responsive 6-column layout for metrics
- Color-coded status indicators (🟢🔴)

---

### 👤 Page 02: Single Customer Prediction

**Purpose:** Individual customer churn prediction with detailed explanation

**Input Form (19 Fields)** organized in 5 sections:

**1. Demographic Information** (4 fields)
- Gender: [Male / Female]
- SeniorCitizen: [0 / 1]
- Partner: [Yes / No]
- Dependents: [Yes / No]

**2. Account Information** (3 fields)
- Tenure (months): [0-72] numeric slider
- MonthlyCharges ($): [0-150] numeric input
- TotalCharges ($): [0-10000] numeric input

**3. Phone & Internet Services** (6 fields)
- PhoneService: [Yes / No]
- MultipleLines: [Yes / No / No phone service]
- InternetService: [DSL / Fiber optic / No]
- OnlineSecurity: [Yes / No / No internet service]
- OnlineBackup: [Yes / No / No internet service]
- DeviceProtection: [Yes / No / No internet service]

**4. Support & Streaming Services** (5 fields)
- TechSupport: [Yes / No / No internet service]
- StreamingTV: [Yes / No / No internet service]
- StreamingMovies: [Yes / No / No internet service]
- Contract: [Month-to-month / One year / Two year]
- PaperlessBilling: [Yes / No]

**5. Payment Details** (1 field)
- PaymentMethod: [Electronic check / Mailed check / Bank transfer / Credit card]

**Output Section:**

**Churn Prediction Results:**
- **Churn Probability**: 0.0000 - 1.0000
- **Prediction**: "🔴 Will Likely Churn" / "🟢 Expected to Stay"
- **Threshold Used**: 0.4356 (optimized for F1 score)
- **Confidence Badge**: ✅ if model AUC > 0.80

**Revenue Impact Analysis:**
- **Revenue at Risk**: MonthlyCharges if churner
- **Annual Revenue Impact**: MonthlyCharges × 12
- **Retention Value**: Estimated MRR preservation

**SHAP Feature Explanation:**
- **Waterfall Chart**: 
  - X-axis: Feature names
  - Y-axis: SHAP value contribution
  - Base value → Feature impacts → Prediction
  - Green: Decreases churn probability
  - Red: Increases churn probability

- **Top 5 Contributing Features Table:**
  - Feature name
  - SHAP value (6 decimal places)
  - Direction (↑ Increases / ↓ Decreases)
  - Impact emoji (🔴 Positive / 🟢 Negative)

**Interactive Features:**
- ✅ Form auto-saves to session state
- ✅ Real-time prediction on "Predict" button
- ✅ Clear button to reset form
- ✅ Copy feature explanations for reporting

---

### 📤 Page 03: Batch Scoring

**Purpose:** Bulk customer analysis via CSV upload

**Upload Interface:**
- Drag-and-drop CSV or file picker
- Accepted format: CSV with 19 required columns
- Max file size: 200MB
- Required columns: gender, SeniorCitizen, Partner, Dependents, tenure, MonthlyCharges, TotalCharges, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod

**Processing:**
1. Load CSV into DataFrame
2. Extract customer ID (if present) or generate sequential IDs
3. Call batch prediction API
4. Return DataFrame with predictions and segmentation

**Output Results:**
- **Predictions DataFrame** with columns:
  - customerID
  - churn_probability (float 0-1)
  - is_churner (bool)
  - segment (0-3)
  - segment_label (Loyal High-Value / Low Engagement / etc.)
  - Original 19 input features
  - Engineered features (8 additional)

**Visualizations:**

**1. Predictive Distribution Histogram**
- X-axis: Churn probability (0-1) in bins
- Y-axis: Customer count
- Color: Gradient by risk band
- Threshold line at 0.4356
- Interactive: Hover for exact counts

**2. Segment Performance Metrics** (4 cards)
- Each segment shows:
  - Segment name
  - Customer count
  - Mean churn probability
  - Churn rate (%)
  - Color-coded risk level

**3. Revenue at Risk Summary**
- Total monthly recurring revenue at risk
- Breakdown by segment
- Breakdown by risk band
- Year-on-year projection

**Features:**
- ✅ Stores results in session state (st.session_state.batch_predictions)
- ✅ Data persists across pages within session
- ✅ Download results as CSV
- ✅ Show processing progress/spinner
- ✅ Error handling with user-friendly messages
- ✅ Empty CSV validation
- ✅ Data type conversion (numpy → Python types for API safety)

---

### 🎯 Page 04: Segment Intelligence

**Purpose:** Deep dive into customer segment characteristics and behaviors

**Segment Overview (4 Cards):**
Each card displays:
- Segment name with emoji icon
- Customer count and percentage
- "Characteristics" expandable section
- Median values for key metrics

**Per-Segment Analytics:**

For each of 4 segments:
1. **Demographic Profile**
   - Gender distribution (pie chart)
   - Age statistics (SeniorCitizen breakdown)
   - Relationship status (Partner, Dependents)

2. **Account Characteristics**
   - Tenure distribution histogram
   - MonthlyCharges box plot
   - TotalCharges distribution

3. **Service Adoption** (Multi-column layout)
   - PhoneService penetration
   - InternetService type distribution
   - Support services adoption rate
   - Streaming service uptake

4. **Risk Metrics** (KPI cards)
   - Churn rate (%)
   - Mean churn probability
   - Revenue at risk
   - Customer lifetime value estimates

5. **Contract & Payment**
   - Contract type distribution (stacked bar)
   - Payment method preferences (pie chart)
   - Paperless billing adoption

**Comparative Analytics:**
- **Segment Comparison Table**
  - All 4 segments side-by-side
  - Key metrics: size, churn_rate, avg_churn_prob, revenue_at_risk
  - Sorting by any column
  - Conditional formatting (red/green)

**Special Features:**
- ✅ Responsive 2-column layout for 4 segments
- ✅ Expandable metric cards
- ✅ Color-coded by segment themes
- ✅ Dynamically loads from batch data or provides test data
- ✅ Missing data handling ("—" for unavailable metrics)

---

### ⚠️ Page 05: Churn Risk Analysis

**Purpose:** Detailed churn risk distribution and financial impact analysis

**Risk Band Overview:**
- **Low Risk** (<0.35): Stable customers, low churn risk
- **Medium Risk** (0.35-0.65): Customers requiring monitoring
- **High Risk** (≥0.65): Immediate intervention needed

**Visualizations:**

**1. Risk Distribution Histogram**
- X-axis: Churn probability (0.0-1.0) with 20 bins
- Y-axis: Customer count
- Color gradient: Green (safe) → Orange (warning) → Red (danger)
- Vertical threshold line at 0.4356 decision boundary
- Interactive tooltips with exact counts

**2. Risk Band Distribution Pie Chart**
- 3 slices: Low, Medium, High
- Percentages relative to total
- Color-coded: Green, Yellow, Red
- Clickable for filtering

**3. Segment × Risk Band Heatmap**
- X-axis: 4 customer segments
- Y-axis: 3 risk bands (Low/Medium/High)
- Cell values: Customer counts
- Color intensity: Risk level (dark = high risk)
- Helps identify segment-specific risks

**4. Revenue at Risk Analysis**
- **Total at Risk**: Sum of MonthlyCharges for high-risk churners
- **By Risk Band**: Breakdown of revenue by risk level
- **By Segment**: Which segments contribute most risk
- **Cumulative Impact**: Year, 2-year projections

**5. Customer Count by Risk**
- Bar chart: Count per band (Low/Medium/High)
- Overlaid size comparison
- Stacked by segment option

**Metrics Tables:**
- **Risk Band Summary**
  - Band name
  - Customer count
  - Percentage of total
  - Average churn probability
  - Total revenue at risk
  - Average customer LTV

- **Segment Risk Crosswalk**
  - Which segments in which risk bands
  - Action recommendations

**Features:**
- ✅ Dynamic thresholds (customizable via business rules)
- ✅ Confidence intervals where applicable
- ✅ Export risk assessment CSV
- ✅ Scenario planning: "What if threshold moves to 0.50?"
- ✅ Historical tracking (if data retained across sessions)

---

### 📋 Page 06: Action Planning

**Purpose:** Targeted customer actions and intervention strategies

**Filterable Customer Table:**
- Rows: One per customer in batch
- Columns: 20+ including:
  - customerID
  - churn_probability (sortable)
  - is_churner (Yes/No)
  - segment (Loyal / Low Engagement / etc.)
  - tenure, MonthlyCharges, Contract
  - Recommended action
  - Priority level
  - Last action (metadata)

**Filtering Options (Multi-select):**
- **Risk Band**: Low / Medium / High
- **Segment**: All 4 segment options
- **Churn Status**: Predicted Churner / Predicted Loyal
- **Contract Type**: Month-to-month / One year / Two year
- **Internet Service**: DSL / Fiber / None
- **Action Category**: Retention offer / Service upgrade / loyalty program / etc.

**Action Recommendations:**
- **High Risk**: Immediate outreach, special offer, executive contact
- **Medium Risk**: Monitor closely, offer incentive, service review
- **Low Risk**: Standard service, watch for changes
- **Loyal High-Value**: Delight programs, VIP treatment

**Table Features:**
- ✅ Sortable columns (click header)
- ✅ Column visibility toggle
- ✅ Search/filter within table
- ✅ Select multiple rows for batch actions
- ✅ Export filtered results as CSV
- ✅ Row highlighting by risk level
- ✅ Pagination (default 50 rows per page)

**Batch Action Operations:**
- Generate action list for CRM upload
- Export customer segments for targeting campaigns
- Create outreach priorities ranked by revenue impact
- Generate retention strategy templates

**Reporting:**
- **Action Summary Report**
  - Action type → Customer count → Revenue impact
  - Timeline recommendations
  - Resource allocation guidance

---

### 🎲 Page 07: What-If Simulator

**Purpose:** Test retention strategies by modifying customer features and simulating outcomes

**Dual Input Modes:**

**Mode 1: Manual Entry**
- Use form from "Single Prediction" page
- Modify any of 19 features
- Simulate scenario on demand

**Mode 2: Load from Batch Data**
- Dropdown of customers from uploaded CSV
- Click "Load" to fetch all 19 fields pre-filled
- Modify specific fields to test scenarios
- Auto-converts numpy types to Python (JSON-safe)

**Scenario Modification Workflow:**
1. Load customer (manual or from batch)
2. Identify high-impact features:
   - Red features = increase churn
   - Green features = decrease churn
3. Modify one or more features:
   - Upfront discount → MonthlyCharges -$10
   - Contract upgrade → Contract: "Two year"
   - Service bundle → Add StreamingTV, StreamingMovies
4. Click "Simulate Scenario"

**Simulation Output:**

**1. Original vs Modified Prediction**
- Side-by-side comparison cards
- Original churn probability
- Modified churn probability
- Delta (change in probability)
- Status change: "Will likely stay now!" / "Still at risk"

**2. Impact Metrics**
- **Probability Change**: +/- X percentage points
- **Risk Band Movement**: High → Medium / Medium → Low
- **Binary Status**: Churn / No Churn
- **Confidence**: Based on model AUC

**3. Feature Contribution (Waterfall)**
- Shows which features changed
- Impact on final prediction
- Identify optimal intervention points

**4. Recommendation Engine**
- "You would need to lower MonthlyCharges by $20 to prevent churn"
- "Contract upgrade alone would reduce risk by 15%"
- "Combination of discounts required for this segment"

**5. Cost-Benefit Analysis**
- **Intervention Cost**: Estimated retention cost
- **Customer LTV**: Lifetime revenue if retained
- **ROI**: LTV / Intervention cost ratio
- **Recommendation**: Worth it? / Not worth it

**Features:**
- ✅ Real-time simulation (sub-second response)
- ✅ JSON serialization fix for numpy types (critical fix)
- ✅ Compare multiple scenarios (save snapshots)
- ✅ Export recommendation to CSV
- ✅ A/B test planning: "What if we offer 20% vs 30% discount?"
- ✅ Undo/redo scenario changes
- ✅ Scenario history (session-based)

---

### 🔍 Page 08: Explainability & Model Insights

**Purpose:** Model transparency through global and per-instance SHAP explanations

**Section 1: Global Feature Importance**

**What It Shows:**
- Which features matter most across ALL customers
- Feature importance ranking (top 10)
- Direction of impact (positive = increases churn, negative = decreases churn)

**Visualizations:**

**1. Feature Importance Bar Chart**
- X-axis: Features (tenure, MonthlyCharges, Contract, etc.)
- Y-axis: Average absolute SHAP value
- Color: Green (decreases churn) / Red (increases churn)
- Sorted by magnitude
- Interactive: Hover for exact values

**2. Feature Importance Table**
| Feature | Importance Score | Impact Direction | Interpretation |
|---------|------------------|------------------|-----------------|
| Contract | 0.0842 | Negative | Month-to-month → higher churn risk |
| Tenure | 0.0693 | Negative | Longer tenure → lower churn risk |
| MonthlyCharges | 0.0645 | Positive | Higher charges → more churn |
| ... | ... | ... | ... |

**3. Model Information Card**
- **Sample Size**: 7032 training samples
- **Explainer Type**: SHAP TreeExplainer (Fast)
- **Background Samples**: 200 used for baseline

---

**Section 2: Per-Customer SHAP Explanation**

**Input Methods:**

1. **Manual Entry**
   - Use integrated form (19 fields)
   - "Get Explanation" button

2. **Batch Data Selection**
   - Dropdown of customers from uploaded CSV
   - Auto-loads all 19 fields
   - Type conversion for JSON safety (critical fix)

**Output:**

**1. SHAP Force Plot (Waterfall Chart)**
- Shows prediction breakdown for ONE customer
- Base value: Model baseline prediction
- Features ranked by impact magnitude
- Color: Red (push toward churn) / Blue (push toward stay)
- Direction: Horizontal bars show contribution direction
- Final value: Model's prediction for this customer

Example layout:
```
Base value: 0.35
  ↓ (Contract = Month-to-month) +0.08 → High churn risk
  ↑ (Tenure = 24 months) -0.05 → Loyalty factor
  ↓ (MonthlyCharges = $95) +0.06 → Cost sensitivity
  ...
Prediction: 0.58 (High Risk)
```

**2. Feature Contributions Table**
| Feature | SHAP Value | Direction | Impact |
|---------|------------|-----------|--------|
| Contract | +0.0847 | ↑ | Increases churn by 8.5% |
| MonthlyCharges | +0.0543 | ↑ | Increases churn by 5.4% |
| Tenure | -0.0692 | ↓ | Decreases churn by 6.9% |

**3. Prediction Explanation**
- "This customer has 58% churn probability (***High Risk***)"
- Key drivers:
  1. Month-to-month contract (biggest risk factor)
  2. Limited tenure history
  3. Generic services (no support bundles)
- Mitigation strategies:
  1. Offer contract upgrade incentive
  2. Bundle support services
  3. Personalized outreach

**4. Model Details**
- **Framework**: LightGBM (Decision Tree Ensemble)
- **Tree Count**: 650 trees
- **Feature Engineering**: 33 engineered features (from 20 raw)
- **Train/Val/Test AUC**: 0.8791 / 0.8732 / 0.8398
- **Decision Threshold**: 0.4356

**Features:**
- ✅ Loads actual feature names (not "feature_32")
- ✅ Handles batch data with type conversion
- ✅ SHAP values cached for fast redraw
- ✅ Export explanation as PNG/PDF
- ✅ Compare: Customer A vs Customer B explanations
- ✅ Feature importance stability analysis

---

### 🏥 Page 09: Model Health & Metadata

**Purpose:** Monitor model performance and system health

**Section 1: Churn Model Overview**

**Model Architecture Cards:**
- **Model Name**: LightGBM Classifier
- **Framework**: LightGBM (Gradient Boosting)
- **Input Features**: 20 (19 customer + 1 segment)
- **Processed Features**: 33 (after engineering)
- **Number of Trees**: 650
- **Max Tree Depth**: 13
- **Decision Threshold**: 0.4356 (product-optimized)
- **Training Data Size**: 7,032 customers

---

**Section 2: Model Performance Metrics**

**Test Set Metrics** (Most representative):
| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **AUC-ROC** | 0.8398 | Excellent discrimination between churners/loyal |
| **Accuracy** | 0.7583 | 75.8% overall correctness |
| **Precision** | 0.5301 | 53% of predicted churners are true churners |
| **Recall** | 0.7857 | 78.6% of actual churners correctly identified |
| **F1 Score** | 0.6331 | Balanced precision-recall performance |

**Color-Coded Interpretation:**
- AUC > 0.85: ✅ Excellent
- AUC > 0.75: ✓ Good
- AUC < 0.75: ⚠️ Fair

**Train/Validation/Test Breakdown:**
- Expandable section showing metrics across all 3 splits
- Identifies overfitting/underfitting patterns
- Variance analysis

---

**Section 3: Feature Engineering Details**

**Expandable: "Feature Engineering Pipeline"**
- Raw input: 20 features
- Drop target column: 19 features
- Categorical encoding: +14 features (one-hot)
- Numerical engineering: +8 features (interactions, aggregates)
  - avg_monthly_spend
  - charge_gap
  - streaming_count
  - security_count
  - [etc.]
- **Total final**: 33 features sent to LightGBM

---

**Section 4: Customer Segmentation Model**

**Segmentation Model Cards:**
- **Model Name**: KMeans Clustering
- **Framework**: scikit-learn
- **Number of Clusters**: 4 segments
- **Algorithm**: KMeans with k-means++
- **Initialization**: 10 runs (n_init=10)
- **Random Seed**: 42 (reproducible)
- **Training Data**: 7,032 customers (same as churn model)

**Segment Definitions:**

| Segment | Name | Description | Size | Risk Profile |
|---------|------|-------------|------|--------------|
| 0 | Loyal High-Value | Long tenure, high spend, low churn | 📊 % | 🟢 Low |
| 1 | Low Engagement | New/inactive, low spend, high churn | 📊 % | 🔴 High |
| 2 | Stable Mid-Value | Moderate tenure, steady spend | 📊 % | 🟡 Medium |
| 3 | At-risk High-Value | High spend but showing churn signals | 📊 % | 🔴 High |

**Each card clickable to:**
- View detailed segment statistics
- See recommended actions for segment
- Filter for segment-specific analysis
- Export segment customer list

---

**Section 5: Explainability Engine Configuration**

**SHAP Explainer Details:**
- **Type**: SHAP (SHapley Additive exPlanations)
- **Explainer Algorithm**: TreeExplainer (Fast, for LightGBM)
- **Background Samples**: 200 (for baseline calculation)
- **Computation Speed**: ~200ms per instance explanation
- **Feature Importance Method**: SHAP Mean |values|

**Features:**
- ✅ SHAP values computed on-demand (cached)
- ✅ Handles missing features gracefully
- ✅ Explains both predictions and segments

---

**Section 6: System Health & Status**

**API Status:**
- **Status**: 🟢 Running / 🔴 Offline
- **Response Time**: [X ms] (average)
- **Last Sync**: [timestamp]
- **Model Load Time**: [X seconds at startup]

**Data Statistics:**
- **Training Set Size**: 7,032 samples
- **Features Used**: 20 raw (33 after engineering)
- **Training Date**: [from metadata]
- **Last Updated**: [from metrics_latest.json]

**Feature Coverage:**
- All 19 required fields present ✅
- No missing value imputation needed ✅
- Categorical encoding verified ✅

---

## Cross-Page Features & Integration

### 🔄 Session State Management

All pages share state:
- **batch_predictions**: DataFrame with all predictions
- **api_client**: Shared API connection
- **session_cache**: SHAP values, model info caching

Benefits:
- Upload once on Page 3, use everywhere
- Fast cross-page navigation
- Consistent data throughout session

### 📊 Data Flow Architecture

```
CSV Upload (Page 3)
    ↓
Batch Predictions → Session State
    ↓
Used by: Page 1, 4, 5, 6, 7, 8
```

### 🔌 API Integration

All pages call FastAPI endpoints:
- `/batch-predictions`: Page 3
- `/predict`: Page 2, 7
- `/feature-importance/global`: Page 8
- `/feature-importance/instance`: Page 2, 8
- `/explanations/model-info`: Page 9
- `/health`: Page 1

### 🎨 UI/UX Patterns

**Consistent throughout:**
1. **Section dividers** (st.divider()) for flow
2. **Color coding**: 
   - 🔴 Red = High churn risk
   - 🟡 Orange = Medium churn risk
   - 🟢 Green = Low churn risk
3. **Expandable sections** for detail exploration
4. **Responsive columns** (auto-layout)
5. **Metric cards** (KPI metric visualizations)
6. **Error handling** with user-friendly messages
7. **Spinner feedback** on async operations

---

## Critical Bug Fixes Applied

### 1. ✅ JSON Serialization (Batch What-If)
- **Issue**: `Object of type int64 is not JSON serializable`
- **Cause**: Pandas `.iloc[0]` returns numpy types
- **Fix**: Added `convert_numpy_to_python()` helper in validators.py
- **Pages affected**: 7, 8

### 2. ✅ API Response Key Extraction (What-If)
- **Issue**: SHAP results empty despite API success
- **Cause**: Extracted non-existent "scenario" key
- **Fix**: Changed to extract full response object
- **Page affected**: 7

### 3. ✅ Batch DataFrame Schema
- **Issue**: Only 7/19 customer fields extracted
- **Cause**: prepare_batch_result_df() incomplete
- **Fix**: Added all 19 customer input fields extraction
- **Page affected**: All batch-dependent pages

### 4. ✅ SHAP Key Names in UI
- **Issue**: KeyError on `f["shap_value"]`
- **Cause**: API returns `f["importance"]` instead
- **Fix**: Updated key access in pages 2 & 8
- **Pages affected**: 2, 8

### 5. ✅ Global Feature Names  
- **Issue**: Showing "feature_32" instead of "tenure"
- **Cause**: Feature names not extracted from preprocessor
- **Fix**: Updated SHAP explainer to use `get_feature_names_out()`
- **Page affected**: 8

### 6. ✅ Model Health Data Missing
- **Issue**: Model type, metrics all showing "Unknown"
- **Cause**: API endpoint returning incomplete data
- **Fix**: Updated to load actual metrics from JSON
- **Page affected**: 9

---

## Data Requirements

### Input CSV Format
**Required columns** (19 fields):
```
customerID, gender, SeniorCitizen, Partner, Dependents,
tenure, MonthlyCharges, TotalCharges, PhoneService, MultipleLines,
InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod
```

### Data Types
- **Numeric**: tenure, MonthlyCharges, TotalCharges
- **Categorical**: All others (Yes/No or specific values)

### Size Limits
- Max file size: 200MB
- Max recommended rows: 100,000 (memory constraints)
- Current training data: 7,032 rows

---

## Performance & Technical Specs

| Aspect | Specification |
|--------|---------------|
| **Streamlit Version** | 1.28.1 |
| **FastAPI Version** | Latest (async) |
| **Python** | 3.12+ |
| **Single Prediction Latency** | ~200-500ms |
| **Batch Processing** | ~10-20ms per customer |
| **SHAP Explanation Latency** | ~200-300ms |
| **API Health Check** | <50ms |
| **Session State Retention** | Entire browser session |

---

## Usage Recommendations

### For Data Analysts
1. Start with **Page 1: Overview** for KPI summary
2. Use **Page 4: Segment Intelligence** for deep-dive analytics
3. Export segment lists from **Page 6: Action Planning**

### For Business Leaders
1. **Page 1** for executive dashboard
2. **Page 5** for revenue-at-risk calculations
3. **Page 6** for prioritized action lists

### For Data Scientists
1. **Page 8: Explainability** for model transparency
2. **Page 9: Model Health** for performance monitoring
3. **Page 7: What-If** for feature impact testing

### For Customer Success Teams
1. **Page 2**: Single customer predictions with explanation
2. **Page 7**: A/B test interventions before deployment
3. **Page 6**: Filtered action lists for outreach campaigns

---

## Deployment Checklist

- [ ] API running on port 8000 (uvicorn api.app:app)
- [ ] Streamlit accessible on port 8501
- [ ] CSV test data prepared (test_batch.csv)
- [ ] Models loaded: churn (LightGBM) + segmentation (KMeans)
- [ ] SHAP explainer initialized with 200 background samples
- [ ] Metrics JSON readable (models/churn/metrics_latest.json)
- [ ] Segment labels JSON readable (models/segmentation/segment_labels.json)
- [ ] API health check returning 200 OK
- [ ] All 9 pages rendering without errors
- [ ] Type conversion working (numpy → Python types)

---

## Support & Troubleshooting

| Issue | Resolution |
|-------|-----------|
| "Models not loaded" | Start API with uvicorn first |
| "Object of type int64" | Type conversion should handle this (fixed) |
| "Unknown model type" | Check metrics_latest.json exists |
| Slow predictions | May indicate SHAP background sampling |
| CSV upload fails | Verify 19 columns present, correct names |

---

**Report Generated**: April 14, 2026  
**Dashboard Version**: 1.0  
**Status**: ✅ Production Ready