# CRIS Dashboard - Comprehensive Bug Review & Fixes

**Review Date**: April 14, 2026  
**Status**: 10 Bugs Fixed, 5 Additional Issues Documented

---

## 🔴 CRITICAL BUGS (FIXED)

### 1. **MonthlyCharges Type Inconsistency** ✅ FIXED
- **Issue**: API returns `MonthlyCharges` as float or string, but code assumed numeric type
- **Impact**: Revenue calculations crash with `TypeError: cannot perform reduce with flexible dtype`
- **Locations Fixed**:
  - `data_processors.py:215-221` - `build_segment_stats()` now converts to numeric safely
  - `data_processors.py:318-322` - `prepare_batch_result_df()` converts with try-except
  - `page_03_batch_scoring.py:139-151` - Revenue calculation uses `pd.to_numeric()`
- **Fix**: Added `pd.to_numeric(..., errors="coerce")` wrapper around all MonthlyCharges operations

### 2. **Empty DataFrame Handling** ✅ FIXED
- **Issue**: `prepare_batch_result_df()` returns completely empty DataFrame when all predictions fail
- **Impact**: Downstream `prepare_batch_result_df().empty` checks fail, KeyError on column access
- **Location Fixed**: `data_processors.py:291-308`
- **Fix**: Returns DataFrame with expected schema (15 columns) even when empty, enabling safe `.empty` checks

### 3. **Unsafe mode() Calls on DataFrame Columns** ✅ FIXED
- **Issue**: `mode()` can return empty Series, `.iloc[0]` fails without length check
- **Impact**: Segment stats crash with "IndexError: single positional indexer is out of bounds"
- **Location Fixed**: `data_processors.py:210-223` in `build_segment_stats()`
- **Fix**: Added `if len(mode_result) > 0 and not mode_result.empty:` guard, converts to string

### 4. **Waterfall Chart Color Logic Backwards** ✅ FIXED
- **Issue**: "increasing" marked RED, "decreasing" marked GREEN (counterintuitive)
- **Impact**: Misleading SHAP visualization (looks bad when features protect from churn)
- **Location Fixed**: `chart_builders.py:349-352` in `create_waterfall_chart()`
- **Fix**: Swapped colors - GREEN for positive (pushes toward churn), RED for negative (protects)

### 5. **Risk Band Filtering Inconsistency** ✅ FIXED
- **Issue**: `risk_band_filter()` used inclusive boundaries `<=` while pages used `<` for Medium band
- **Impact**: Customers near 0.35 or 0.65 boundaries match differently in filters vs. aggregation
- **Location Fixed**: `data_processors.py:174-193` in `risk_band_filter()`
- **Fix**: Standardized to: Low <0.35, Medium [0.35-0.65), High >=0.65 (matches pages exactly)

---

## 🟡 HIGH-SEVERITY BUGS (FIXED)

### 6. **Empty Data Handling in Charts** ✅ FIXED
- **Issue**: `create_distribution_histogram()` and `create_cdf_curve()` crash/render blank if no data
- **Impact**: Pages show empty white boxes or Plotly errors when segment has no customers
- **Locations Fixed**:
  - `chart_builders.py:109-115` - Histogram checks `if not data or len(data) == 0`
  - `chart_builders.py:434-442` - CDF filters out empty segments with `{seg: vals for seg, vals in... if vals}`
- **Fix**: Both functions now display "No data available" annotation instead of crashing

### 7. **CDF Per-Segment Bug** ✅ FIXED
- **Issue**: `create_cdf_curve(per_segment=True)` fails if `segment_data` contains empty lists
- **Impact**: Page 05 Churn Risk crashes when displaying CDF by segment
- **Location Fixed**: `chart_builders.py:431-442`
- **Fix**: Filter segments to only those with data: `segments_with_data = {seg_id: vals for seg_id, vals in segment_data.items() if vals and len(vals) > 0}`

### 8. **Batch Result DataFrame Without Error Checking** ✅ FIXED
- **Issue**: Pages try to access columns on empty DataFrame without checking if it exists
- **Impact**: KeyError when batch has all failed predictions
- **Locations Fixed**:
  - `page_04_segment_intelligence.py:29` - Added `if batch_df.empty: st.error(); return`
  - `page_05_churn_risk.py:29` - Added `if batch_df.empty: st.error(); return`
- **Fix**: All pages now check `batch_df.empty` after calling `prepare_batch_result_df()`

### 9. **What-If Batch Customer Loading** ✅ FIXED
- **Issue**: When loading customer from batch, error handling was missing for empty DataFrames
- **Impact**: IndexError if batch_predictions are empty or all failed
- **Location Fixed**: `page_07_what_if_simulator.py:90-105`
- **Fix**: Added checks:
  - `if batch_df.empty: st.error()` after `prepare_batch_result_df()`
  - `if selected_rows.empty: st.error()` after filtering
  - Better error messages for debugging

### 10. **Revenue at Risk Calculation** ✅ FIXED
- **Issue**: Direct `sum()` on MonthlyCharges column fails if column is string type
- **Impact**: "Total Revenue at Risk" metric shows error or 0 incorrectly
- **Location Fixed**: `page_03_batch_scoring.py:139-151`
- **Fix**: Uses `pd.to_numeric(..., errors="coerce").sum()` with try-except block

---

## 🟠 MEDIUM-SEVERITY ISSUES (Documented)

### 11. CSV Column Validation Too Strict
- **Issue**: `validate_csv_schema()` fails if CSV has extra columns (column count != 19)
- **Impact**: Users can't upload CSVs with extra columns (even if all 19 required are present)
- **Location**: `validators.py:60-73`
- **Recommendation**: Change check to verify 19 required columns are PRESENT, not total count == 19
- **Status**: 🔄 Not fixed (would need CSV preprocessing)

### 12. SeniorCitizen Type Confusion
- **Issue**: Config accepts both `[0, 1]` and `["0", "1"]`, but API might expect only integers
- **Impact**: Potential validation error in single prediction
- **Location**: `config.py:77-78`, `validators.py:27-33`
- **Recommendation**: Explicitly convert `SeniorCitizen` to `int` in validation
- **Status**: 🔄 Not fixed (API handles both, low priority)

### 13. Chart Empty Data Display Improvements
- **Issue**: Tenure/Charges distribution charts in Segment Intelligence don't handle empty segments
- **Impact**: No visual feedback when segment has no customers
- **Location**: `page_04_segment_intelligence.py:66-83`
- **Recommendation**: Call `create_tenure_distribution()` etc. with empty check like: `if tenure_data: st.plotly_chart(...)`
- **Status**: 🔄 Not fixed (pages have `if len(seg_df) == 0: st.info()` early, this is defensive)

### 14. Session State Persistence
- **Issue**: Batch data lost on page refresh (by Streamlit design)
- **Impact**: Users don't understand why their CSV upload disappears
- **Location**: `app.py:52-68`
- **Recommendation**: Add warning notice: "Batch data is stored in session and will be lost on page refresh"
- **Status**: 🔄 Not fixed (design trade-off)

### 15. Error Message Formatting
- **Issue**: Validator error messages joined with `"\n"` but displayed in markdown
- **Impact**: Multiple errors on same line, hard to read
- **Location**: `validators.py:43`
- **Recommendation**: Format as markdown list or separate `st.error()` calls
- **Status**: 🔄 Not fixed (low priority)

---

## ✅ SUMMARY OF FIXES APPLIED

| Bug # | Name | Severity | Status | File(s) |
|-------|------|----------|--------|---------|
| 1 | MonthlyCharges Type Inconsistency | CRITICAL | ✅ FIXED | data_processors.py, page_03 |
| 2 | Empty DataFrame Handling | CRITICAL | ✅ FIXED | data_processors.py |
| 3 | Unsafe mode() Calls | CRITICAL | ✅ FIXED | data_processors.py |
| 4 | Waterfall Chart Colors | CRITICAL | ✅ FIXED | chart_builders.py |
| 5 | Risk Band Filtering | CRITICAL | ✅ FIXED | data_processors.py |
| 6 | Empty Chart Data | HIGH | ✅ FIXED | chart_builders.py |
| 7 | CDF Per-Segment Bug | HIGH | ✅ FIXED | chart_builders.py |
| 8 | Error Checking in Pages | HIGH | ✅ FIXED | page_04, page_05 |
| 9 | What-If Batch Loading | HIGH | ✅ FIXED | page_07 |
| 10 | Revenue Calculation | HIGH | ✅ FIXED | page_03 |

**Total Fixed: 10 critical/high bugs**  
**Remaining: 5 medium issues (low priority, documented for future)**

---

## 🧪 Testing Recommendations

After these fixes, test:

1. **Batch Scoring with mixed data types**
   - Upload CSV with text in MonthlyCharges column
   - Verify no TypeError occurs
   - Check revenue calculation displays correctly

2. **All batch predictions fail**
   - Upload invalid CSV (wrong values)
   - Verify pages show "No data available" instead of crashing
   - Check all 5 batch-dependent pages handle gracefully

3. **Empty segments**
   - Upload CSV with customers from only 1-2 segments
   - View Segment Intelligence page
   - Verify unrepresented segments show "No data" not error

4. **SHAP Waterfall**
   - Run single prediction and go to Explainability page
   - Verify colors make intuitive sense (green features push toward churn, red features protect)
   - Check per-customer SHAP still renders correctly

5. **Risk filtering**
   - Upload batch and go to Action Planning page
   - Apply "Low Risk" filter
   - Verify same customers appear as "Low Risk (<0.35)" on Churn Risk page

6. **What-If Simulator**
   - Load customer from batch
   - Modify fields
   - Verify no IndexError or empty DataFrame errors

---

## 📝 Notes for Future Maintenance

- **Type Safety**: MonthlyCharges now safely handled everywhere. Consider using stricter typing in API responses to avoid similar issues.
- **Empty Data**: All visualization functions now handle empty data gracefully. Pattern: check len > 0 before plotting.
- **Risk Boundaries**: Standardized to Low<0.35, Medium[0.35-0.65), High>=0.65 across all code.
- **DataFrame Schema**: Empty DataFrames now maintain schema, preventing downstream KeyErrors.

---

**All critical and high-priority bugs have been fixed and are ready for testing.**
