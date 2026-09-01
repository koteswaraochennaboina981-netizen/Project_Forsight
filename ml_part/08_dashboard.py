# ============================================================
# PROJECT FORESIGHT - PHASE 8: DASHBOARD / FINAL APPLICATION
# ============================================================
#
# This is the single front-end for the whole pipeline. Every
# artifact produced by Phases 1-7 (raw data profiling, EDA
# charts, engineered features, forecasts, inventory
# recommendations and model evaluation) is surfaced here, along
# with a rule-based "Current Status & Suggestions" engine that
# reads the latest numbers and turns them into plain-language
# status statements and action items.
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


# ============================================================
# [1] PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Project Foresight",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# [2] PROJECT PATHS
# ============================================================

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_PATH, "data")

EDA_PATH = os.path.join(DATA_PATH, "eda")
FEATURES_PATH = os.path.join(DATA_PATH, "features")
FORECAST_PATH = os.path.join(DATA_PATH, "forecasting")
INVENTORY_PATH = os.path.join(DATA_PATH, "inventory_optimization")
EVALUATION_PATH = os.path.join(DATA_PATH, "model_evaluation")


# ============================================================
# [3] FILE PATHS
# ============================================================

# ---- EDA (Phase 3) ----
DATASET_SUMMARY_FILE = os.path.join(EDA_PATH, "dataset_summary.csv")
EDA_INVENTORY_STATUS_FILE = os.path.join(EDA_PATH, "inventory_status.csv")
RISK_PRODUCTS_FILE = os.path.join(EDA_PATH, "risk_products.csv")
SKU_INVENTORY_RISK_FILE = os.path.join(EDA_PATH, "sku_inventory_risk.csv")
PROMOTION_ANALYSIS_FILE = os.path.join(EDA_PATH, "promotion_analysis.csv")
CATEGORY_SALES_FILE = os.path.join(EDA_PATH, "category_sales.csv")
BRAND_SALES_FILE = os.path.join(EDA_PATH, "brand_sales.csv")
TOP_SKUS_REVENUE_FILE = os.path.join(EDA_PATH, "top_skus_revenue.csv")
TOP_SKUS_UNITS_FILE = os.path.join(EDA_PATH, "top_skus_units.csv")
STORE_SALES_FILE = os.path.join(EDA_PATH, "store_sales.csv")
CITY_SALES_FILE = os.path.join(EDA_PATH, "city_sales.csv")
STORE_TYPE_SALES_FILE = os.path.join(EDA_PATH, "store_type_sales.csv")
SEASON_SALES_FILE = os.path.join(EDA_PATH, "season_sales.csv")
HOLIDAY_SALES_FILE = os.path.join(EDA_PATH, "holiday_sales.csv")
DAILY_SALES_FILE = os.path.join(EDA_PATH, "daily_sales.csv")
WEEKLY_SALES_FILE = os.path.join(EDA_PATH, "weekly_sales.csv")
MONTHLY_SALES_FILE = os.path.join(EDA_PATH, "monthly_sales.csv")

# ---- Feature engineering (Phase 4) ----
FEATURE_SUMMARY_FILE = os.path.join(FEATURES_PATH, "feature_summary.csv")
FEATURE_DICTIONARY_FILE = os.path.join(FEATURES_PATH, "feature_dictionary.csv")

# ---- Forecasting (Phase 5) ----
FORECAST_FILE = os.path.join(FORECAST_PATH, "demand_forecasts.csv")
FORECAST_METRICS_FILE = os.path.join(FORECAST_PATH, "forecast_metrics.csv")
FORECAST_SUMMARY_FILE = os.path.join(FORECAST_PATH, "forecast_summary.csv")
MODEL_COMPARISON_FILE = os.path.join(FORECAST_PATH, "model_comparison.csv")
FEATURE_IMPORTANCE_FILE = os.path.join(FORECAST_PATH, "feature_importance.csv")

# ---- Inventory optimization (Phase 6) ----
INVENTORY_RECOMMENDATIONS_FILE = os.path.join(INVENTORY_PATH, "inventory_recommendations.csv")
INVENTORY_RISK_FILE = os.path.join(INVENTORY_PATH, "inventory_risk_analysis.csv")
INVENTORY_METRICS_FILE = os.path.join(INVENTORY_PATH, "inventory_metrics.csv")
INVENTORY_SUMMARY_FILE = os.path.join(INVENTORY_PATH, "inventory_summary.csv")

# ---- Model evaluation (Phase 7) ----
OVERALL_FORECAST_EVALUATION_FILE = os.path.join(EVALUATION_PATH, "overall_forecast_evaluation.csv")
SKU_EVALUATION_FILE = os.path.join(EVALUATION_PATH, "sku_forecast_evaluation.csv")
STORE_EVALUATION_FILE = os.path.join(EVALUATION_PATH, "store_forecast_evaluation.csv")
ERROR_ANALYSIS_FILE = os.path.join(EVALUATION_PATH, "forecast_error_analysis.csv")
ERROR_SUMMARY_FILE = os.path.join(EVALUATION_PATH, "forecast_error_summary.csv")
INVENTORY_EVALUATION_FILE = os.path.join(EVALUATION_PATH, "inventory_evaluation.csv")
INVENTORY_STATUS_FILE = os.path.join(EVALUATION_PATH, "inventory_status_evaluation.csv")
REPLENISHMENT_EVALUATION_FILE = os.path.join(EVALUATION_PATH, "replenishment_evaluation.csv")
HIGH_RISK_FILE = os.path.join(EVALUATION_PATH, "high_risk_inventory.csv")
MODEL_PERFORMANCE_FILE = os.path.join(EVALUATION_PATH, "model_performance_summary.csv")

# ---- EDA chart images ----
EDA_CHARTS = {
    "daily_revenue_trend": os.path.join(EDA_PATH, "daily_revenue_trend.png"),
    "weekly_revenue_trend": os.path.join(EDA_PATH, "weekly_revenue_trend.png"),
    "monthly_revenue_trend": os.path.join(EDA_PATH, "monthly_revenue_trend.png"),
    "category_revenue": os.path.join(EDA_PATH, "category_revenue.png"),
    "top_10_brands": os.path.join(EDA_PATH, "top_10_brands.png"),
    "top_10_skus_revenue": os.path.join(EDA_PATH, "top_10_skus_revenue.png"),
    "top_10_skus_units": os.path.join(EDA_PATH, "top_10_skus_units.png"),
    "top_10_stores_revenue": os.path.join(EDA_PATH, "top_10_stores_revenue.png"),
    "city_revenue": os.path.join(EDA_PATH, "city_revenue.png"),
    "store_type_revenue": os.path.join(EDA_PATH, "store_type_revenue.png"),
    "promotion_vs_nonpromotion": os.path.join(EDA_PATH, "promotion_vs_nonpromotion.png"),
    "holiday_revenue": os.path.join(EDA_PATH, "holiday_revenue.png"),
    "seasonal_revenue": os.path.join(EDA_PATH, "seasonal_revenue.png"),
    "correlation_matrix": os.path.join(EDA_PATH, "correlation_matrix.png"),
    "revenue_distribution": os.path.join(EDA_PATH, "revenue_distribution.png"),
    "sales_distribution": os.path.join(EDA_PATH, "sales_distribution.png"),
    "price_vs_demand": os.path.join(EDA_PATH, "price_vs_demand.png"),
    "inventory_health": os.path.join(EDA_PATH, "inventory_health.png"),
}


# ============================================================
# [4] HELPER FUNCTIONS
# ============================================================

def load_csv(file_path):
    """Safely load a CSV file. Returns None if it does not exist."""

    if not os.path.exists(file_path):
        return None

    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


def find_column(df, candidates):
    """Return the first matching column from candidates."""

    if df is None:
        return None

    for column in candidates:
        if column in df.columns:
            return column

    return None


def format_number(value):
    """Format numbers for dashboard display."""

    if value is None or pd.isna(value):
        return "N/A"

    try:
        value = float(value)

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"{value / 1_000:.2f}K"

        return f"{value:,.2f}"

    except Exception:
        return str(value)


def format_integer(value):
    """Format integer-like values."""

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return str(value)


def format_currency(value):
    """Format currency values."""

    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"Rs. {format_number(value)}"
    except Exception:
        return str(value)


def safe_sum(df, column):
    """Safely calculate column sum."""

    if df is None or column not in df.columns:
        return 0.0

    return pd.to_numeric(df[column], errors="coerce").fillna(0).sum()


def safe_mean(df, column):
    """Safely calculate column mean."""

    if df is None or column not in df.columns:
        return np.nan

    return pd.to_numeric(df[column], errors="coerce").mean()


def safe_metric_from_pairs(df, metric_column, value_column, metric_name, default=np.nan):
    """
    Look up a value from a long-format 'metric, value' style CSV
    (used by forecast_summary.csv / inventory_summary.csv / etc.)
    """

    if df is None or metric_column not in df.columns or value_column not in df.columns:
        return default

    matches = df[df[metric_column].astype(str) == str(metric_name)]

    if len(matches) == 0:
        return default

    value = pd.to_numeric(matches[value_column].iloc[0], errors="coerce")

    return value if pd.notna(value) else default


def show_image_if_exists(path, caption=None):
    """Render an EDA chart image if the file is present on disk."""

    if path and os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
        return True

    st.info(f"Chart not found: `{os.path.basename(path) if path else 'unknown'}`")
    return False


def render_status(level, message):
    """Render a status/suggestion line using the right Streamlit widget."""

    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)


# ============================================================
# [5] LOAD DATA
# ============================================================

# ---- EDA ----
dataset_summary_df = load_csv(DATASET_SUMMARY_FILE)
eda_inventory_status_df = load_csv(EDA_INVENTORY_STATUS_FILE)
risk_products_df = load_csv(RISK_PRODUCTS_FILE)
sku_inventory_risk_df = load_csv(SKU_INVENTORY_RISK_FILE)
promotion_analysis_df = load_csv(PROMOTION_ANALYSIS_FILE)
category_sales_df = load_csv(CATEGORY_SALES_FILE)
brand_sales_df = load_csv(BRAND_SALES_FILE)
top_skus_revenue_df = load_csv(TOP_SKUS_REVENUE_FILE)
top_skus_units_df = load_csv(TOP_SKUS_UNITS_FILE)
store_sales_df = load_csv(STORE_SALES_FILE)
city_sales_df = load_csv(CITY_SALES_FILE)
store_type_sales_df = load_csv(STORE_TYPE_SALES_FILE)
season_sales_df = load_csv(SEASON_SALES_FILE)
holiday_sales_df = load_csv(HOLIDAY_SALES_FILE)
daily_sales_df = load_csv(DAILY_SALES_FILE)
weekly_sales_df = load_csv(WEEKLY_SALES_FILE)
monthly_sales_df = load_csv(MONTHLY_SALES_FILE)

# ---- Feature engineering ----
feature_summary_df = load_csv(FEATURE_SUMMARY_FILE)
feature_dictionary_df = load_csv(FEATURE_DICTIONARY_FILE)

# ---- Forecasting ----
forecast_df = load_csv(FORECAST_FILE)
forecast_metrics_df = load_csv(FORECAST_METRICS_FILE)
forecast_summary_df = load_csv(FORECAST_SUMMARY_FILE)
model_comparison_df = load_csv(MODEL_COMPARISON_FILE)
feature_importance_df = load_csv(FEATURE_IMPORTANCE_FILE)

# ---- Inventory optimization ----
inventory_df = load_csv(INVENTORY_RECOMMENDATIONS_FILE)
inventory_risk_df = load_csv(INVENTORY_RISK_FILE)
inventory_metrics_df = load_csv(INVENTORY_METRICS_FILE)
inventory_summary_df = load_csv(INVENTORY_SUMMARY_FILE)

# ---- Model evaluation ----
overall_forecast_df = load_csv(OVERALL_FORECAST_EVALUATION_FILE)
sku_evaluation_df = load_csv(SKU_EVALUATION_FILE)
store_evaluation_df = load_csv(STORE_EVALUATION_FILE)
error_analysis_df = load_csv(ERROR_ANALYSIS_FILE)
error_summary_df = load_csv(ERROR_SUMMARY_FILE)
inventory_evaluation_df = load_csv(INVENTORY_EVALUATION_FILE)
inventory_status_df = load_csv(INVENTORY_STATUS_FILE)
replenishment_df = load_csv(REPLENISHMENT_EVALUATION_FILE)
high_risk_df = load_csv(HIGH_RISK_FILE)
model_performance_df = load_csv(MODEL_PERFORMANCE_FILE)


# ============================================================
# [6] DATA AVAILABILITY CHECK
# ============================================================

required_files = {
    "Demand forecasts": forecast_df,
    "Inventory recommendations": inventory_df,
    "Model performance": model_performance_df,
    "Inventory metrics": inventory_metrics_df,
    "SKU evaluation": sku_evaluation_df,
    "Store evaluation": store_evaluation_df,
    "High-risk inventory": high_risk_df,
}

missing_files = [name for name, dataframe in required_files.items() if dataframe is None]

optional_files = {
    "EDA charts (Phase 3)": len(EDA_CHARTS) if os.path.isdir(EDA_PATH) else 0,
    "Feature dictionary (Phase 4)": feature_dictionary_df,
    "Model comparison (Phase 5)": model_comparison_df,
    "Inventory risk analysis (Phase 6)": inventory_risk_df,
}


# ============================================================
# [7] SIDEBAR
# ============================================================

st.sidebar.title("📦 Project Foresight")

st.sidebar.markdown(
    """
### AI-Driven Inventory Management

Use the navigation below to explore every stage of the pipeline,
from raw-data EDA all the way to live restock suggestions.
"""
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Data Overview & EDA",
        "Feature Engineering",
        "Demand Forecasting",
        "Inventory Optimization",
        "Model Evaluation",
        "Risk Analysis",
        "SKU / Store Explorer",
        "Insights & Suggestions",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Project Foresight • Final Application • Phase 8")


# ============================================================
# [8] HEADER
# ============================================================

st.title("📦 Project Foresight")

st.subheader("AI-Driven Demand Forecasting & Inventory Optimization")

st.markdown(
    """
This dashboard presents the complete output of the Project Foresight
pipeline: raw-data exploration, engineered features, demand forecasts,
inventory optimization, model evaluation, and live, data-driven
restocking suggestions.
"""
)

st.markdown("---")


# ============================================================
# [9] FILE ERROR HANDLING (core files only)
# ============================================================

if len(missing_files) > 0:

    st.error("Some required Phase 5-7 output files are missing.")

    st.write("Missing datasets:")

    for file_name in missing_files:
        st.write(f"- {file_name}")

    st.info("Run Phases 5-7 first and then restart the dashboard.")

    st.stop()


# ============================================================
# [10] PREPARE COMMON COLUMNS
# ============================================================

forecast_df["date"] = pd.to_datetime(forecast_df["date"], errors="coerce")

actual_column = find_column(forecast_df, ["actual_units_sold", "units_sold", "actual"])
prediction_column = find_column(forecast_df, ["predicted_units_sold", "predicted_demand", "forecast", "prediction"])
sku_column = find_column(forecast_df, ["sku_id", "SKU_ID", "sku"])
store_column = find_column(forecast_df, ["store_id", "STORE_ID", "store"])


# ============================================================
# [11] CURRENT STATUS & SUGGESTIONS ENGINE
# ============================================================
#
# This section reads the latest numbers straight out of the
# loaded dataframes (no hard-coded values) and turns them into
# a "current status" summary plus a ranked list of suggested
# actions. It is reused on both the Executive Overview page and
# the dedicated Insights & Suggestions page.
# ============================================================

def build_status_and_suggestions():

    status_items = []
    suggestions = []

    total_combinations = len(inventory_df) if inventory_df is not None else 0

    # ---- Inventory health snapshot ----
    if inventory_df is not None and "inventory_status" in inventory_df.columns:

        status_counts = inventory_df["inventory_status"].astype(str).str.upper().value_counts()

        stockout_count = int(status_counts.get("STOCKOUT", 0))
        critical_count = int(status_counts.get("CRITICAL", 0))
        reorder_count = int(status_counts.get("REORDER", 0))
        overstock_count = int(status_counts.get("OVERSTOCK", 0))
        healthy_count = int(status_counts.get("HEALTHY", 0))

        stockout_pct = (stockout_count / total_combinations * 100) if total_combinations else 0
        critical_pct = (critical_count / total_combinations * 100) if total_combinations else 0
        overstock_pct = (overstock_count / total_combinations * 100) if total_combinations else 0

        status_items.append((
            "info" if stockout_pct < 20 else "warning" if stockout_pct < 50 else "error",
            f"**Inventory health:** {total_combinations:,} SKU-store combinations tracked — "
            f"{stockout_count:,} STOCKOUT ({stockout_pct:.1f}%), {critical_count:,} CRITICAL, "
            f"{reorder_count:,} REORDER, {overstock_count:,} OVERSTOCK, {healthy_count:,} HEALTHY."
        ))

        if stockout_count > 0:
            urgent_cost = safe_sum(
                inventory_df[inventory_df["inventory_status"].astype(str).str.upper() == "STOCKOUT"],
                "estimated_replenishment_cost"
            )
            suggestions.append((
                "error",
                f"🚨 **Act now:** {stockout_count:,} SKU-store combinations are in STOCKOUT. "
                f"Estimated replenishment cost to clear them is {format_currency(urgent_cost)}."
            ))

        if critical_count > 0:
            suggestions.append((
                "warning",
                f"⚠️ **Critical watch-list:** {critical_count:,} combinations are CRITICAL "
                f"(below safety stock). Prioritize these ahead of routine REORDER items."
            ))

        if overstock_count > 0:
            overstock_value = safe_sum(
                inventory_df[inventory_df["inventory_status"].astype(str).str.upper() == "OVERSTOCK"],
                "inventory_cost_value"
            )
            suggestions.append((
                "info",
                f"📦 **Working capital tied up:** {overstock_count:,} combinations are OVERSTOCK, "
                f"holding roughly {format_currency(overstock_value)} in inventory cost value. "
                f"Consider promotions or transfers to free this capital."
            ))

    # ---- Replenishment cost outlook ----
    total_order_qty = safe_metric_from_pairs(
        replenishment_df, "metric", "value", "total_recommended_order_quantity"
    )
    total_replenishment_cost = safe_metric_from_pairs(
        replenishment_df, "metric", "value", "total_estimated_replenishment_cost"
    )

    if pd.notna(total_replenishment_cost):
        status_items.append((
            "info",
            f"**Replenishment outlook:** {format_integer(total_order_qty)} units recommended for order, "
            f"at an estimated cost of {format_currency(total_replenishment_cost)}."
        ))

    # ---- Forecast accuracy ----
    if model_performance_df is not None and len(model_performance_df) > 0:

        model_row = model_performance_df.iloc[0]
        best_model = model_row.get("best_model", "N/A")
        wape = pd.to_numeric(model_row.get("test_WAPE_percent", np.nan), errors="coerce")
        smape = pd.to_numeric(model_row.get("test_sMAPE_percent", np.nan), errors="coerce")

        if pd.notna(wape):

            accuracy_level = "success" if wape < 10 else "warning" if wape < 25 else "error"

            status_items.append((
                accuracy_level,
                f"**Forecast accuracy:** best model is **{best_model}** with a test WAPE of "
                f"{wape:.2f}% and sMAPE of {smape:.2f}%."
            ))

            if wape >= 25:
                suggestions.append((
                    "warning",
                    f"📉 **Model needs attention:** WAPE of {wape:.2f}% is high. "
                    f"Consider retraining with more recent data or engineering additional features."
                ))
            elif wape < 10:
                suggestions.append((
                    "success",
                    f"✅ **Forecasting is reliable:** WAPE of {wape:.2f}% is strong — "
                    f"current inventory recommendations can be trusted with normal review cadence."
                ))

    # ---- Forecast error bias ----
    if error_summary_df is not None and "error_category" in error_summary_df.columns:

        error_lookup = error_summary_df.set_index("error_category")["record_count"].to_dict()

        under_count = int(error_lookup.get("UNDER_FORECAST", 0))
        over_count = int(error_lookup.get("OVER_FORECAST", 0))
        exact_count = int(error_lookup.get("EXACT", 0))
        total_errors = under_count + over_count + exact_count

        if total_errors > 0:

            status_items.append((
                "info",
                f"**Forecast bias:** {under_count:,} under-forecasted, {over_count:,} "
                f"over-forecasted, {exact_count:,} exact matches "
                f"({under_count / total_errors * 100:.1f}% / {over_count / total_errors * 100:.1f}% / "
                f"{exact_count / total_errors * 100:.1f}%)."
            ))

            if under_count > over_count * 1.3:
                suggestions.append((
                    "warning",
                    "📈 **Systematic under-forecasting detected:** the model under-predicts more often "
                    "than it over-predicts. This can quietly drive stockouts — consider raising safety "
                    "stock or adding a bias-correction term."
                ))
            elif over_count > under_count * 1.3:
                suggestions.append((
                    "info",
                    "📉 **Systematic over-forecasting detected:** the model over-predicts more often "
                    "than it under-predicts, which inflates recommended order quantities and ties up "
                    "extra capital — consider tightening reorder points."
                ))

    # ---- Worst-performing SKUs / stores ----
    if sku_evaluation_df is not None and "WAPE_percent" in sku_evaluation_df.columns:

        worst_skus = sku_evaluation_df.sort_values("WAPE_percent", ascending=False).head(5)

        if len(worst_skus) > 0 and "sku_id" in worst_skus.columns:
            sku_list = ", ".join(worst_skus["sku_id"].astype(str).tolist())
            suggestions.append((
                "info",
                f"🔍 **Investigate these SKUs:** highest forecast error (by WAPE) — {sku_list}."
            ))

    # ---- Promotion effectiveness (from EDA) ----
    if promotion_analysis_df is not None and "promotion_status" in promotion_analysis_df.columns:

        promo_lookup = promotion_analysis_df.set_index("promotion_status")

        if "Promotion" in promo_lookup.index and "No Promotion" in promo_lookup.index:

            promo_records = promo_lookup.loc["Promotion", "records"]
            promo_units = promo_lookup.loc["Promotion", "units_sold"]
            base_records = promo_lookup.loc["No Promotion", "records"]
            base_units = promo_lookup.loc["No Promotion", "units_sold"]

            promo_avg = promo_units / promo_records if promo_records else 0
            base_avg = base_units / base_records if base_records else 0

            if base_avg > 0:
                uplift_pct = (promo_avg - base_avg) / base_avg * 100

                suggestions.append((
                    "info",
                    f"🏷️ **Promotion effect:** promoted records average {promo_avg:.2f} units/record vs "
                    f"{base_avg:.2f} for non-promoted — a {uplift_pct:+.1f}% difference. "
                    f"Use this when planning promo-driven demand spikes."
                ))

    return status_items, suggestions


status_items, suggestions = build_status_and_suggestions()


# ============================================================
# [12] EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.header("📊 Executive Overview")

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    test_rows = len(forecast_df)
    unique_skus = forecast_df[sku_column].nunique() if sku_column else 0
    unique_stores = forecast_df[store_column].nunique() if store_column else 0
    total_actual = safe_sum(forecast_df, actual_column) if actual_column else 0
    total_forecast = safe_sum(forecast_df, prediction_column) if prediction_column else 0
    total_stock = safe_sum(inventory_df, "stock_on_hand")
    total_order = safe_sum(inventory_df, "recommended_order_quantity")

    stockout_count = 0

    if "inventory_status" in inventory_df.columns:
        stockout_count = int(
            (inventory_df["inventory_status"].astype(str).str.upper() == "STOCKOUT").sum()
        )

    # ---- KPI row 1 ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Forecast Records", format_integer(test_rows))
    with col2:
        st.metric("Unique SKUs", format_integer(unique_skus))
    with col3:
        st.metric("Unique Stores", format_integer(unique_stores))
    with col4:
        st.metric("Forecast Demand", format_number(total_forecast))

    # ---- KPI row 2 ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Actual Demand", format_number(total_actual))
    with col2:
        st.metric("Stock On Hand", format_integer(total_stock))
    with col3:
        st.metric("Recommended Order", format_integer(total_order))
    with col4:
        st.metric("Stockout Risk", format_integer(stockout_count))

    st.markdown("---")

    # --------------------------------------------------------
    # Current status
    # --------------------------------------------------------

    st.subheader("🩺 Current Status")

    for level, message in status_items:
        render_status(level, message)

    st.markdown("---")

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    st.subheader("💡 Suggestions & Recommended Actions")

    if len(suggestions) == 0:
        st.info("No suggestions to surface right now — everything looks nominal.")
    else:
        for level, message in suggestions:
            render_status(level, message)

    st.caption("See the **Insights & Suggestions** page for supporting tables and detail.")

    st.markdown("---")

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    st.subheader("🎯 Forecasting Model Performance")

    if model_performance_df is not None:

        model_row = model_performance_df.iloc[0]

        best_model = model_row.get("best_model", "N/A")
        mae = model_row.get("test_MAE", np.nan)
        rmse = model_row.get("test_RMSE", np.nan)
        smape = model_row.get("test_sMAPE_percent", np.nan)
        wape = model_row.get("test_WAPE_percent", np.nan)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Best Model", str(best_model))
        with col2:
            st.metric("MAE", f"{float(mae):.4f}")
        with col3:
            st.metric("RMSE", f"{float(rmse):.4f}")
        with col4:
            st.metric("sMAPE", f"{float(smape):.2f}%")
        with col5:
            st.metric("WAPE", f"{float(wape):.2f}%")

    st.markdown("---")

    # --------------------------------------------------------
    # Inventory status / replenishment priority charts
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📦 Inventory Status")

        if "inventory_status" in inventory_df.columns:
            status_counts = (
                inventory_df["inventory_status"]
                .value_counts()
                .rename_axis("status")
                .reset_index(name="count")
            )
            st.bar_chart(status_counts.set_index("status"))

    with col_b:
        st.subheader("🚚 Replenishment Priority")

        if "replenishment_priority" in inventory_df.columns:
            priority_counts = (
                inventory_df["replenishment_priority"]
                .value_counts()
                .rename_axis("priority")
                .reset_index(name="count")
            )
            st.bar_chart(priority_counts.set_index("priority"))


# ============================================================
# [13] DATA OVERVIEW & EDA
# ============================================================

elif page == "Data Overview & EDA":

    st.header("🔬 Data Overview & Exploratory Data Analysis")

    st.write(
        "Every chart and summary table produced during Phase 2-3 "
        "(data cleaning & EDA) is reproduced below."
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    if dataset_summary_df is not None:
        st.subheader("📋 Dataset Summary")
        st.dataframe(dataset_summary_df, use_container_width=True)

    tabs = st.tabs([
        "Sales Trends",
        "Category / Brand / Store",
        "Promotions, Holidays & Seasonality",
        "Statistical Analysis",
        "Inventory Health & Risk (EDA)",
    ])

    # --------------------------------------------------------
    # Tab 1: Sales trends
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader("Daily Revenue Trend")
        show_image_if_exists(EDA_CHARTS["daily_revenue_trend"])
        if daily_sales_df is not None:
            with st.expander("View daily sales data"):
                st.dataframe(daily_sales_df, use_container_width=True, height=300)

        st.subheader("Weekly Revenue Trend")
        show_image_if_exists(EDA_CHARTS["weekly_revenue_trend"])
        if weekly_sales_df is not None:
            with st.expander("View weekly sales data"):
                st.dataframe(weekly_sales_df, use_container_width=True, height=300)

        st.subheader("Monthly Revenue Trend")
        show_image_if_exists(EDA_CHARTS["monthly_revenue_trend"])
        if monthly_sales_df is not None:
            with st.expander("View monthly sales data"):
                st.dataframe(monthly_sales_df, use_container_width=True, height=300)

    # --------------------------------------------------------
    # Tab 2: Category / brand / store performance
    # --------------------------------------------------------

    with tabs[1]:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Revenue by Category")
            show_image_if_exists(EDA_CHARTS["category_revenue"])
            if category_sales_df is not None:
                st.dataframe(category_sales_df, use_container_width=True, height=250)

        with col2:
            st.subheader("Top 10 Brands")
            show_image_if_exists(EDA_CHARTS["top_10_brands"])
            if brand_sales_df is not None:
                st.dataframe(brand_sales_df, use_container_width=True, height=250)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Top 10 SKUs by Revenue")
            show_image_if_exists(EDA_CHARTS["top_10_skus_revenue"])
            if top_skus_revenue_df is not None:
                st.dataframe(top_skus_revenue_df, use_container_width=True, height=250)

        with col4:
            st.subheader("Top 10 SKUs by Units")
            show_image_if_exists(EDA_CHARTS["top_10_skus_units"])
            if top_skus_units_df is not None:
                st.dataframe(top_skus_units_df, use_container_width=True, height=250)

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Top 10 Stores by Revenue")
            show_image_if_exists(EDA_CHARTS["top_10_stores_revenue"])
            if store_sales_df is not None:
                st.dataframe(store_sales_df, use_container_width=True, height=250)

        with col6:
            st.subheader("Revenue by Store Type")
            show_image_if_exists(EDA_CHARTS["store_type_revenue"])
            if store_type_sales_df is not None:
                st.dataframe(store_type_sales_df, use_container_width=True, height=250)

        st.subheader("Revenue by City")
        show_image_if_exists(EDA_CHARTS["city_revenue"])
        if city_sales_df is not None:
            st.dataframe(city_sales_df, use_container_width=True, height=250)

    # --------------------------------------------------------
    # Tab 3: Promotions, holidays, seasonality
    # --------------------------------------------------------

    with tabs[2]:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Promotion vs Non-Promotion")
            show_image_if_exists(EDA_CHARTS["promotion_vs_nonpromotion"])
            if promotion_analysis_df is not None:
                st.dataframe(promotion_analysis_df, use_container_width=True)

        with col2:
            st.subheader("Holiday vs Non-Holiday Revenue")
            show_image_if_exists(EDA_CHARTS["holiday_revenue"])
            if holiday_sales_df is not None:
                st.dataframe(holiday_sales_df, use_container_width=True)

        st.subheader("Seasonal Revenue")
        show_image_if_exists(EDA_CHARTS["seasonal_revenue"])
        if season_sales_df is not None:
            st.dataframe(season_sales_df, use_container_width=True)

    # --------------------------------------------------------
    # Tab 4: Statistical analysis
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader("Correlation Matrix")
        show_image_if_exists(EDA_CHARTS["correlation_matrix"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Revenue Distribution")
            show_image_if_exists(EDA_CHARTS["revenue_distribution"])

        with col2:
            st.subheader("Sales Distribution")
            show_image_if_exists(EDA_CHARTS["sales_distribution"])

        st.subheader("Price vs Demand")
        show_image_if_exists(EDA_CHARTS["price_vs_demand"])

    # --------------------------------------------------------
    # Tab 5: Inventory health & risk (from EDA phase)
    # --------------------------------------------------------

    with tabs[4]:

        st.subheader("Inventory Health (EDA snapshot)")
        show_image_if_exists(EDA_CHARTS["inventory_health"])
        if eda_inventory_status_df is not None:
            st.dataframe(eda_inventory_status_df, use_container_width=True)

        if risk_products_df is not None:
            st.subheader("Highest-Risk Products (raw-data view)")
            st.write(f"Flagged products: **{len(risk_products_df):,}**")
            st.dataframe(risk_products_df, use_container_width=True, height=350)

        if sku_inventory_risk_df is not None:
            with st.expander("Full SKU inventory risk table"):
                st.dataframe(sku_inventory_risk_df, use_container_width=True, height=400)


# ============================================================
# [14] FEATURE ENGINEERING
# ============================================================

elif page == "Feature Engineering":

    st.header("🧪 Feature Engineering")

    st.write(
        "Summary of the engineered feature set (Phase 4) used to train "
        "the demand forecasting models."
    )

    if feature_summary_df is not None:

        summary_lookup = feature_summary_df.set_index("metric")["value"].to_dict()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", format_integer(summary_lookup.get("rows")))
        with col2:
            st.metric("Columns", format_integer(summary_lookup.get("columns")))
        with col3:
            st.metric("Missing Cells", format_integer(summary_lookup.get("missing_cells")))
        with col4:
            st.metric("Duplicate Rows", format_integer(summary_lookup.get("duplicate_rows")))

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Feature Dictionary")

        if feature_dictionary_df is not None:
            search_term = st.text_input("Search features", "")

            display_dict = feature_dictionary_df

            if search_term:
                display_dict = display_dict[
                    display_dict["feature"].astype(str).str.contains(search_term, case=False, na=False)
                ]

            st.dataframe(display_dict, use_container_width=True, height=450)
        else:
            st.info("Feature dictionary not available.")

    with col_right:
        st.subheader("Feature Importance")

        if feature_importance_df is not None and "feature" in feature_importance_df.columns:

            top_n = st.slider("Show top N features", 5, min(30, len(feature_importance_df)), 15)

            importance_chart = (
                feature_importance_df
                .sort_values("importance", ascending=False)
                .head(top_n)
                .set_index("feature")
            )

            st.bar_chart(importance_chart["importance"])

            with st.expander("Full feature importance table"):
                st.dataframe(feature_importance_df, use_container_width=True, height=350)
        else:
            st.info("Feature importance data not available.")


# ============================================================
# [15] DEMAND FORECASTING
# ============================================================

elif page == "Demand Forecasting":

    st.header("📈 Demand Forecasting")

    if forecast_df is None:
        st.error("Demand forecast dataset is unavailable.")
        st.stop()

    if forecast_summary_df is not None:

        summary_lookup = forecast_summary_df.set_index("metric")["value"].to_dict()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Training Rows", format_integer(summary_lookup.get("training_rows")))
        with col2:
            st.metric("Validation Rows", format_integer(summary_lookup.get("validation_rows")))
        with col3:
            st.metric("Test Rows", format_integer(summary_lookup.get("test_rows")))

    st.write(f"Forecast records: **{len(forecast_df):,}**")

    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    if "date" in forecast_df.columns:

        min_date = forecast_df["date"].min()
        max_date = forecast_df["date"].max()

        selected_dates = st.date_input(
            "Select forecast period",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

            start_date = pd.Timestamp(selected_dates[0])
            end_date = pd.Timestamp(selected_dates[1])

            filtered_forecast = forecast_df[
                (forecast_df["date"] >= start_date) & (forecast_df["date"] <= end_date)
            ].copy()

        else:
            filtered_forecast = forecast_df.copy()

    else:
        filtered_forecast = forecast_df.copy()

    # --------------------------------------------------------
    # Actual vs predicted
    # --------------------------------------------------------

    if actual_column and prediction_column:

        daily_forecast = (
            filtered_forecast
            .groupby("date")[[actual_column, prediction_column]]
            .sum()
            .reset_index()
            .set_index("date")
        )

        st.subheader("Actual vs Forecast Demand")
        st.line_chart(daily_forecast)

    # --------------------------------------------------------
    # Forecast error distribution
    # --------------------------------------------------------

    if "forecast_error" in filtered_forecast.columns:

        st.subheader("Forecast Error Distribution")

        error_hist = (
            filtered_forecast["forecast_error"]
            .round(0)
            .value_counts()
            .sort_index()
        )

        st.bar_chart(error_hist)

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    if model_comparison_df is not None:

        st.subheader("Model Comparison")
        st.dataframe(model_comparison_df, use_container_width=True)

        numeric_columns = [
            column for column in ["MAE", "RMSE", "sMAPE_percent", "WAPE_percent"]
            if column in model_comparison_df.columns
        ]

        if len(numeric_columns) > 0 and "model" in model_comparison_df.columns:

            selected_metric = st.selectbox("Comparison Metric", numeric_columns)

            comparison_chart = model_comparison_df[["model", selected_metric]].set_index("model")

            st.bar_chart(comparison_chart)

    # --------------------------------------------------------
    # Forecast table
    # --------------------------------------------------------

    st.subheader("Forecast Records")

    display_columns = [
        column for column in [
            "date", "sku_id", "store_id", "actual_units_sold",
            "predicted_units_sold", "forecast_error", "absolute_error"
        ]
        if column in filtered_forecast.columns
    ]

    st.dataframe(filtered_forecast[display_columns], use_container_width=True, height=400)

    csv_data = filtered_forecast.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Forecast Data",
        data=csv_data,
        file_name="demand_forecasts_filtered.csv",
        mime="text/csv"
    )


# ============================================================
# [16] INVENTORY OPTIMIZATION
# ============================================================

elif page == "Inventory Optimization":

    st.header("📦 Inventory Optimization")

    st.write(f"SKU-store combinations: **{len(inventory_df):,}**")

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_stock = safe_sum(inventory_df, "stock_on_hand")
    total_demand = safe_sum(inventory_df, "forecast_total_demand")
    total_order = safe_sum(inventory_df, "recommended_order_quantity")
    inventory_cost = safe_sum(inventory_df, "inventory_cost_value")
    replenishment_cost = safe_sum(inventory_df, "estimated_replenishment_cost")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Stock On Hand", format_integer(total_stock))
    with col2:
        st.metric("Forecast Demand", format_number(total_demand))
    with col3:
        st.metric("Recommended Order", format_integer(total_order))
    with col4:
        st.metric("Inventory Cost", format_number(inventory_cost))
    with col5:
        st.metric("Replenishment Cost", format_number(replenishment_cost))

    st.markdown("---")

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    filtered_inventory = inventory_df.copy()

    if "inventory_status" in inventory_df.columns:

        statuses = sorted(inventory_df["inventory_status"].dropna().astype(str).unique().tolist())

        selected_status = st.multiselect("Inventory Status", statuses, default=statuses)

        filtered_inventory = filtered_inventory[
            filtered_inventory["inventory_status"].astype(str).isin(selected_status)
        ]

    if "replenishment_priority" in filtered_inventory.columns:

        priorities = sorted(filtered_inventory["replenishment_priority"].dropna().astype(str).unique().tolist())

        selected_priority = st.multiselect("Replenishment Priority", priorities, default=priorities)

        filtered_inventory = filtered_inventory[
            filtered_inventory["replenishment_priority"].astype(str).isin(selected_priority)
        ]

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:
        if "inventory_status" in filtered_inventory.columns:
            st.subheader("Inventory Status Distribution")
            status_counts = (
                filtered_inventory["inventory_status"]
                .value_counts()
                .rename_axis("status")
                .reset_index(name="count")
            )
            st.bar_chart(status_counts.set_index("status"))

    with col_b:
        if "category" in filtered_inventory.columns and "recommended_order_quantity" in filtered_inventory.columns:
            st.subheader("Recommended Order by Category")
            category_orders = (
                filtered_inventory
                .groupby("category")["recommended_order_quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            st.bar_chart(category_orders)

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    st.subheader("Inventory Recommendations")

    st.dataframe(filtered_inventory, use_container_width=True, height=500)

    csv_data = filtered_inventory.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Inventory Recommendations",
        data=csv_data,
        file_name="inventory_recommendations_filtered.csv",
        mime="text/csv"
    )


# ============================================================
# [17] MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("🎯 Model Evaluation")

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    if model_performance_df is not None:
        st.subheader("Overall Model Performance")
        st.dataframe(model_performance_df, use_container_width=True)

    if overall_forecast_df is not None:
        st.subheader("Overall Forecast Evaluation")
        st.dataframe(overall_forecast_df, use_container_width=True)

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    if model_comparison_df is not None:

        st.subheader("Model Comparison")
        st.dataframe(model_comparison_df, use_container_width=True)

        numeric_columns = [
            column for column in ["MAE", "RMSE", "sMAPE_percent", "WAPE_percent"]
            if column in model_comparison_df.columns
        ]

        if len(numeric_columns) > 0 and "model" in model_comparison_df.columns:

            selected_metric = st.selectbox("Comparison Metric", numeric_columns)

            comparison_chart = model_comparison_df[["model", selected_metric]].set_index("model")

            st.bar_chart(comparison_chart)

    # --------------------------------------------------------
    # Forecast error summary
    # --------------------------------------------------------

    if error_summary_df is not None:

        st.subheader("Forecast Error Category Breakdown")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(error_summary_df, use_container_width=True)

        with col2:
            st.bar_chart(error_summary_df.set_index("error_category")["record_count"])

    # --------------------------------------------------------
    # SKU / store performance
    # --------------------------------------------------------

    if sku_evaluation_df is not None:

        st.subheader("SKU-Level Evaluation")
        st.write(f"Evaluated SKUs: **{len(sku_evaluation_df):,}**")
        st.dataframe(sku_evaluation_df, use_container_width=True, height=400)

    if store_evaluation_df is not None:

        st.subheader("Store-Level Evaluation")
        st.write(f"Evaluated stores: **{len(store_evaluation_df):,}**")
        st.dataframe(store_evaluation_df, use_container_width=True, height=400)

        if "WAPE_percent" in store_evaluation_df.columns and "store_id" in store_evaluation_df.columns:
            st.subheader("Store WAPE Comparison")
            st.bar_chart(store_evaluation_df.set_index("store_id")["WAPE_percent"])

    # --------------------------------------------------------
    # Inventory & replenishment evaluation
    # --------------------------------------------------------

    col_x, col_y = st.columns(2)

    with col_x:
        if inventory_evaluation_df is not None:
            st.subheader("Inventory Evaluation")
            st.dataframe(inventory_evaluation_df, use_container_width=True)

    with col_y:
        if replenishment_df is not None:
            st.subheader("Replenishment Evaluation")
            st.dataframe(replenishment_df, use_container_width=True)

    if inventory_status_df is not None:
        st.subheader("Inventory Status Evaluation")
        st.dataframe(inventory_status_df, use_container_width=True)

        if "inventory_status" in inventory_status_df.columns and "count" in inventory_status_df.columns:
            st.bar_chart(inventory_status_df.set_index("inventory_status")["count"])


# ============================================================
# [18] RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.header("⚠️ Inventory Risk Analysis")

    # --------------------------------------------------------
    # High-risk inventory
    # --------------------------------------------------------

    if high_risk_df is not None:

        st.subheader("High-Risk Inventory")

        st.error(f"{len(high_risk_df):,} SKU-store combinations are classified as high risk.")

        filtered_risk = high_risk_df.copy()

        if "replenishment_priority" in filtered_risk.columns:

            priorities = sorted(filtered_risk["replenishment_priority"].dropna().astype(str).unique().tolist())

            selected_priority = st.multiselect("Risk Priority", priorities, default=priorities)

            filtered_risk = filtered_risk[
                filtered_risk["replenishment_priority"].astype(str).isin(selected_priority)
            ]

        st.dataframe(filtered_risk, use_container_width=True, height=500)

        csv_data = filtered_risk.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download High-Risk Inventory",
            data=csv_data,
            file_name="high_risk_inventory_filtered.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------
    # Full inventory risk analysis (Phase 6)
    # --------------------------------------------------------

    if inventory_risk_df is not None:

        st.subheader("Full Inventory Risk Analysis")

        st.dataframe(inventory_risk_df, use_container_width=True, height=400)

    # --------------------------------------------------------
    # Highest-risk raw products (EDA)
    # --------------------------------------------------------

    if risk_products_df is not None:

        st.subheader("Highest-Risk Products (raw-data EDA view)")
        st.dataframe(risk_products_df, use_container_width=True, height=350)

    # --------------------------------------------------------
    # Error summary / analysis
    # --------------------------------------------------------

    if error_summary_df is not None:
        st.subheader("Forecast Error Summary")
        st.dataframe(error_summary_df, use_container_width=True)

    if error_analysis_df is not None:
        st.subheader("Forecast Error Analysis")
        st.dataframe(error_analysis_df, use_container_width=True, height=400)


# ============================================================
# [19] SKU / STORE EXPLORER
# ============================================================

elif page == "SKU / Store Explorer":

    st.header("🔎 SKU / Store Explorer")

    if forecast_df is None:
        st.error("Forecast data unavailable.")
        st.stop()

    explorer_df = forecast_df.copy()

    # --------------------------------------------------------
    # SKU selector
    # --------------------------------------------------------

    selected_sku = None

    if sku_column:

        sku_values = sorted(explorer_df[sku_column].dropna().astype(str).unique().tolist())

        selected_sku = st.selectbox("Select SKU", ["All"] + sku_values)

        if selected_sku != "All":
            explorer_df = explorer_df[explorer_df[sku_column].astype(str) == selected_sku]

    # --------------------------------------------------------
    # Store selector
    # --------------------------------------------------------

    selected_store = None

    if store_column:

        store_values = sorted(explorer_df[store_column].dropna().astype(str).unique().tolist())

        selected_store = st.selectbox("Select Store", ["All"] + store_values)

        if selected_store != "All":
            explorer_df = explorer_df[explorer_df[store_column].astype(str) == selected_store]

    # --------------------------------------------------------
    # Selected data
    # --------------------------------------------------------

    st.subheader("Demand Forecast")

    if actual_column and prediction_column and "date" in explorer_df.columns:

        chart_data = (
            explorer_df
            .groupby("date")[[actual_column, prediction_column]]
            .sum()
        )

        st.line_chart(chart_data)

    st.dataframe(explorer_df, use_container_width=True, height=300)

    # --------------------------------------------------------
    # Matching inventory
    # --------------------------------------------------------

    if inventory_df is not None and selected_sku is not None and selected_sku != "All":

        matching_inventory = inventory_df.copy()

        inventory_sku_column = find_column(matching_inventory, ["sku_id", "SKU_ID", "sku"])

        if inventory_sku_column:

            matching_inventory = matching_inventory[
                matching_inventory[inventory_sku_column].astype(str) == str(selected_sku)
            ]

            if selected_store is not None and selected_store != "All":

                inventory_store_column = find_column(matching_inventory, ["store_id", "STORE_ID", "store"])

                if inventory_store_column:
                    matching_inventory = matching_inventory[
                        matching_inventory[inventory_store_column].astype(str) == str(selected_store)
                    ]

            st.subheader("Inventory Recommendation")

            if len(matching_inventory) > 0:
                st.dataframe(matching_inventory, use_container_width=True)
            else:
                st.info("No inventory recommendation found for the selected combination.")

    # --------------------------------------------------------
    # Matching SKU / store evaluation
    # --------------------------------------------------------

    if sku_evaluation_df is not None and selected_sku is not None and selected_sku != "All":

        matching_eval = sku_evaluation_df[sku_evaluation_df["sku_id"].astype(str) == str(selected_sku)]

        if len(matching_eval) > 0:
            st.subheader("SKU Forecast Accuracy")
            st.dataframe(matching_eval, use_container_width=True)

    if store_evaluation_df is not None and selected_store is not None and selected_store != "All":

        matching_store_eval = store_evaluation_df[
            store_evaluation_df["store_id"].astype(str) == str(selected_store)
        ]

        if len(matching_store_eval) > 0:
            st.subheader("Store Forecast Accuracy")
            st.dataframe(matching_store_eval, use_container_width=True)


# ============================================================
# [20] INSIGHTS & SUGGESTIONS
# ============================================================

elif page == "Insights & Suggestions":

    st.header("💡 Insights & Suggestions")

    st.write(
        "A live, data-driven read of the current pipeline output — "
        "what's happening right now, and what to do about it."
    )

    # --------------------------------------------------------
    # Current status
    # --------------------------------------------------------

    st.subheader("🩺 Current Status")

    for level, message in status_items:
        render_status(level, message)

    st.markdown("---")

    # --------------------------------------------------------
    # Suggestions
    # --------------------------------------------------------

    st.subheader("📌 Suggested Actions")

    if len(suggestions) == 0:
        st.info("No suggestions to surface right now — everything looks nominal.")
    else:
        for level, message in suggestions:
            render_status(level, message)

    st.markdown("---")

    # --------------------------------------------------------
    # Supporting tables
    # --------------------------------------------------------

    st.subheader("📋 Supporting Detail")

    col1, col2 = st.columns(2)

    with col1:

        if inventory_df is not None and "inventory_status" in inventory_df.columns:

            st.markdown("**Top items needing urgent restock**")

            urgent = inventory_df[
                inventory_df["inventory_status"].astype(str).str.upper().isin(["STOCKOUT", "CRITICAL"])
            ]

            if "estimated_replenishment_cost" in urgent.columns:
                urgent = urgent.sort_values("estimated_replenishment_cost", ascending=False)

            display_cols = [
                c for c in [
                    "sku_id", "store_id", "sku_name", "inventory_status",
                    "stock_on_hand", "recommended_order_quantity", "estimated_replenishment_cost"
                ]
                if c in urgent.columns
            ]

            st.dataframe(urgent[display_cols].head(20), use_container_width=True, height=350)

    with col2:

        if sku_evaluation_df is not None and "WAPE_percent" in sku_evaluation_df.columns:

            st.markdown("**SKUs with the highest forecast error**")

            worst = sku_evaluation_df.sort_values("WAPE_percent", ascending=False).head(20)

            st.dataframe(worst, use_container_width=True, height=350)

    if inventory_df is not None and "inventory_status" in inventory_df.columns:

        st.markdown("**Overstocked items (capital tied up)**")

        overstock = inventory_df[inventory_df["inventory_status"].astype(str).str.upper() == "OVERSTOCK"]

        if "inventory_cost_value" in overstock.columns:
            overstock = overstock.sort_values("inventory_cost_value", ascending=False)

        display_cols = [
            c for c in [
                "sku_id", "store_id", "sku_name", "category",
                "stock_on_hand", "inventory_cost_value", "days_of_forecast_coverage"
            ]
            if c in overstock.columns
        ]

        st.dataframe(overstock[display_cols].head(20), use_container_width=True, height=350)


# ============================================================
# [21] FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Project Foresight | AI-Driven Demand Forecasting "
    "and Inventory Optimization | Phase 8"
)
