# ============================================================
# PROJECT FORESIGHT - PHASE 6: INVENTORY OPTIMIZATION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# [1] PATHS
# ============================================================

print("\n" + "=" * 80)
print("PROJECT FORESIGHT - PHASE 6: INVENTORY OPTIMIZATION")
print("=" * 80)

# Phase 6 script is inside:
# Project_Forsight/ml_part/
#
# Therefore:
# ".." moves from ml_part -> Project_Forsight

BASE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

FORECAST_PATH = os.path.join(
    DATA_PATH,
    "forecasting"
)

PROCESSED_PATH = os.path.join(
    DATA_PATH,
    "processed"
)

INVENTORY_OUTPUT_PATH = os.path.join(
    DATA_PATH,
    "inventory_optimization"
)

REPORT_PATH = os.path.join(
    BASE_PATH,
    "reports",
    "inventory_optimization"
)

os.makedirs(
    INVENTORY_OUTPUT_PATH,
    exist_ok=True
)

os.makedirs(
    REPORT_PATH,
    exist_ok=True
)

print("\n[1] PATHS")
print("-" * 80)

print(
    f"Project root       : {BASE_PATH}"
)

print(
    f"Forecast input     : {FORECAST_PATH}"
)

print(
    f"Processed input    : {PROCESSED_PATH}"
)

print(
    f"Inventory output   : {INVENTORY_OUTPUT_PATH}"
)

print(
    f"Reports            : {REPORT_PATH}"
)


# ============================================================
# [2] CHECKING INPUT FILES
# ============================================================

print("\n[2] CHECKING INPUT FILES")
print("-" * 80)

forecast_file = os.path.join(
    FORECAST_PATH,
    "demand_forecasts.csv"
)

inventory_file = os.path.join(
    PROCESSED_PATH,
    "inventory_snapshot_clean.csv"
)

sku_file = os.path.join(
    PROCESSED_PATH,
    "sku_master_clean.csv"
)

required_files = {
    "demand_forecasts.csv": forecast_file,
    "inventory_snapshot_clean.csv": inventory_file,
    "sku_master_clean.csv": sku_file
}

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{file_name} not found:\n{file_path}"
        )

    print(
        f"✓ {file_name} found"
    )


# ============================================================
# [3] LOADING INPUT DATA
# ============================================================

print("\n[3] LOADING INPUT DATA")
print("-" * 80)

forecast_df = pd.read_csv(
    forecast_file
)

inventory_df = pd.read_csv(
    inventory_file
)

sku_df = pd.read_csv(
    sku_file
)

print("✓ Demand forecasts loaded")
print("✓ Inventory snapshot loaded")
print("✓ SKU master loaded")

print()

print(
    f"Forecast rows       : {len(forecast_df):,}"
)

print(
    f"Inventory rows      : {len(inventory_df):,}"
)

print(
    f"SKU master rows     : {len(sku_df):,}"
)


# ============================================================
# [4] BASIC VALIDATION
# ============================================================

print("\n[4] BASIC DATA VALIDATION")
print("-" * 80)

forecast_required = [
    "date",
    "sku_id",
    "store_id",
    "actual_units_sold",
    "predicted_units_sold"
]

inventory_required = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "last_restock_date"
]

sku_required = [
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "unit_price",
    "cost_price",
    "brand"
]

missing_forecast = [
    column
    for column in forecast_required
    if column not in forecast_df.columns
]

missing_inventory = [
    column
    for column in inventory_required
    if column not in inventory_df.columns
]

missing_sku = [
    column
    for column in sku_required
    if column not in sku_df.columns
]

if missing_forecast:

    raise ValueError(
        "Forecast columns missing: "
        + ", ".join(missing_forecast)
    )

if missing_inventory:

    raise ValueError(
        "Inventory columns missing: "
        + ", ".join(missing_inventory)
    )

if missing_sku:

    raise ValueError(
        "SKU columns missing: "
        + ", ".join(missing_sku)
    )

print("✓ Forecast columns validated")
print("✓ Inventory columns validated")
print("✓ SKU master columns validated")


# ============================================================
# [5] DATE CONVERSION
# ============================================================

print("\n[5] CONVERTING DATE COLUMNS")
print("-" * 80)

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"],
    errors="coerce"
)

inventory_df["last_restock_date"] = pd.to_datetime(
    inventory_df["last_restock_date"],
    errors="coerce"
)

if forecast_df["date"].isna().any():

    raise ValueError(
        "Invalid dates found in demand forecasts."
    )

if inventory_df["last_restock_date"].isna().any():

    print(
        "WARNING: Missing last_restock_date values found."
    )

print("✓ Forecast dates converted")
print("✓ Inventory dates converted")


# ============================================================
# [6] NUMERIC DATA VALIDATION
# ============================================================

print("\n[6] VALIDATING NUMERIC DATA")
print("-" * 80)

numeric_inventory_columns = [
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

numeric_sku_columns = [
    "unit_price",
    "cost_price"
]

numeric_forecast_columns = [
    "actual_units_sold",
    "predicted_units_sold"
]

for column in (
    numeric_inventory_columns
    + numeric_sku_columns
    + numeric_forecast_columns
):

    if column in inventory_df.columns:

        inventory_df[column] = pd.to_numeric(
            inventory_df[column],
            errors="coerce"
        )

    if column in sku_df.columns:

        sku_df[column] = pd.to_numeric(
            sku_df[column],
            errors="coerce"
        )

    if column in forecast_df.columns:

        forecast_df[column] = pd.to_numeric(
            forecast_df[column],
            errors="coerce"
        )

print("✓ Numeric columns converted")


# ============================================================
# [7] FORECAST VALIDATION
# ============================================================

print("\n[7] VALIDATING DEMAND FORECASTS")
print("-" * 80)

if forecast_df["predicted_units_sold"].isna().any():

    raise ValueError(
        "Predicted demand contains missing values."
    )

negative_forecasts = (
    forecast_df["predicted_units_sold"] < 0
).sum()

if negative_forecasts > 0:

    print(
        f"WARNING: {negative_forecasts:,} negative forecasts found."
    )

    forecast_df["predicted_units_sold"] = np.maximum(
        forecast_df["predicted_units_sold"],
        0
    )

print(
    f"Forecast rows       : {len(forecast_df):,}"
)

print(
    f"Forecast start      : "
    f"{forecast_df['date'].min().date()}"
)

print(
    f"Forecast end        : "
    f"{forecast_df['date'].max().date()}"
)

print("✓ Forecast data validated")


# ============================================================
# [8] BUILDING FORECAST DEMAND SUMMARY
# ============================================================

print("\n[8] BUILDING FORECAST DEMAND SUMMARY")
print("-" * 80)

forecast_summary = (
    forecast_df
    .groupby(
        [
            "store_id",
            "sku_id"
        ],
        as_index=False
    )
    .agg(
        forecast_total_demand=(
            "predicted_units_sold",
            "sum"
        ),

        forecast_average_daily_demand=(
            "predicted_units_sold",
            "mean"
        ),

        forecast_demand_std=(
            "predicted_units_sold",
            "std"
        ),

        forecast_max_daily_demand=(
            "predicted_units_sold",
            "max"
        ),

        forecast_days=(
            "date",
            "nunique"
        )
    )
)

forecast_summary["forecast_demand_std"] = (
    forecast_summary[
        "forecast_demand_std"
    ].fillna(0)
)

print(
    f"SKU-store combinations : "
    f"{len(forecast_summary):,}"
)

print("✓ Forecast demand summary created")


# ============================================================
# [9] MERGING INVENTORY DATA
# ============================================================

print("\n[9] MERGING INVENTORY DATA")
print("-" * 80)

inventory_columns = [
    "store_id",
    "sku_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock",
    "last_restock_date"
]

inventory_subset = inventory_df[
    inventory_columns
].copy()

inventory_subset = inventory_subset.drop_duplicates(
    subset=[
        "store_id",
        "sku_id"
    ],
    keep="last"
)

optimization_df = forecast_summary.merge(
    inventory_subset,
    on=[
        "store_id",
        "sku_id"
    ],
    how="left",
    validate="one_to_one"
)

print("✓ Inventory data merged")


# ============================================================
# [10] MERGING SKU MASTER
# ============================================================

print("\n[10] MERGING SKU MASTER")
print("-" * 80)

sku_columns = [
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "unit_price",
    "cost_price",
    "brand"
]

sku_subset = sku_df[
    sku_columns
].copy()

sku_subset = sku_subset.drop_duplicates(
    subset="sku_id",
    keep="last"
)

optimization_df = optimization_df.merge(
    sku_subset,
    on="sku_id",
    how="left",
    validate="many_to_one"
)

print("✓ SKU master merged")


# ============================================================
# [11] MERGE VALIDATION
# ============================================================

print("\n[11] MERGE VALIDATION")
print("-" * 80)

inventory_missing = (
    optimization_df["stock_on_hand"]
    .isna()
    .sum()
)

sku_missing = (
    optimization_df["sku_name"]
    .isna()
    .sum()
)

print(
    f"Missing inventory matches : "
    f"{inventory_missing:,}"
)

print(
    f"Missing SKU matches       : "
    f"{sku_missing:,}"
)

if inventory_missing > 0:

    print(
        "WARNING: Some forecast SKU-store combinations "
        "do not have inventory records."
    )

if sku_missing > 0:

    print(
        "WARNING: Some forecast SKUs do not have SKU master records."
    )


# ============================================================
# [12] HANDLING INVENTORY VALUES
# ============================================================

print("\n[12] HANDLING INVENTORY VALUES")
print("-" * 80)

inventory_numeric = [
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

for column in inventory_numeric:

    optimization_df[column] = (
        pd.to_numeric(
            optimization_df[column],
            errors="coerce"
        )
        .fillna(0)
    )

print("✓ Inventory numeric values handled")


# ============================================================
# [13] HANDLING SKU VALUES
# ============================================================

print("\n[13] HANDLING SKU MASTER VALUES")
print("-" * 80)

optimization_df["unit_price"] = (
    pd.to_numeric(
        optimization_df["unit_price"],
        errors="coerce"
    )
    .fillna(0)
)

optimization_df["cost_price"] = (
    pd.to_numeric(
        optimization_df["cost_price"],
        errors="coerce"
    )
    .fillna(0)
)

categorical_columns = [
    "sku_name",
    "category",
    "subcategory",
    "brand"
]

for column in categorical_columns:

    optimization_df[column] = (
        optimization_df[column]
        .fillna("Unknown")
    )

print("✓ SKU master values handled")


# ============================================================
# [14] CREATING INVENTORY POSITION FEATURES
# ============================================================

print("\n[14] CREATING INVENTORY POSITION FEATURES")
print("-" * 80)

optimization_df["stock_above_reorder"] = (
    optimization_df["stock_on_hand"]
    -
    optimization_df["reorder_point"]
)

optimization_df["stock_above_safety"] = (
    optimization_df["stock_on_hand"]
    -
    optimization_df["safety_stock"]
)

optimization_df["reorder_gap"] = (
    optimization_df["reorder_point"]
    -
    optimization_df["stock_on_hand"]
)

optimization_df["safety_stock_gap"] = (
    optimization_df["safety_stock"]
    -
    optimization_df["stock_on_hand"]
)

print("✓ stock_above_reorder")
print("✓ stock_above_safety")
print("✓ reorder_gap")
print("✓ safety_stock_gap")


# ============================================================
# [15] DEMAND-TO-STOCK FEATURES
# ============================================================

print("\n[15] CREATING DEMAND-TO-STOCK FEATURES")
print("-" * 80)

optimization_df["days_of_forecast_coverage"] = np.where(
    optimization_df["forecast_average_daily_demand"] > 0,

    optimization_df["stock_on_hand"]
    /
    optimization_df["forecast_average_daily_demand"],

    np.inf
)

optimization_df["stock_to_forecast_ratio"] = np.where(
    optimization_df["forecast_total_demand"] > 0,

    optimization_df["stock_on_hand"]
    /
    optimization_df["forecast_total_demand"],

    np.inf
)

optimization_df["forecast_minus_stock"] = (
    optimization_df["forecast_total_demand"]
    -
    optimization_df["stock_on_hand"]
)

print("✓ days_of_forecast_coverage")
print("✓ stock_to_forecast_ratio")
print("✓ forecast_minus_stock")


# ============================================================
# [16] FORECAST DEMAND RISK
# ============================================================

print("\n[16] CREATING FORECAST DEMAND RISK FEATURES")
print("-" * 80)

optimization_df["demand_variability_ratio"] = np.where(
    optimization_df["forecast_average_daily_demand"] > 0,

    optimization_df["forecast_demand_std"]
    /
    optimization_df["forecast_average_daily_demand"],

    0
)

optimization_df["demand_volatility_flag"] = np.where(
    optimization_df["demand_variability_ratio"] >= 0.50,
    1,
    0
)

print("✓ demand_variability_ratio")
print("✓ demand_volatility_flag")


# ============================================================
# [17] STOCKOUT RISK
# ============================================================

print("\n[17] CALCULATING STOCKOUT RISK")
print("-" * 80)

optimization_df["stockout_risk_flag"] = np.where(
    (
        optimization_df["stock_on_hand"] <= 0
    )
    |
    (
        optimization_df["stock_on_hand"]
        <
        optimization_df["safety_stock"]
    )
    |
    (
        optimization_df["stock_on_hand"]
        <
        optimization_df["reorder_point"]
    ),

    1,

    0
)

print("✓ stockout_risk_flag")


# ============================================================
# [18] OVERSTOCK RISK
# ============================================================

print("\n[18] CALCULATING OVERSTOCK RISK")
print("-" * 80)

optimization_df["overstock_risk_flag"] = np.where(
    (
        optimization_df["stock_on_hand"]
        >
        optimization_df["reorder_point"]
    )
    &
    (
        optimization_df["stock_on_hand"]
        >
        optimization_df["forecast_total_demand"]
    ),

    1,

    0
)

print("✓ overstock_risk_flag")


# ============================================================
# [19] RECOMMENDED REPLENISHMENT QUANTITY
# ============================================================

print("\n[19] CALCULATING RECOMMENDED REPLENISHMENT")
print("-" * 80)

# The recommendation brings stock back toward the
# existing reorder point while also considering
# the forecasted demand over the available forecast horizon.

target_inventory = np.maximum(
    optimization_df["reorder_point"],
    optimization_df["safety_stock"]
)

forecast_gap = (
    optimization_df["forecast_total_demand"]
    -
    optimization_df["stock_on_hand"]
)

replenishment_need = np.maximum(
    optimization_df["reorder_point"]
    -
    optimization_df["stock_on_hand"],
    0
)

optimization_df["recommended_order_quantity"] = np.maximum(
    replenishment_need,
    np.maximum(
        forecast_gap,
        0
    )
)

# Round to whole units because inventory quantities
# represent physical units.

optimization_df["recommended_order_quantity"] = np.ceil(
    optimization_df[
        "recommended_order_quantity"
    ]
).astype(int)

print("✓ Recommended order quantity calculated")


# ============================================================
# [20] REPLENISHMENT COST
# ============================================================

print("\n[20] CALCULATING REPLENISHMENT COST")
print("-" * 80)

optimization_df["estimated_replenishment_cost"] = (
    optimization_df[
        "recommended_order_quantity"
    ]
    *
    optimization_df[
        "cost_price"
    ]
)

optimization_df["estimated_replenishment_retail_value"] = (
    optimization_df[
        "recommended_order_quantity"
    ]
    *
    optimization_df[
        "unit_price"
    ]
)

print("✓ Estimated replenishment cost calculated")
print("✓ Estimated replenishment retail value calculated")


# ============================================================
# [21] CURRENT INVENTORY VALUE
# ============================================================

print("\n[21] CALCULATING CURRENT INVENTORY VALUE")
print("-" * 80)

optimization_df["inventory_cost_value"] = (
    optimization_df["stock_on_hand"]
    *
    optimization_df["cost_price"]
)

optimization_df["inventory_retail_value"] = (
    optimization_df["stock_on_hand"]
    *
    optimization_df["unit_price"]
)

print("✓ Inventory cost value calculated")
print("✓ Inventory retail value calculated")


# ============================================================
# [22] INVENTORY STATUS
# ============================================================

print("\n[22] ASSIGNING INVENTORY STATUS")
print("-" * 80)


def determine_inventory_status(row):

    stock = row["stock_on_hand"]
    reorder = row["reorder_point"]
    safety = row["safety_stock"]
    forecast = row["forecast_total_demand"]

    if stock <= 0:

        return "STOCKOUT"

    if stock < safety:

        return "CRITICAL"

    if stock < reorder:

        return "REORDER"

    if stock < forecast:

        return "AT_RISK"

    if stock > reorder and stock > forecast:

        return "OVERSTOCK"

    return "HEALTHY"


optimization_df["inventory_status"] = (
    optimization_df.apply(
        determine_inventory_status,
        axis=1
    )
)

print("✓ Inventory status assigned")


# ============================================================
# [23] REPLENISHMENT PRIORITY
# ============================================================

print("\n[23] ASSIGNING REPLENISHMENT PRIORITY")
print("-" * 80)


def determine_priority(row):

    status = row["inventory_status"]
    order_quantity = row[
        "recommended_order_quantity"
    ]

    if status == "STOCKOUT":

        return "URGENT"

    if status == "CRITICAL":

        return "HIGH"

    if status == "REORDER":

        return "HIGH"

    if status == "AT_RISK":

        return "MEDIUM"

    if (
        status == "HEALTHY"
        and order_quantity > 0
    ):

        return "LOW"

    return "NONE"


optimization_df["replenishment_priority"] = (
    optimization_df.apply(
        determine_priority,
        axis=1
    )
)

print("✓ Replenishment priority assigned")


# ============================================================
# [24] FINAL FEATURE CLEANUP
# ============================================================

print("\n[24] FINAL FEATURE CLEANUP")
print("-" * 80)

numeric_columns = (
    optimization_df
    .select_dtypes(
        include=[np.number]
    )
    .columns
)

optimization_df[numeric_columns] = (
    optimization_df[numeric_columns]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

optimization_df[numeric_columns] = (
    optimization_df[numeric_columns]
    .fillna(0)
)

print("✓ Infinite values handled")
print("✓ Numeric missing values handled")


# ============================================================
# [25] SORTING FINAL DATASET
# ============================================================

print("\n[25] SORTING FINAL DATASET")
print("-" * 80)

optimization_df = (
    optimization_df
    .sort_values(
        [
            "replenishment_priority",
            "store_id",
            "sku_id"
        ]
    )
    .reset_index(drop=True)
)

print("✓ Dataset sorted")


# ============================================================
# [26] FINAL INVENTORY RECOMMENDATIONS
# ============================================================

print("\n[26] CREATING INVENTORY RECOMMENDATIONS")
print("-" * 80)

recommendation_columns = [
    "store_id",
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "brand",

    "forecast_total_demand",
    "forecast_average_daily_demand",
    "forecast_demand_std",
    "forecast_days",

    "stock_on_hand",
    "reorder_point",
    "safety_stock",

    "days_of_forecast_coverage",
    "stock_above_reorder",
    "stock_above_safety",

    "stockout_risk_flag",
    "overstock_risk_flag",
    "demand_volatility_flag",

    "inventory_status",
    "replenishment_priority",

    "recommended_order_quantity",

    "unit_price",
    "cost_price",

    "inventory_cost_value",
    "inventory_retail_value",

    "estimated_replenishment_cost",
    "estimated_replenishment_retail_value",

    "last_restock_date"
]

inventory_recommendations = optimization_df[
    recommendation_columns
].copy()

print(
    f"Recommendation rows : "
    f"{len(inventory_recommendations):,}"
)

print("✓ Inventory recommendations created")


# ============================================================
# [27] INVENTORY RISK ANALYSIS
# ============================================================

print("\n[27] CREATING INVENTORY RISK ANALYSIS")
print("-" * 80)

risk_columns = [
    "store_id",
    "sku_id",
    "sku_name",
    "category",

    "stock_on_hand",
    "reorder_point",
    "safety_stock",

    "forecast_total_demand",
    "forecast_average_daily_demand",
    "forecast_demand_std",

    "days_of_forecast_coverage",
    "demand_variability_ratio",

    "stockout_risk_flag",
    "overstock_risk_flag",
    "demand_volatility_flag",

    "inventory_status",
    "replenishment_priority",

    "recommended_order_quantity"
]

inventory_risk_analysis = optimization_df[
    risk_columns
].copy()

print("✓ Inventory risk analysis created")


# ============================================================
# [28] INVENTORY METRICS
# ============================================================

print("\n[28] CREATING INVENTORY METRICS")
print("-" * 80)

inventory_metrics = pd.DataFrame({
    "metric": [
        "sku_store_combinations",
        "total_stock_on_hand",
        "total_forecast_demand",
        "total_recommended_order_quantity",
        "total_inventory_cost_value",
        "total_inventory_retail_value",
        "total_estimated_replenishment_cost",
        "stockout_risk_count",
        "critical_inventory_count",
        "reorder_inventory_count",
        "at_risk_inventory_count",
        "healthy_inventory_count",
        "overstock_inventory_count",
        "high_priority_replenishment_count",
        "urgent_replenishment_count"
    ],

    "value": [
        len(optimization_df),

        optimization_df[
            "stock_on_hand"
        ].sum(),

        optimization_df[
            "forecast_total_demand"
        ].sum(),

        optimization_df[
            "recommended_order_quantity"
        ].sum(),

        optimization_df[
            "inventory_cost_value"
        ].sum(),

        optimization_df[
            "inventory_retail_value"
        ].sum(),

        optimization_df[
            "estimated_replenishment_cost"
        ].sum(),

        (
            optimization_df[
                "stockout_risk_flag"
            ] == 1
        ).sum(),

        (
            optimization_df[
                "inventory_status"
            ] == "CRITICAL"
        ).sum(),

        (
            optimization_df[
                "inventory_status"
            ] == "REORDER"
        ).sum(),

        (
            optimization_df[
                "inventory_status"
            ] == "AT_RISK"
        ).sum(),

        (
            optimization_df[
                "inventory_status"
            ] == "HEALTHY"
        ).sum(),

        (
            optimization_df[
                "inventory_status"
            ] == "OVERSTOCK"
        ).sum(),

        (
            optimization_df[
                "replenishment_priority"
            ] == "HIGH"
        ).sum(),

        (
            optimization_df[
                "replenishment_priority"
            ] == "URGENT"
        ).sum()
    ]
})

print(
    inventory_metrics.to_string(
        index=False
    )
)

print("✓ Inventory metrics created")


# ============================================================
# [29] INVENTORY SUMMARY
# ============================================================

print("\n[29] CREATING INVENTORY SUMMARY")
print("-" * 80)

status_counts = (
    optimization_df[
        "inventory_status"
    ]
    .value_counts()
)

priority_counts = (
    optimization_df[
        "replenishment_priority"
    ]
    .value_counts()
)

inventory_summary = pd.DataFrame({
    "section": [
        "DATA",
        "DATA",
        "DATA",
        "INVENTORY",
        "INVENTORY",
        "INVENTORY",
        "INVENTORY",
        "INVENTORY",
        "RISK",
        "RISK",
        "RISK",
        "RISK",
        "PRIORITY",
        "PRIORITY",
        "PRIORITY",
        "PRIORITY"
    ],

    "metric": [
        "forecast_start_date",
        "forecast_end_date",
        "sku_store_combinations",

        "total_stock_on_hand",
        "total_forecast_demand",
        "total_recommended_order_quantity",
        "total_inventory_cost_value",
        "total_estimated_replenishment_cost",

        "stockout_count",
        "critical_count",
        "reorder_count",
        "overstock_count",

        "urgent_count",
        "high_count",
        "medium_count",
        "low_count"
    ],

    "value": [
        forecast_df["date"].min().date(),
        forecast_df["date"].max().date(),
        len(optimization_df),

        optimization_df[
            "stock_on_hand"
        ].sum(),

        optimization_df[
            "forecast_total_demand"
        ].sum(),

        optimization_df[
            "recommended_order_quantity"
        ].sum(),

        optimization_df[
            "inventory_cost_value"
        ].sum(),

        optimization_df[
            "estimated_replenishment_cost"
        ].sum(),

        status_counts.get(
            "STOCKOUT",
            0
        ),

        status_counts.get(
            "CRITICAL",
            0
        ),

        status_counts.get(
            "REORDER",
            0
        ),

        status_counts.get(
            "OVERSTOCK",
            0
        ),

        priority_counts.get(
            "URGENT",
            0
        ),

        priority_counts.get(
            "HIGH",
            0
        ),

        priority_counts.get(
            "MEDIUM",
            0
        ),

        priority_counts.get(
            "LOW",
            0
        )
    ]
})

print("✓ Inventory summary created")


# ============================================================
# [30] SAVING INVENTORY RECOMMENDATIONS
# ============================================================

print("\n[30] SAVING INVENTORY RECOMMENDATIONS")
print("-" * 80)

recommendation_file = os.path.join(
    INVENTORY_OUTPUT_PATH,
    "inventory_recommendations.csv"
)

inventory_recommendations.to_csv(
    recommendation_file,
    index=False
)

print(
    "✓ inventory_recommendations.csv"
)


# ============================================================
# [31] SAVING INVENTORY RISK ANALYSIS
# ============================================================

print("\n[31] SAVING INVENTORY RISK ANALYSIS")
print("-" * 80)

risk_file = os.path.join(
    INVENTORY_OUTPUT_PATH,
    "inventory_risk_analysis.csv"
)

inventory_risk_analysis.to_csv(
    risk_file,
    index=False
)

print(
    "✓ inventory_risk_analysis.csv"
)


# ============================================================
# [32] SAVING INVENTORY METRICS
# ============================================================

print("\n[32] SAVING INVENTORY METRICS")
print("-" * 80)

metrics_file = os.path.join(
    INVENTORY_OUTPUT_PATH,
    "inventory_metrics.csv"
)

inventory_metrics.to_csv(
    metrics_file,
    index=False
)

print(
    "✓ inventory_metrics.csv"
)


# ============================================================
# [33] SAVING INVENTORY SUMMARY
# ============================================================

print("\n[33] SAVING INVENTORY SUMMARY")
print("-" * 80)

summary_file = os.path.join(
    INVENTORY_OUTPUT_PATH,
    "inventory_summary.csv"
)

inventory_summary.to_csv(
    summary_file,
    index=False
)

print(
    "✓ inventory_summary.csv"
)


# ============================================================
# [34] FINAL VALIDATION
# ============================================================

print("\n[34] FINAL INVENTORY VALIDATION")
print("-" * 80)

print(
    f"Recommendation rows       : "
    f"{len(inventory_recommendations):,}"
)

print(
    f"Missing SKU IDs            : "
    f"{inventory_recommendations['sku_id'].isna().sum():,}"
)

print(
    f"Missing store IDs          : "
    f"{inventory_recommendations['store_id'].isna().sum():,}"
)

print(
    f"Missing inventory status   : "
    f"{inventory_recommendations['inventory_status'].isna().sum():,}"
)

print(
    f"Missing order quantities   : "
    f"{inventory_recommendations['recommended_order_quantity'].isna().sum():,}"
)

negative_orders = (
    inventory_recommendations[
        "recommended_order_quantity"
    ] < 0
).sum()

print(
    f"Negative order quantities  : "
    f"{negative_orders:,}"
)

duplicate_keys = (
    inventory_recommendations
    .duplicated(
        subset=[
            "store_id",
            "sku_id"
        ]
    )
    .sum()
)

print(
    f"Duplicate SKU-store pairs  : "
    f"{duplicate_keys:,}"
)

if (
    inventory_recommendations[
        "recommended_order_quantity"
    ].isna().sum()
    == 0
):

    print(
        "✓ No missing order quantities"
    )

if negative_orders == 0:

    print(
        "✓ No negative order quantities"
    )

if duplicate_keys == 0:

    print(
        "✓ No duplicate SKU-store pairs"
    )


# ============================================================
# [35] FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("PHASE 6 INVENTORY OPTIMIZATION COMPLETED")
print("=" * 80)

print("\nInventory status summary:")

print(
    optimization_df[
        "inventory_status"
    ]
    .value_counts()
    .to_string()
)

print("\nReplenishment priority summary:")

print(
    optimization_df[
        "replenishment_priority"
    ]
    .value_counts()
    .to_string()
)

print("\nKey results:")

print(
    f"Total stock on hand          : "
    f"{optimization_df['stock_on_hand'].sum():,.0f}"
)

print(
    f"Total forecast demand        : "
    f"{optimization_df['forecast_total_demand'].sum():,.2f}"
)

print(
    f"Recommended order quantity   : "
    f"{optimization_df['recommended_order_quantity'].sum():,.0f}"
)

print(
    f"Stockout-risk combinations   : "
    f"{(optimization_df['stockout_risk_flag'] == 1).sum():,}"
)

print(
    f"Inventory cost value         : "
    f"{optimization_df['inventory_cost_value'].sum():,.2f}"
)

print(
    f"Estimated replenishment cost : "
    f"{optimization_df['estimated_replenishment_cost'].sum():,.2f}"
)

print("\nFiles generated:")

print(
    "✓ inventory_recommendations.csv"
)

print(
    "✓ inventory_risk_analysis.csv"
)

print(
    "✓ inventory_metrics.csv"
)

print(
    "✓ inventory_summary.csv"
)

print("\nInventory optimization output directory:")

print(
    INVENTORY_OUTPUT_PATH
)

print("\nNEXT PHASE:")

print(
    "PHASE 7 → MODEL EVALUATION"
)

print("=" * 80)