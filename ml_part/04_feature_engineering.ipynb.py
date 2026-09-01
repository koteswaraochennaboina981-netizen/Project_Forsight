# ============================================================
# PROJECT FORESIGHT - PHASE 4: FEATURE ENGINEERING
# ============================================================
#
# Purpose:
# Create forecasting-ready features from cleaned datasets.
#
# Input:
# data/processed/
#
# Output:
# data/features/
#
# Target:
# units_sold
# ============================================================


# ============================================================
# [1] IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

print("=" * 80)
print("PROJECT FORESIGHT - PHASE 4: FEATURE ENGINEERING")
print("=" * 80)


# ============================================================
# [2] PATHS
# ============================================================

# FIX (critical reproducibility bug): this was a hardcoded, machine-specific
# Windows path, so the script only ran on one laptop. This file lives one
# directory below the project root (Project_Forsight/ml_part/...), so we
# walk up a single parent to get back to the project root.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

FEATURE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "features"
)

os.makedirs(FEATURE_PATH, exist_ok=True)

print("\n[1] PATHS")
print("-" * 80)
print("Processed data :", PROCESSED_PATH)
print("Feature output :", FEATURE_PATH)


# ============================================================
# [3] CHECKING PROCESSED FILES
# ============================================================

required_files = [
    "sales_daily_clean.csv",
    "sku_master_clean.csv",
    "calendar_clean.csv",
    "inventory_snapshot_clean.csv",
    "store_master_clean.csv",
    "promotions_clean.csv",
    "customer_master_clean.csv",
    "sku_inventory_flags_clean.csv"
]

print("\n[2] CHECKING PROCESSED FILES")
print("-" * 80)

for file in required_files:

    file_path = os.path.join(
        PROCESSED_PATH,
        file
    )

    if os.path.exists(file_path):
        print(f"✓ {file}")
    else:
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# [4] LOADING PROCESSED DATA
# ============================================================

print("\n[3] LOADING PROCESSED DATA")
print("-" * 80)

sales = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "sales_daily_clean.csv"
    )
)

sku_master = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "sku_master_clean.csv"
    )
)

calendar = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "calendar_clean.csv"
    )
)

inventory = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "inventory_snapshot_clean.csv"
    )
)

store_master = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "store_master_clean.csv"
    )
)

promotions = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "promotions_clean.csv"
    )
)

customer_master = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "customer_master_clean.csv"
    )
)

inventory_flags = pd.read_csv(
    os.path.join(
        PROCESSED_PATH,
        "sku_inventory_flags_clean.csv"
    )
)

print("✓ Sales loaded")
print("✓ SKU master loaded")
print("✓ Calendar loaded")
print("✓ Inventory loaded")
print("✓ Store master loaded")
print("✓ Promotions loaded")
print("✓ Customer master loaded")
print("✓ Inventory flags loaded")


# ============================================================
# [5] CONVERT DATE COLUMNS
# ============================================================

print("\n[4] CONVERTING DATE COLUMNS")
print("-" * 80)

sales["date"] = pd.to_datetime(
    sales["date"],
    errors="coerce"
)

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

promotions["start_date"] = pd.to_datetime(
    promotions["start_date"],
    errors="coerce"
)

promotions["end_date"] = pd.to_datetime(
    promotions["end_date"],
    errors="coerce"
)

inventory["last_restock_date"] = pd.to_datetime(
    inventory["last_restock_date"],
    errors="coerce"
)

inventory_flags["window_start"] = pd.to_datetime(
    inventory_flags["window_start"],
    errors="coerce"
)

inventory_flags["window_end"] = pd.to_datetime(
    inventory_flags["window_end"],
    errors="coerce"
)

print("✓ Dates converted")


# ============================================================
# [6] BASIC VALIDATION
# ============================================================

print("\n[5] BASIC DATA VALIDATION")
print("-" * 80)

print(f"Sales rows       : {len(sales):,}")
print(f"SKU master rows  : {len(sku_master):,}")
print(f"Calendar rows    : {len(calendar):,}")
print(f"Inventory rows   : {len(inventory):,}")
print(f"Store master     : {len(store_master):,}")
print(f"Promotions       : {len(promotions):,}")
print(f"Customers        : {len(customer_master):,}")
print(f"Inventory flags  : {len(inventory_flags):,}")

required_sales_columns = [
    "date",
    "sku_id",
    "store_id",
    "units_sold",
    "revenue",
    "unit_price"
]

for column in required_sales_columns:

    if column not in sales.columns:

        raise ValueError(
            f"Missing required sales column: {column}"
        )

if sales["date"].isna().any():

    raise ValueError(
        "Sales dataset contains invalid dates."
    )

print("✓ Required sales columns validated")


# ============================================================
# [7] SORT SALES DATA
# ============================================================

print("\n[6] SORTING SALES DATA")
print("-" * 80)

sales = sales.sort_values(
    [
        "sku_id",
        "store_id",
        "date"
    ]
).reset_index(drop=True)

print("✓ Sales sorted by SKU, store and date")


# ============================================================
# [7.1] BUILD COMPLETE DAILY SKU-STORE FORECASTING DATASET
# ============================================================

print("\n[7] BUILDING COMPLETE DAILY FORECASTING DATASET")
print("-" * 80)

# ------------------------------------------------------------
# IMPORTANT:
# sales_daily_clean.csv contains only dates where a SKU-store
# combination has a recorded sales observation.
#
# For daily demand forecasting, missing dates are treated as
# zero sales.
#
# We create a continuous daily series for every SKU-store
# combination from its first observed date to its last observed
# date.
# ------------------------------------------------------------

sales["date"] = pd.to_datetime(
    sales["date"],
    errors="coerce"
)

sales["units_sold"] = pd.to_numeric(
    sales["units_sold"],
    errors="coerce"
).fillna(0)

sales["revenue"] = pd.to_numeric(
    sales["revenue"],
    errors="coerce"
).fillna(0)

sales["unit_price"] = pd.to_numeric(
    sales["unit_price"],
    errors="coerce"
)

sales["promo_flag"] = pd.to_numeric(
    sales["promo_flag"],
    errors="coerce"
).fillna(0).astype(int)


# ------------------------------------------------------------
# Remove accidental duplicate SKU-store-date records
# ------------------------------------------------------------

sales = (
    sales
    .groupby(
        ["sku_id", "store_id", "date"],
        as_index=False
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        unit_price=("unit_price", "mean"),
        promo_flag=("promo_flag", "max")
    )
)


# ------------------------------------------------------------
# Create continuous daily records
# ------------------------------------------------------------

complete_series = []

for (sku_id, store_id), group in sales.groupby(
    ["sku_id", "store_id"]
):

    group = group.sort_values("date").copy()

    start_date = group["date"].min()
    end_date = group["date"].max()

    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    temp = pd.DataFrame({
        "date": date_range
    })

    temp["sku_id"] = sku_id
    temp["store_id"] = store_id

    temp = temp.merge(
        group,
        on=["sku_id", "store_id", "date"],
        how="left"
    )

    # Missing dates = no recorded sale
    temp["units_sold"] = (
        temp["units_sold"]
        .fillna(0)
    )

    temp["revenue"] = (
        temp["revenue"]
        .fillna(0)
    )

    # Carry last known price forward
    temp["unit_price"] = (
        temp["unit_price"]
        .ffill()
        .bfill()
    )

    # No promotion on missing sales dates initially
    temp["promo_flag"] = (
        temp["promo_flag"]
        .fillna(0)
        .astype(int)
    )

    complete_series.append(temp)


features = pd.concat(
    complete_series,
    ignore_index=True
)


features = features.sort_values(
    ["sku_id", "store_id", "date"]
).reset_index(drop=True)


print(
    f"Original sales rows : {len(sales):,}"
)

print(
    f"Daily forecasting rows : {len(features):,}"
)

print("✓ Complete daily SKU-store series created")


# ============================================================
# [8] CREATING DATE FEATURES
# ============================================================

print("\n[8] CREATING DATE FEATURES")
print("-" * 80)

features["year"] = (
    features["date"].dt.year
)

features["month"] = (
    features["date"].dt.month
)

features["day"] = (
    features["date"].dt.day
)

features["month_num"] = (
    features["date"].dt.month
)

features["day_of_month"] = (
    features["date"].dt.day
)

features["day_of_week"] = (
    features["date"].dt.dayofweek
)

features["week_of_year"] = (
    features["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

features["quarter"] = (
    features["date"].dt.quarter
)

features["is_weekend"] = (
    features["date"].dt.dayofweek >= 5
).astype(int)

print("✓ Date features created")


# ============================================================
# [9] MERGE CALENDAR FEATURES
# ============================================================

print("\n[9] MERGING CALENDAR FEATURES")
print("-" * 80)

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

calendar_columns = [
    "date",
    "week",
    "season",
    "is_holiday",
    "promo_event"
]

calendar_columns = [
    c for c in calendar_columns
    if c in calendar.columns
]

calendar_features = (
    calendar[calendar_columns]
    .drop_duplicates("date")
    .copy()
)

features = features.merge(
    calendar_features,
    on="date",
    how="left"
)

# Calendar promotion should affect all SKU-store
# combinations on that date.

if "promo_event" in features.columns:

    calendar_promo = pd.to_numeric(
        features["promo_event"],
        errors="coerce"
    ).fillna(0)

    original_promo = pd.to_numeric(
        features["promo_flag"],
        errors="coerce"
    ).fillna(0)

    features["promo_flag"] = (
        (calendar_promo > 0) |
        (original_promo > 0)
    ).astype(int)

else:

    features["promo_flag"] = (
        pd.to_numeric(
            features["promo_flag"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

print("✓ Calendar features merged")


# ============================================================
# [10] MERGE SKU MASTER
# ============================================================

print("\n[10] MERGING SKU MASTER FEATURES")
print("-" * 80)

sku_columns = [
    "sku_id",
    "category",
    "brand"
]

optional_sku_columns = [
    "product_name",
    "unit_cost",
    "selling_price"
]

for column in optional_sku_columns:

    if column in sku_master.columns:
        sku_columns.append(column)

sku_features = (
    sku_master[
        [
            c for c in sku_columns
            if c in sku_master.columns
        ]
    ]
    .drop_duplicates("sku_id")
)

features = features.merge(
    sku_features,
    on="sku_id",
    how="left"
)

print("✓ SKU master features merged")


# ============================================================
# [11] MERGE STORE MASTER
# ============================================================

print("\n[11] MERGING STORE MASTER FEATURES")
print("-" * 80)

store_columns = [
    "store_id",
    "store_name",
    "store_type",
    "city"
]

store_columns = [
    c for c in store_columns
    if c in store_master.columns
]

store_features = (
    store_master[
        store_columns
    ]
    .drop_duplicates("store_id")
)

features = features.merge(
    store_features,
    on="store_id",
    how="left"
)

print("✓ Store master features merged")


# ============================================================
# [12] DEMAND LAG FEATURES
# ============================================================

print("\n[12] CREATING DEMAND LAG FEATURES")
print("-" * 80)

group_columns = [
    "sku_id",
    "store_id"
]

# Because the dataset is now DAILY,
# shift(7) really means 7 DAYS ago.

lag_periods = [
    1,
    7,
    14,
    28
]

for lag in lag_periods:

    features[f"lag_{lag}"] = (
        features
        .groupby(group_columns)["units_sold"]
        .shift(lag)
    )

    valid_count = (
        features[f"lag_{lag}"]
        .notna()
        .sum()
    )

    print(
        f"✓ lag_{lag} "
        f"(valid values: {valid_count:,})"
    )


# ============================================================
# [13] ROLLING DEMAND FEATURES
# ============================================================

print("\n[13] CREATING ROLLING DEMAND FEATURES")
print("-" * 80)

rolling_windows = [
    7,
    14,
    28
]

for window in rolling_windows:

    features[f"rolling_mean_{window}"] = (
        features
        .groupby(group_columns)["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )
    )

    features[f"rolling_std_{window}"] = (
        features
        .groupby(group_columns)["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=window,
                min_periods=2
            )
            .std()
        )
    )

    print(
        f"✓ rolling_mean_{window}"
    )

    print(
        f"✓ rolling_std_{window}"
    )


# ============================================================
# [14] EXPANDING DEMAND FEATURES
# ============================================================

print("\n[14] CREATING EXPANDING DEMAND FEATURES")
print("-" * 80)

features["expanding_mean"] = (
    features
    .groupby(group_columns)["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )
)

print("✓ expanding_mean")


# ============================================================
# [15] DEMAND TREND FEATURES
# ============================================================

print("\n[15] CREATING DEMAND TREND FEATURES")
print("-" * 80)

features["demand_change_1d"] = (
    features["lag_1"]
    - features["lag_7"]
)

features["demand_ratio_1d_7d"] = (
    features["lag_1"]
    /
    features["lag_7"].replace(
        0,
        np.nan
    )
)

features["demand_ratio_1d_7d"] = (
    features["demand_ratio_1d_7d"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

print("✓ demand_change_1d")
print("✓ demand_ratio_1d_7d")


# ============================================================
# [16] PROMOTION HISTORY FEATURES
# ============================================================

print("\n[16] CREATING PROMOTION HISTORY FEATURES")
print("-" * 80)

features["promo_lag_1"] = (
    features
    .groupby(group_columns)["promo_flag"]
    .shift(1)
)

features["promo_lag_7"] = (
    features
    .groupby(group_columns)["promo_flag"]
    .shift(7)
)

print("✓ promo_lag_1")
print("✓ promo_lag_7")


# ============================================================
# [17] PRICE FEATURES
# ============================================================

print("\n[17] CREATING PRICE FEATURES")
print("-" * 80)

features["price_lag_1"] = (
    features
    .groupby(group_columns)["unit_price"]
    .shift(1)
)

features["price_change"] = (
    features["unit_price"]
    - features["price_lag_1"]
)

features["price_change_pct"] = (
    features["price_change"]
    /
    features["price_lag_1"].replace(
        0,
        np.nan
    )
)

features["price_change_pct"] = (
    features["price_change_pct"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

print("✓ price_lag_1")
print("✓ price_change")
print("✓ price_change_pct")


# ============================================================
# [18] INVENTORY FEATURES
# ============================================================

print("\n[18] MERGING INVENTORY FEATURES")
print("-" * 80)

# inventory_snapshot_clean.csv contains one current
# inventory record per SKU-store combination.
#
# Therefore we use it as current inventory/context,
# NOT as historical daily inventory.

inventory_columns = [
    "sku_id",
    "store_id",
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]

available_inventory_columns = [
    c for c in inventory_columns
    if c in inventory.columns
]

inventory_features = (
    inventory[
        available_inventory_columns
    ]
    .drop_duplicates(
        subset=["sku_id", "store_id"]
    )
)

features = features.merge(
    inventory_features,
    on=["sku_id", "store_id"],
    how="left"
)

print("✓ Inventory features merged")


# ============================================================
# [19] INVENTORY-DERIVED FEATURES
# ============================================================

print("\n[19] CREATING INVENTORY-DERIVED FEATURES")
print("-" * 80)

features["stock_vs_reorder"] = (
    features["stock_on_hand"]
    - features["reorder_point"]
)

features["stock_vs_safety"] = (
    features["stock_on_hand"]
    - features["safety_stock"]
)

features["below_reorder_point"] = (
    features["stock_on_hand"]
    <
    features["reorder_point"]
).astype(int)

features["below_safety_stock"] = (
    features["stock_on_hand"]
    <
    features["safety_stock"]
).astype(int)

features["stockout_flag"] = (
    features["stock_on_hand"]
    <= 0
).astype(int)

print("✓ stock_vs_reorder")
print("✓ stock_vs_safety")
print("✓ below_reorder_point")
print("✓ below_safety_stock")
print("✓ stockout_flag")


# ============================================================
# [21] SKU-LEVEL HISTORICAL FEATURES
# ============================================================

print("\n[21] CREATING SKU-LEVEL FEATURES")
print("-" * 80)

# IMPORTANT:
# These are historical expanding features.
# We DO NOT calculate statistics using the entire dataset,
# because that would allow future sales to influence the past.

features["sku_total_units"] = (
    features
    .groupby("sku_id")["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .sum()
    )
)

features["sku_avg_units"] = (
    features
    .groupby("sku_id")["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )
)

features["sku_total_revenue"] = (
    features
    .groupby("sku_id")["revenue"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .sum()
    )
)

features["sku_avg_revenue"] = (
    features
    .groupby("sku_id")["revenue"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )
)

print("✓ sku_total_units")
print("✓ sku_avg_units")
print("✓ sku_total_revenue")
print("✓ sku_avg_revenue")


# ============================================================
# [22] STORE-LEVEL HISTORICAL FEATURES
# ============================================================

print("\n[22] CREATING STORE-LEVEL FEATURES")
print("-" * 80)

features["store_total_units"] = (
    features
    .groupby("store_id")["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .sum()
    )
)

features["store_avg_units"] = (
    features
    .groupby("store_id")["units_sold"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )
)

features["store_total_revenue"] = (
    features
    .groupby("store_id")["revenue"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .sum()
    )
)

features["store_avg_revenue"] = (
    features
    .groupby("store_id")["revenue"]
    .transform(
        lambda x:
        x.shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )
)

print("✓ store_total_units")
print("✓ store_avg_units")
print("✓ store_total_revenue")
print("✓ store_avg_revenue")


# ============================================================
# [23] CATEGORY HISTORICAL FEATURES
# ============================================================

print("\n[23] CREATING CATEGORY FEATURES")
print("-" * 80)

if "category" in features.columns:

    features["category_total_units"] = (
        features
        .groupby("category")["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .sum()
        )
    )

    features["category_avg_units"] = (
        features
        .groupby("category")["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .mean()
        )
    )

    features["category_total_revenue"] = (
        features
        .groupby("category")["revenue"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .sum()
        )
    )

    print("✓ category_total_units")
    print("✓ category_avg_units")
    print("✓ category_total_revenue")


# ============================================================
# [24] BRAND HISTORICAL FEATURES
# ============================================================

print("\n[24] CREATING BRAND FEATURES")
print("-" * 80)

if "brand" in features.columns:

    features["brand_total_units"] = (
        features
        .groupby("brand")["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .sum()
        )
    )

    features["brand_avg_units"] = (
        features
        .groupby("brand")["units_sold"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .mean()
        )
    )

    features["brand_total_revenue"] = (
        features
        .groupby("brand")["revenue"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding(
                min_periods=1
            )
            .sum()
        )
    )

    print("✓ brand_total_units")
    print("✓ brand_avg_units")
    print("✓ brand_total_revenue")


# ============================================================
# [25] INVENTORY FLAG FEATURES
# ============================================================

print("\n[25] CREATING INVENTORY FLAG FEATURES")
print("-" * 80)

if (
    "sku_id" in inventory_flags.columns
    and "flag" in inventory_flags.columns
):

    flag_summary = (
        inventory_flags
        .groupby("sku_id")
        .agg(
            flag_count=("flag", "count"),

            stockout_risk_flag=(
                "flag",
                lambda x:
                int(
                    "STOCKOUT_RISK"
                    in set(x)
                )
            ),

            slow_mover_flag=(
                "flag",
                lambda x:
                int(
                    "SLOW_MOVER"
                    in set(x)
                )
            )
        )
        .reset_index()
    )

    features = features.merge(
        flag_summary,
        on="sku_id",
        how="left"
    )

    features["flag_count"] = (
        features["flag_count"]
        .fillna(0)
    )

    features["stockout_risk_flag"] = (
        features["stockout_risk_flag"]
        .fillna(0)
        .astype(int)
    )

    features["slow_mover_flag"] = (
        features["slow_mover_flag"]
        .fillna(0)
        .astype(int)
    )

    print("✓ flag_count")
    print("✓ stockout_risk_flag")
    print("✓ slow_mover_flag")


# ============================================================
# [26] LAG FEATURE COVERAGE ANALYSIS
# ============================================================

print("\n[26] LAG FEATURE COVERAGE ANALYSIS")
print("-" * 80)

for lag in lag_periods:

    column = f"lag_{lag}"

    valid = features[column].notna().sum()
    missing = features[column].isna().sum()

    percentage = (
        valid / len(features) * 100
    )

    print(
        f"{column:10} "
        f"valid={valid:,} "
        f"missing={missing:,} "
        f"coverage={percentage:.2f}%"
    )

print()
print(
    "NOTE: Low lag coverage is expected when individual "
    "SKU-store time series contain few observations."
)


# ============================================================
# [27] MISSING VALUE ANALYSIS
# ============================================================

print("\n[27] ANALYZING FEATURE MISSING VALUES")
print("-" * 80)

missing_summary = (
    features
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_summary = missing_summary[
    missing_summary > 0
]

print("Top missing-value columns:")

if len(missing_summary) > 0:

    print(
        missing_summary.head(20)
    )

else:

    print(
        "No missing values found."
    )


# ============================================================
# [28] HANDLE FORECASTING FEATURE MISSING VALUES
# ============================================================

print("\n[28] HANDLING FORECASTING FEATURE MISSING VALUES")
print("-" * 80)

# For the first observations of each time series,
# historical values simply do not exist.
#
# We fill these unavailable historical values with 0
# so that the final dataset is model-ready.

forecast_feature_prefixes = (
    "lag_",
    "rolling_",
    "expanding_",
    "demand_",
    "promo_lag_",
    "price_lag_"
)

numeric_columns = features.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if column.startswith(
        forecast_feature_prefixes
    ):

        features[column] = (
            features[column]
            .fillna(0)
        )

print(
    "✓ Forecasting feature missing values handled"
)


# ============================================================
# [29] HANDLE CATEGORICAL MISSING VALUES
# ============================================================

print("\n[29] HANDLING CATEGORICAL MISSING VALUES")
print("-" * 80)

categorical_columns = (
    features
    .select_dtypes(
        include=[
            "object",
            "string"
        ]
    )
    .columns
)

for column in categorical_columns:

    features[column] = (
        features[column]
        .fillna("Unknown")
    )

print(
    "✓ Categorical missing values handled"
)


# ============================================================
# [30] FINAL NUMERIC CLEANUP
# ============================================================

print("\n[30] FINAL NUMERIC CLEANUP")
print("-" * 80)

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

numeric_columns = features.select_dtypes(
    include=np.number
).columns

features[numeric_columns] = (
    features[numeric_columns]
    .fillna(0)
)

print("✓ Infinite values handled")
print("✓ Remaining numeric missing values filled")


# ============================================================
# [31] SORT FINAL DATASET
# ============================================================

print("\n[31] SORTING FINAL FEATURE DATASET")
print("-" * 80)

features = features.sort_values(
    [
        "date",
        "store_id",
        "sku_id"
    ]
).reset_index(drop=True)

print("✓ Dataset sorted")


# ============================================================
# [32] FEATURE VALIDATION
# ============================================================

print("\n[32] FEATURE VALIDATION")
print("-" * 80)

print(
    f"Rows           : {len(features):,}"
)

print(
    f"Columns        : {len(features.columns):,}"
)

print(
    f"Missing cells  : "
    f"{features.isnull().sum().sum():,}"
)

print(
    f"Duplicate rows : "
    f"{features.duplicated().sum():,}"
)

if features.isnull().sum().sum() == 0:

    print("✓ No missing cells")

else:

    print("⚠ Missing cells still present")

if features.duplicated().sum() == 0:

    print("✓ No duplicate rows")

else:

    print("⚠ Duplicate rows detected")


# ============================================================
# [33] TARGET VALIDATION
# ============================================================

print("\n[33] TARGET VALIDATION")
print("-" * 80)

if "units_sold" not in features.columns:

    raise ValueError(
        "Target column 'units_sold' is missing!"
    )

print(
    "Target column : units_sold"
)

print(
    f"Target mean   : "
    f"{features['units_sold'].mean():.4f}"
)

print(
    f"Target min    : "
    f"{features['units_sold'].min():.4f}"
)

print(
    f"Target max    : "
    f"{features['units_sold'].max():.4f}"
)

print("✓ Target validated")


# ============================================================
# [34] FINAL FEATURE LIST
# ============================================================

print("\n[34] FINAL FEATURE LIST")
print("-" * 80)

for i, column in enumerate(
    features.columns,
    start=1
):

    print(
        f"{i:3}. {column}"
    )

print(
    f"\nTotal features/columns: "
    f"{len(features.columns)}"
)


# ============================================================
# [35] SAVE MAIN FEATURE DATASET
# ============================================================

print("\n[35] SAVING FEATURE DATASET")
print("-" * 80)

main_output = os.path.join(
    FEATURE_PATH,
    "forecasting_features.csv"
)

features.to_csv(
    main_output,
    index=False
)

print(
    "✓ forecasting_features.csv"
)


# ============================================================
# [36] SAVE FEATURE DICTIONARY
# ============================================================

print("\n[36] SAVING FEATURE DICTIONARY")
print("-" * 80)

feature_dictionary = pd.DataFrame({

    "feature": features.columns,

    "data_type": [
        str(features[column].dtype)
        for column in features.columns
    ],

    "missing_values": [
        int(
            features[column]
            .isnull()
            .sum()
        )
        for column in features.columns
    ],

    "unique_values": [
        int(
            features[column]
            .nunique()
        )
        for column in features.columns
    ]
})

feature_dictionary_output = os.path.join(
    FEATURE_PATH,
    "feature_dictionary.csv"
)

feature_dictionary.to_csv(
    feature_dictionary_output,
    index=False
)

print(
    "✓ feature_dictionary.csv"
)


# ============================================================
# [37] SAVE FEATURE SUMMARY
# ============================================================

print("\n[37] SAVING FEATURE SUMMARY")
print("-" * 80)

summary = pd.DataFrame({

    "metric": [

        "rows",
        "columns",
        "missing_cells",
        "duplicate_rows",
        "unique_skus",
        "unique_stores",
        "unique_dates",
        "target_mean",
        "target_min",
        "target_max"
    ],

    "value": [

        len(features),

        len(features.columns),

        int(
            features
            .isnull()
            .sum()
            .sum()
        ),

        int(
            features
            .duplicated()
            .sum()
        ),

        features["sku_id"].nunique(),

        features["store_id"].nunique(),

        features["date"].nunique(),

        features["units_sold"].mean(),

        features["units_sold"].min(),

        features["units_sold"].max()
    ]
})

summary_output = os.path.join(
    FEATURE_PATH,
    "feature_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print(
    "✓ feature_summary.csv"
)


# ============================================================
# [38] FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print(
    "PHASE 4 FEATURE ENGINEERING COMPLETED"
)
print("=" * 80)

print("\nFinal dataset:")

print(
    f"Rows    : {len(features):,}"
)

print(
    f"Columns : {len(features.columns):,}"
)

print("\nFiles generated:")

print(
    "✓ forecasting_features.csv"
)

print(
    "✓ feature_dictionary.csv"
)

print(
    "✓ feature_summary.csv"
)

print("\nOutput directory:")

print(
    FEATURE_PATH
)

print("\nNEXT PHASE:")

print(
    "PHASE 5 → DEMAND FORECASTING"
)

print("=" * 80)