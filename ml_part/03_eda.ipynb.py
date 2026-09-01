# ============================================================
# PROJECT FORESIGHT
# PHASE 3 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# [1] PATHS
# ============================================================

# FIX (critical reproducibility bug): this was a hardcoded, machine-specific
# Windows path (C:\Users\...\Project_Forsight), so the script only ran on
# one laptop. A grader cloning the repo could not run it at all. Anchoring
# to this script's own location makes the pipeline portable, per the
# "reproducibility check" in the engagement brief (Section 13).
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
EDA_DIR = BASE_DIR / "data" / "eda"

EDA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("PROJECT FORESIGHT - PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 75)

print("\n## [1] PATHS")
print("Processed data:", PROCESSED_DIR)
print("EDA output    :", EDA_DIR)

# ============================================================
# [2] REQUIRED FILES
# ============================================================

print("\n## [2] CHECKING PROCESSED FILES")

required_files = {
    "sales": "sales_daily_clean.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory": "inventory_snapshot_clean.csv",
    "store_master": "store_master_clean.csv",
    "promotions": "promotions_clean.csv",
    "customer_master": "customer_master_clean.csv",
    "inventory_flags": "sku_inventory_flags_clean.csv",
}

for dataset_name, filename in required_files.items():

    file_path = PROCESSED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required processed file not found:\n{file_path}"
        )

    print(f"✓ {filename}")

# ============================================================
# [3] LOAD PROCESSED DATA
# ============================================================

print("\n## [3] LOADING PROCESSED DATA")

sales = pd.read_csv(
    PROCESSED_DIR / "sales_daily_clean.csv"
)

sku_master = pd.read_csv(
    PROCESSED_DIR / "sku_master_clean.csv"
)

calendar = pd.read_csv(
    PROCESSED_DIR / "calendar_clean.csv"
)

inventory = pd.read_csv(
    PROCESSED_DIR / "inventory_snapshot_clean.csv"
)

store_master = pd.read_csv(
    PROCESSED_DIR / "store_master_clean.csv"
)

promotions = pd.read_csv(
    PROCESSED_DIR / "promotions_clean.csv"
)

customer_master = pd.read_csv(
    PROCESSED_DIR / "customer_master_clean.csv"
)

inventory_flags = pd.read_csv(
    PROCESSED_DIR / "sku_inventory_flags_clean.csv"
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
# [4] DATE CONVERSION
# ============================================================

print("\n## [4] CONVERTING DATE COLUMNS")

sales["date"] = pd.to_datetime(
    sales["date"],
    errors="coerce"
)

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

inventory["last_restock_date"] = pd.to_datetime(
    inventory["last_restock_date"],
    errors="coerce"
)

store_master["opening_date"] = pd.to_datetime(
    store_master["opening_date"],
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

customer_master["registration_date"] = pd.to_datetime(
    customer_master["registration_date"],
    errors="coerce"
)

print("✓ Dates converted")

# ============================================================
# [5] REQUIRED COLUMN VALIDATION
# ============================================================

print("\n## [5] VALIDATING REQUIRED COLUMNS")

required_columns = {

    "sales": [
        "date",
        "sku_id",
        "store_id",
        "units_sold",
        "revenue",
        "unit_price",
        "promo_flag",
    ],

    "sku_master": [
        "sku_id",
        "sku_name",
        "category",
        "brand",
    ],

    "calendar": [
        "date",
        "season",
        "is_holiday",
        "week",
    ],

    "inventory": [
        "sku_id",
        "store_id",
        "stock_on_hand",
        "reorder_point",
        "safety_stock",
    ],

    "store_master": [
        "store_id",
        "store_name",
        "city",
        "store_type",
    ],

    "customer_master": [
        "cust_id",
        "age",
        "loyalty_segment",
        "preferred_channel",
    ],

    "inventory_flags": [
        "sku_id",
        "flag",
    ],
}

datasets = {
    "sales": sales,
    "sku_master": sku_master,
    "calendar": calendar,
    "inventory": inventory,
    "store_master": store_master,
    "customer_master": customer_master,
    "inventory_flags": inventory_flags,
}

for dataset_name, columns in required_columns.items():

    missing_columns = [
        column
        for column in columns
        if column not in datasets[dataset_name].columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    print(f"✓ {dataset_name}")

# ============================================================
# [6] DATASET SUMMARY
# ============================================================

print("\n## [6] DATASET SUMMARY")

dataset_summary = pd.DataFrame({
    "dataset": [
        "Sales",
        "SKU Master",
        "Calendar",
        "Inventory",
        "Store Master",
        "Promotions",
        "Customer Master",
        "Inventory Flags",
    ],

    "rows": [
        len(sales),
        len(sku_master),
        len(calendar),
        len(inventory),
        len(store_master),
        len(promotions),
        len(customer_master),
        len(inventory_flags),
    ],

    "columns": [
        sales.shape[1],
        sku_master.shape[1],
        calendar.shape[1],
        inventory.shape[1],
        store_master.shape[1],
        promotions.shape[1],
        customer_master.shape[1],
        inventory_flags.shape[1],
    ],
})

print(dataset_summary.to_string(index=False))

# ============================================================
# [7] OVERALL SALES SUMMARY
# ============================================================

print("\n## [7] OVERALL SALES SUMMARY")

total_units = sales["units_sold"].sum()
total_revenue = sales["revenue"].sum()
unique_skus = sales["sku_id"].nunique()
unique_stores = sales["store_id"].nunique()
unique_dates = sales["date"].nunique()

daily_sales = (
    sales
    .groupby("date")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .reset_index()
)

average_daily_revenue = daily_sales["revenue"].mean()

print(f"Total units sold       : {total_units:,.0f}")
print(f"Total revenue          : ₹{total_revenue:,.2f}")
print(f"Average daily revenue  : ₹{average_daily_revenue:,.2f}")
print(f"Unique SKUs            : {unique_skus:,}")
print(f"Unique stores          : {unique_stores:,}")
print(f"Unique dates           : {unique_dates:,}")

# ============================================================
# [8] DAILY SALES TREND
# ============================================================

print("\n## [8] DAILY SALES TREND")

print(daily_sales.head())

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["revenue"],
    marker="o",
    markersize=2
)

plt.title("Daily Revenue Trend")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "daily_revenue_trend.png",
    dpi=300
)

plt.show()

# ============================================================
# [9] MONTHLY SALES TREND
# ============================================================

print("\n## [9] MONTHLY SALES TREND")

sales["month"] = (
    sales["date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    sales
    .groupby("month")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .reset_index()
)

print(monthly_sales)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["revenue"],
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "monthly_revenue_trend.png",
    dpi=300
)

plt.show()

# ============================================================
# [10] WEEKLY SALES TREND
# ============================================================

print("\n## [10] WEEKLY SALES TREND")

weekly_sales = (
    sales
    .assign(
        week=sales["date"].dt.to_period("W").astype(str)
    )
    .groupby("week")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .reset_index()
)

print(weekly_sales.head(10))

plt.figure(figsize=(14, 6))

plt.plot(
    weekly_sales["week"],
    weekly_sales["revenue"],
    marker="o",
    markersize=3
)

plt.title("Weekly Revenue Trend")
plt.xlabel("Week")
plt.ylabel("Revenue")
plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "weekly_revenue_trend.png",
    dpi=300
)

plt.show()

# ============================================================
# [11] TOP 10 SKUs BY REVENUE
# ============================================================

print("\n## [11] TOP 10 SKUs BY REVENUE")

top_skus_revenue = (
    sales
    .groupby("sku_id")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .head(10)
    .reset_index()
)

top_skus_revenue = top_skus_revenue.merge(
    sku_master[
        ["sku_id", "sku_name", "category", "brand"]
    ],
    on="sku_id",
    how="left"
)

print(top_skus_revenue)

plt.figure(figsize=(12, 6))

plt.bar(
    top_skus_revenue["sku_id"].astype(str),
    top_skus_revenue["revenue"]
)

plt.title("Top 10 SKUs by Revenue")
plt.xlabel("SKU")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "top_10_skus_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [12] TOP 10 SKUs BY UNITS SOLD
# ============================================================

print("\n## [12] TOP 10 SKUs BY UNITS SOLD")

top_skus_units = (
    sales
    .groupby("sku_id")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "units_sold",
        ascending=False
    )
    .head(10)
    .reset_index()
)

top_skus_units = top_skus_units.merge(
    sku_master[
        ["sku_id", "sku_name", "category", "brand"]
    ],
    on="sku_id",
    how="left"
)

print(top_skus_units)

plt.figure(figsize=(12, 6))

plt.bar(
    top_skus_units["sku_id"].astype(str),
    top_skus_units["units_sold"]
)

plt.title("Top 10 SKUs by Units Sold")
plt.xlabel("SKU")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "top_10_skus_units.png",
    dpi=300
)

plt.show()

# ============================================================
# [13] CATEGORY PERFORMANCE
# ============================================================

print("\n## [13] CATEGORY PERFORMANCE")

category_sales = (
    sales
    .merge(
        sku_master[
            ["sku_id", "category"]
        ],
        on="sku_id",
        how="left"
    )
    .groupby("category")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print(category_sales)

plt.figure(figsize=(12, 6))

plt.bar(
    category_sales["category"].astype(str),
    category_sales["revenue"]
)

plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "category_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [14] BRAND PERFORMANCE
# ============================================================

print("\n## [14] BRAND PERFORMANCE")

brand_sales = (
    sales
    .merge(
        sku_master[
            ["sku_id", "brand"]
        ],
        on="sku_id",
        how="left"
    )
    .groupby("brand")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print("\nTop 10 Brands:")
print(brand_sales.head(10))

plt.figure(figsize=(12, 6))

plt.bar(
    brand_sales.head(10)["brand"].astype(str),
    brand_sales.head(10)["revenue"]
)

plt.title("Top 10 Brands by Revenue")
plt.xlabel("Brand")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "top_10_brands.png",
    dpi=300
)

plt.show()

# ============================================================
# [15] STORE PERFORMANCE
# ============================================================

print("\n## [15] STORE PERFORMANCE")

store_sales = (
    sales
    .merge(
        store_master[
            ["store_id", "store_name", "city", "store_type"]
        ],
        on="store_id",
        how="left"
    )
    .groupby(
        [
            "store_id",
            "store_name",
            "city",
            "store_type"
        ]
    )
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print("\nTop 10 Stores:")
print(store_sales.head(10))

plt.figure(figsize=(12, 6))

plt.bar(
    store_sales.head(10)["store_id"].astype(str),
    store_sales.head(10)["revenue"]
)

plt.title("Top 10 Stores by Revenue")
plt.xlabel("Store")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "top_10_stores_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [16] STORE TYPE PERFORMANCE
# ============================================================

print("\n## [16] STORE TYPE PERFORMANCE")

store_type_sales = (
    sales
    .merge(
        store_master[
            ["store_id", "store_type"]
        ],
        on="store_id",
        how="left"
    )
    .groupby("store_type")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print(store_type_sales)

plt.figure(figsize=(10, 5))

plt.bar(
    store_type_sales["store_type"].astype(str),
    store_type_sales["revenue"]
)

plt.title("Revenue by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "store_type_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [17] CITY PERFORMANCE
# ============================================================

print("\n## [17] CITY PERFORMANCE")

city_sales = (
    sales
    .merge(
        store_master[
            ["store_id", "city"]
        ],
        on="store_id",
        how="left"
    )
    .groupby("city")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print(city_sales)

plt.figure(figsize=(12, 6))

plt.bar(
    city_sales["city"].astype(str),
    city_sales["revenue"]
)

plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "city_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [18] PROMOTION VS NON-PROMOTION
# ============================================================

print("\n## [18] PROMOTION VS NON-PROMOTION SALES")

promotion_analysis = (
    sales
    .groupby("promo_flag")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        records=("promo_flag", "size")
    )
    .reset_index()
)

promotion_analysis["promotion_status"] = (
    promotion_analysis["promo_flag"]
    .map({
        0: "No Promotion",
        1: "Promotion"
    })
    .fillna("Unknown")
)

print(promotion_analysis)

plt.figure(figsize=(8, 5))

plt.bar(
    promotion_analysis["promotion_status"],
    promotion_analysis["revenue"]
)

plt.title("Promotion vs Non-Promotion Revenue")
plt.xlabel("Promotion Status")
plt.ylabel("Revenue")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "promotion_vs_nonpromotion.png",
    dpi=300
)

plt.show()

# ============================================================
# [19] PROMOTION LIFT
# ============================================================

print("\n## [19] PROMOTION SALES LIFT")

promotion_average = (
    sales
    .groupby("promo_flag")
    .agg(
        avg_units_sold=("units_sold", "mean"),
        avg_revenue=("revenue", "mean"),
        records=("promo_flag", "size")
    )
)

print(promotion_average)

if 0 in promotion_average.index and 1 in promotion_average.index:

    promo_units = promotion_average.loc[
        1, "avg_units_sold"
    ]

    nonpromo_units = promotion_average.loc[
        0, "avg_units_sold"
    ]

    promo_revenue = promotion_average.loc[
        1, "avg_revenue"
    ]

    nonpromo_revenue = promotion_average.loc[
        0, "avg_revenue"
    ]

    unit_lift = (
        (promo_units - nonpromo_units)
        / nonpromo_units
        * 100
        if nonpromo_units != 0
        else np.nan
    )

    revenue_lift = (
        (promo_revenue - nonpromo_revenue)
        / nonpromo_revenue
        * 100
        if nonpromo_revenue != 0
        else np.nan
    )

    print(
        f"Average units during promotion : {promo_units:.2f}"
    )

    print(
        f"Average units without promotion: {nonpromo_units:.2f}"
    )

    print(
        f"Unit sales lift                 : {unit_lift:.2f}%"
    )

    print(
        f"Revenue lift                    : {revenue_lift:.2f}%"
    )

else:

    print(
        "Promotion lift cannot be calculated because "
        "both promo_flag groups are not present."
    )

# ============================================================
# [20] SEASONAL PERFORMANCE
# ============================================================

print("\n## [20] SEASONAL PERFORMANCE")

season_sales = (
    sales
    .merge(
        calendar[
            ["date", "season"]
        ],
        on="date",
        how="left"
    )
    .groupby("season")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum")
    )
    .sort_values(
        "revenue",
        ascending=False
    )
    .reset_index()
)

print(season_sales)

plt.figure(figsize=(10, 5))

plt.bar(
    season_sales["season"].astype(str),
    season_sales["revenue"]
)

plt.title("Revenue by Season")
plt.xlabel("Season")
plt.ylabel("Revenue")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "seasonal_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [21] HOLIDAY VS NON-HOLIDAY
# ============================================================

print("\n## [21] HOLIDAY VS NON-HOLIDAY SALES")

holiday_sales = (
    sales
    .merge(
        calendar[
            ["date", "is_holiday"]
        ],
        on="date",
        how="left"
    )
    .groupby("is_holiday")
    .agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_units=("units_sold", "mean"),
        avg_revenue=("revenue", "mean")
    )
    .reset_index()
)

holiday_sales["holiday_status"] = (
    holiday_sales["is_holiday"]
    .map({
        0: "Non-Holiday",
        1: "Holiday"
    })
    .fillna("Unknown")
)

print(holiday_sales)

plt.figure(figsize=(8, 5))

plt.bar(
    holiday_sales["holiday_status"],
    holiday_sales["revenue"]
)

plt.title("Holiday vs Non-Holiday Revenue")
plt.xlabel("Day Type")
plt.ylabel("Revenue")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "holiday_revenue.png",
    dpi=300
)

plt.show()

# ============================================================
# [22] ZERO DEMAND ANALYSIS
# ============================================================

print("\n## [22] ZERO DEMAND ANALYSIS")

zero_demand_records = (
        sales["units_sold"] == 0
).sum()

total_sales_records = len(sales)

zero_demand_percent = (
        zero_demand_records
        / total_sales_records
        * 100
)

print(
    f"Sales records       : {total_sales_records:,}"
)

print(
    f"Zero-demand records : {zero_demand_records:,}"
)

print(
    f"Zero-demand %       : {zero_demand_percent:.2f}%"
)

# ============================================================
# [23] INVENTORY HEALTH
# ============================================================

print("\n## [23] INVENTORY HEALTH")

inventory_analysis = inventory.copy()

inventory_analysis["below_reorder"] = (
        inventory_analysis["stock_on_hand"]
        < inventory_analysis["reorder_point"]
)

inventory_analysis["below_safety"] = (
        inventory_analysis["stock_on_hand"]
        < inventory_analysis["safety_stock"]
)

inventory_analysis["stockout"] = (
        inventory_analysis["stock_on_hand"] == 0
)

print(
    f"Total inventory records : {len(inventory_analysis):,}"
)

print(
    f"Below reorder point     : "
    f"{inventory_analysis['below_reorder'].sum():,}"
)

print(
    f"Below safety stock      : "
    f"{inventory_analysis['below_safety'].sum():,}"
)

print(
    f"Stockout records        : "
    f"{inventory_analysis['stockout'].sum():,}"
)

# ============================================================
# [24] INVENTORY STATUS DISTRIBUTION
# ============================================================

print("\n## [24] INVENTORY STATUS DISTRIBUTION")


def get_inventory_status(row):
    stock = row["stock_on_hand"]
    safety = row["safety_stock"]
    reorder = row["reorder_point"]

    if stock == 0:
        return "Stockout"

    if stock < safety:
        return "Critical"

    if stock < reorder:
        return "Reorder Required"

    if stock > reorder * 3:
        return "Overstock"

    return "Healthy"


inventory_analysis["inventory_status"] = (
    inventory_analysis
    .apply(
        get_inventory_status,
        axis=1
    )
)

inventory_status_summary = (
    inventory_analysis["inventory_status"]
    .value_counts()
    .rename_axis("status")
    .reset_index(name="count")
)

print(inventory_status_summary)

plt.figure(figsize=(10, 5))

plt.bar(
    inventory_status_summary["status"],
    inventory_status_summary["count"]
)

plt.title("Inventory Health Distribution")
plt.xlabel("Inventory Status")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    EDA_DIR / "inventory_health.png",
    dpi=300
)

plt.show()

# ============================================================
# [25] STOCKOUT RISK BY SKU
# ============================================================

print("\n## [25] STOCKOUT RISK BY SKU")

sku_inventory_risk = (
    inventory_analysis
    .groupby("sku_id")
    .agg(
        total_records=("sku_id", "size"),
        stockout_records=("stockout", "sum"),
        avg_stock=("stock_on_hand", "mean"),
        avg_reorder_point=("reorder_point", "mean")
    )
    .reset_index()
)

sku_inventory_risk["stockout_rate"] = (
        sku_inventory_risk["stockout_records"]
        / sku_inventory_risk["total_records"]
        * 100
)

sku_inventory_risk = (
    sku_inventory_risk
    .sort_values(
        [
            "stockout_rate",
            "stockout_records"
        ],
        ascending=False
    )
)

print(
    sku_inventory_risk.head(20)
)

# ============================================================
# [26] OVERSTOCK ANALYSIS
# ============================================================

print("\n## [26] OVERSTOCK ANALYSIS")

overstock = inventory_analysis[
    inventory_analysis["inventory_status"] == "Overstock"
    ]

print(
    f"Overstock records : {len(overstock):,}"
)

if not overstock.empty:

    overstock_skus = (
        overstock
        .groupby("sku_id")
        .agg(
            records=("sku_id", "size"),
            avg_stock=("stock_on_hand", "mean"),
            avg_reorder_point=("reorder_point", "mean")
        )
        .sort_values(
            "avg_stock",
            ascending=False
        )
        .reset_index()
    )

    print("\nTop Overstock SKUs:")
    print(overstock_skus.head(20))

else:

    overstock_skus = pd.DataFrame(
        columns=[
            "sku_id",
            "records",
            "avg_stock",
            "avg_reorder_point"
        ]
    )

    print("No overstock records found.")

# ============================================================
# [27] SALES DISTRIBUTION
# ============================================================

print("\n## [27] SALES DISTRIBUTION")

print(
    sales["units_sold"].describe()
)

plt.figure(figsize=(10, 5))

plt.hist(
    sales["units_sold"],
    bins=30
)

plt.title("Distribution of Units Sold")
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "sales_distribution.png",
    dpi=300
)

plt.show()

# ============================================================
# [28] REVENUE DISTRIBUTION
# ============================================================

print("\n## [28] REVENUE DISTRIBUTION")

print(
    sales["revenue"].describe()
)

plt.figure(figsize=(10, 5))

plt.hist(
    sales["revenue"],
    bins=30
)

plt.title("Distribution of Revenue")
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "revenue_distribution.png",
    dpi=300
)

plt.show()

# ============================================================
# [29] PRICE VS DEMAND
# ============================================================

print("\n## [29] PRICE VS DEMAND")

price_demand_corr = (
    sales[
        ["unit_price", "units_sold"]
    ]
    .corr()
)

print(price_demand_corr)

plt.figure(figsize=(8, 6))

plt.scatter(
    sales["unit_price"],
    sales["units_sold"],
    alpha=0.3
)

plt.title("Unit Price vs Units Sold")
plt.xlabel("Unit Price")
plt.ylabel("Units Sold")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "price_vs_demand.png",
    dpi=300
)

plt.show()

# ============================================================
# [30] CORRELATION ANALYSIS
# ============================================================

print("\n## [30] CORRELATION ANALYSIS")

numeric_columns = [
    "units_sold",
    "revenue",
    "unit_price",
    "promo_flag"
]

correlation_matrix = (
    sales[numeric_columns]
    .corr()
)

print(correlation_matrix)

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title("Sales Feature Correlation Matrix")
plt.tight_layout()

plt.savefig(
    EDA_DIR / "correlation_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# [31] CUSTOMER ANALYSIS
# ============================================================

print("\n## [31] CUSTOMER MASTER ANALYSIS")

total_customers = (
    customer_master["cust_id"]
    .nunique()
)

print(
    f"Total customers: {total_customers:,}"
)

print("\nLoyalty Segment Distribution:")

print(
    customer_master[
        "loyalty_segment"
    ].value_counts()
)

print("\nPreferred Channel Distribution:")

print(
    customer_master[
        "preferred_channel"
    ].value_counts()
)

print("\nCustomer Age Statistics:")

print(
    customer_master[
        "age"
    ].describe()
)

# ============================================================
# [32] INVENTORY FLAGS ANALYSIS
# ============================================================

print("\n## [32] INVENTORY FLAGS ANALYSIS")

print(
    f"Total inventory flags: {len(inventory_flags):,}"
)

print("\nFlag distribution:")

print(
    inventory_flags[
        "flag"
    ].value_counts()
)

print("\nTop flagged SKUs:")

print(
    inventory_flags[
        "sku_id"
    ]
    .value_counts()
    .head(20)
)

# ============================================================
# [33] PRODUCTS WITH INVENTORY RISK
# ============================================================

print("\n## [33] TOP PRODUCTS WITH INVENTORY RISK")

risk_products = (
    inventory_analysis
    .groupby("sku_id")
    .agg(
        avg_stock=("stock_on_hand", "mean"),
        avg_reorder=("reorder_point", "mean"),
        avg_safety=("safety_stock", "mean"),
        stockout_count=("stockout", "sum"),
        critical_count=("below_safety", "sum")
    )
    .reset_index()
)

risk_products = (
    risk_products
    .sort_values(
        [
            "stockout_count",
            "critical_count"
        ],
        ascending=False
    )
)

risk_products = risk_products.merge(
    sku_master[
        [
            "sku_id",
            "sku_name",
            "category",
            "brand"
        ]
    ],
    on="sku_id",
    how="left"
)

print(
    risk_products.head(20)
)

# ============================================================
# [34] FINAL BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PHASE 3 - EDA BUSINESS SUMMARY")
print("=" * 75)

stockout_records = int(
    inventory_analysis["stockout"].sum()
)

below_safety_records = int(
    inventory_analysis["below_safety"].sum()
)

below_reorder_records = int(
    inventory_analysis["below_reorder"].sum()
)

overstock_records = int(
    (
            inventory_analysis["inventory_status"]
            == "Overstock"
    ).sum()
)

print(
    f"""
1. Total Revenue
   ₹{total_revenue:,.2f}

2. Total Units Sold
   {total_units:,.0f}

3. Average Daily Revenue
   ₹{average_daily_revenue:,.2f}

4. Unique SKUs
   {unique_skus:,}

5. Unique Stores
   {unique_stores:,}

6. Sales Days
   {unique_dates:,}

7. Zero-Demand Records
   {zero_demand_records:,}
   ({zero_demand_percent:.2f}%)

8. Inventory Records
   {len(inventory_analysis):,}

9. Stockout Records
   {stockout_records:,}

10. Below Safety Stock
    {below_safety_records:,}

11. Below Reorder Point
    {below_reorder_records:,}

12. Overstock Records
    {overstock_records:,}
"""
)

# ============================================================
# [35] SAVE EDA TABLES
# ============================================================

print("\n## [35] SAVING EDA RESULTS")

eda_tables = {

    "dataset_summary.csv":
        dataset_summary,

    "daily_sales.csv":
        daily_sales,

    "monthly_sales.csv":
        monthly_sales,

    "weekly_sales.csv":
        weekly_sales,

    "top_skus_revenue.csv":
        top_skus_revenue,

    "top_skus_units.csv":
        top_skus_units,

    "category_sales.csv":
        category_sales,

    "brand_sales.csv":
        brand_sales,

    "store_sales.csv":
        store_sales,

    "store_type_sales.csv":
        store_type_sales,

    "city_sales.csv":
        city_sales,

    "promotion_analysis.csv":
        promotion_analysis,

    "season_sales.csv":
        season_sales,

    "holiday_sales.csv":
        holiday_sales,

    "inventory_status.csv":
        inventory_status_summary,

    "sku_inventory_risk.csv":
        sku_inventory_risk,

    "risk_products.csv":
        risk_products,

}

for filename, dataframe in eda_tables.items():
    dataframe.to_csv(
        EDA_DIR / filename,
        index=False
    )

    print(f"✓ {filename}")

# ============================================================
# [36] PHASE 3 COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("PHASE 3 - EDA COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    """
✓ Daily sales analysis
✓ Monthly sales analysis
✓ Weekly sales analysis
✓ Top SKU revenue analysis
✓ Top SKU unit analysis
✓ Category analysis
✓ Brand analysis
✓ Store analysis
✓ Store type analysis
✓ City analysis
✓ Promotion analysis
✓ Promotion lift analysis
✓ Seasonal analysis
✓ Holiday analysis
✓ Zero-demand analysis
✓ Inventory health analysis
✓ Stockout risk analysis
✓ Overstock analysis
✓ Sales distribution analysis
✓ Revenue distribution analysis
✓ Price-demand analysis
✓ Correlation analysis
✓ Customer analysis
✓ Inventory flag analysis
✓ Inventory risk-product analysis
✓ EDA tables saved
✓ EDA charts saved

NEXT PHASE:
PHASE 4 → FEATURE ENGINEERING
"""
)

print(
    f"\nAll EDA outputs are available in:\n{EDA_DIR}"
)
