# ============================================================
# PROJECT FORESIGHT
# PHASE 2: DATA CLEANING & DATA ENGINEERING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PROJECT FORESIGHT - PHASE 2: DATA CLEANING & DATA ENGINEERING")
print("=" * 80)


# ============================================================
# 1. PATHS
# ============================================================

# FIX (reproducibility): anchor to this script's location instead of the
# caller's current working directory. See the same fix in
# 01_data_understanding.ipynb.py for the full rationale.
BASE_PATH = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_PATH / "data" / "raw"
PROCESSED_PATH = BASE_PATH / "data" / "processed"

# Create processed folder if it does not exist
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

print("\n[1] PATHS")
print("-" * 80)

print("Raw data      :", RAW_PATH.resolve())
print("Processed data:", PROCESSED_PATH.resolve())


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("\n[2] LOADING RAW DATA")
print("-" * 80)

sales = pd.read_csv(
    RAW_PATH / "sales_daily.csv"
)

sku_master = pd.read_csv(
    RAW_PATH / "sku_master.csv"
)

calendar = pd.read_csv(
    RAW_PATH / "calendar.csv"
)

inventory = pd.read_csv(
    RAW_PATH / "inventory_snapshot.csv"
)

store_master = pd.read_csv(
    RAW_PATH / "store_master.csv"
)

promotions = pd.read_csv(
    RAW_PATH / "promotions.csv"
)

customer_master = pd.read_csv(
    RAW_PATH / "customer_master.csv"
)

inventory_flags = pd.read_csv(
    RAW_PATH / "sku_inventory_flags.csv"
)

print("✓ All raw datasets loaded successfully")


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

print("\n[3] STANDARDIZING COLUMN NAMES")
print("-" * 80)

datasets = {
    "sales": sales,
    "sku_master": sku_master,
    "calendar": calendar,
    "inventory": inventory,
    "store_master": store_master,
    "promotions": promotions,
    "customer_master": customer_master,
    "inventory_flags": inventory_flags
}

for name, df in datasets.items():

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print(f"✓ {name}")


# ============================================================
# 4. CLEAN STRING COLUMNS
# ============================================================

print("\n[4] CLEANING STRING COLUMNS")
print("-" * 80)

for name, df in datasets.items():

    string_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in string_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    print(f"✓ {name}")


# ============================================================
# 5. NORMALIZE SKU IDs
# ============================================================

print("\n[5] NORMALIZING SKU IDs")
print("-" * 80)

def normalize_sku(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Extract numeric portion
    number = value.replace("SKU", "")

    try:
        number = int(number)

        # Standard format: SKU + 5 digits
        return f"SKU{number:05d}"

    except ValueError:

        return np.nan


sku_datasets = [
    sales,
    sku_master,
    inventory,
    inventory_flags
]

for df in sku_datasets:

    if "sku_id" in df.columns:

        df["sku_id"] = (
            df["sku_id"]
            .apply(normalize_sku)
        )

print("✓ SKU IDs normalized")

print("\nExample:")
print("Sales SKU:", sales["sku_id"].iloc[0])
print("Master SKU:", sku_master["sku_id"].iloc[0])


# ============================================================
# 6. NORMALIZE STORE IDs
# ============================================================

print("\n[6] NORMALIZING STORE IDs")
print("-" * 80)

def normalize_store(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if value.startswith("ST"):

        number = value.replace("ST", "")

        try:
            return f"ST{int(number):02d}"
        except ValueError:
            return value

    return value


for df in [sales, inventory, store_master]:

    if "store_id" in df.columns:

        df["store_id"] = (
            df["store_id"]
            .apply(normalize_store)
        )

print("✓ Store IDs normalized")


# ============================================================
# 7. CONVERT DATE COLUMNS
# ============================================================

print("\n[7] CONVERTING DATE COLUMNS")
print("-" * 80)

date_columns = {

    "sales": ["date"],

    "calendar": ["date"],

    "inventory": ["last_restock_date"],

    "store_master": ["opening_date"],

    "promotions": [
        "start_date",
        "end_date"
    ],

    "customer_master": [
        "registration_date"
    ],

    "inventory_flags": [
        "window_start",
        "window_end"
    ]
}

for dataset_name, columns in date_columns.items():

    df = datasets[dataset_name]

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    print(f"✓ {dataset_name}")


# ============================================================
# 8. NUMERIC DATA TYPES
# ============================================================

print("\n[8] CONVERTING NUMERIC COLUMNS")
print("-" * 80)

numeric_columns = {

    "sales": [
        "units_sold",
        "revenue",
        "unit_price",
        "promo_flag"
    ],

    "sku_master": [
        "unit_price",
        "cost_price"
    ],

    "calendar": [
        "week",
        "month",
        "is_holiday",
        "promo_event"
    ],

    "inventory": [
        "stock_on_hand",
        "reorder_point",
        "safety_stock"
    ],

    "promotions": [
        "discount_pct"
    ],

    "customer_master": [
        "age"
    ]
}

for dataset_name, columns in numeric_columns.items():

    df = datasets[dataset_name]

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    print(f"✓ {dataset_name}")


# ============================================================
# 9. REMOVE EXACT DUPLICATES
# ============================================================

print("\n[9] DUPLICATE CHECK")
print("-" * 80)

for name, df in datasets.items():

    before = len(df)

    df.drop_duplicates(
        inplace=True
    )

    after = len(df)

    removed = before - after

    print(
        f"{name:20} : "
        f"{removed} duplicates removed"
    )


# ============================================================
# 10. SALES VALIDATION
# ============================================================

print("\n[10] SALES VALIDATION")
print("-" * 80)

# Units sold cannot be negative
negative_units = (
    sales["units_sold"] < 0
).sum()

# Revenue cannot be negative
negative_revenue = (
    sales["revenue"] < 0
).sum()

# Price cannot be negative
negative_price = (
    sales["unit_price"] < 0
).sum()

print(
    "Negative units sold:",
    negative_units
)

print(
    "Negative revenue:",
    negative_revenue
)

print(
    "Negative unit price:",
    negative_price
)

# We will not automatically remove them.
# They need investigation if any exist.


# ============================================================
# 11. INVENTORY VALIDATION
# ============================================================

print("\n[11] INVENTORY VALIDATION")
print("-" * 80)

for column in [
    "stock_on_hand",
    "reorder_point",
    "safety_stock"
]:

    if column in inventory.columns:

        negative_values = (
            inventory[column] < 0
        ).sum()

        print(
            f"{column:20}: "
            f"{negative_values} negative values"
        )


# ============================================================
# 12. CUSTOMER VALIDATION
# ============================================================

print("\n[12] CUSTOMER VALIDATION")
print("-" * 80)

if "age" in customer_master.columns:

    invalid_age = (
        (customer_master["age"] < 0)
        |
        (customer_master["age"] > 120)
    ).sum()

    print(
        "Invalid customer ages:",
        invalid_age
    )


# ============================================================
# 13. CALENDAR VALIDATION
# ============================================================

print("\n[13] CALENDAR VALIDATION")
print("-" * 80)

print(
    "Calendar start:",
    calendar["date"].min()
)

print(
    "Calendar end:",
    calendar["date"].max()
)

print(
    "Calendar rows:",
    len(calendar)
)

print(
    "Unique promo_event values:",
    calendar["promo_event"].nunique()
)

if calendar["promo_event"].nunique() == 1:

    print(
        "⚠ promo_event contains only one unique value."
    )


# ============================================================
# 14. INVENTORY FLAGS
# ============================================================

print("\n[14] INVENTORY FLAGS")
print("-" * 80)

print(
    "Total flags:",
    len(inventory_flags)
)

print(
    "Missing window_start:",
    inventory_flags["window_start"].isna().sum()
)

print(
    "Missing window_end:",
    inventory_flags["window_end"].isna().sum()
)

print(
    "✓ Missing flag windows retained as missing."
)


# ============================================================
# 15. SKU RELATIONSHIP VALIDATION
# ============================================================

print("\n[15] SKU RELATIONSHIP VALIDATION")
print("-" * 80)

sales_skus = set(
    sales["sku_id"]
    .dropna()
)

master_skus = set(
    sku_master["sku_id"]
    .dropna()
)

inventory_skus = set(
    inventory["sku_id"]
    .dropna()
)

flag_skus = set(
    inventory_flags["sku_id"]
    .dropna()
)

print(
    "Sales SKUs:",
    len(sales_skus)
)

print(
    "Master SKUs:",
    len(master_skus)
)

print(
    "Inventory SKUs:",
    len(inventory_skus)
)

print(
    "Flag SKUs:",
    len(flag_skus)
)

print(
    "Sales SKUs missing from master:",
    len(sales_skus - master_skus)
)

print(
    "Inventory SKUs missing from master:",
    len(inventory_skus - master_skus)
)

print(
    "Flag SKUs missing from master:",
    len(flag_skus - master_skus)
)


# ============================================================
# 16. STORE RELATIONSHIP VALIDATION
# ============================================================

print("\n[16] STORE RELATIONSHIP VALIDATION")
print("-" * 80)

sales_stores = set(
    sales["store_id"]
    .dropna()
)

master_stores = set(
    store_master["store_id"]
    .dropna()
)

inventory_stores = set(
    inventory["store_id"]
    .dropna()
)

print(
    "Sales stores:",
    len(sales_stores)
)

print(
    "Master stores:",
    len(master_stores)
)

print(
    "Inventory stores:",
    len(inventory_stores)
)

print(
    "Sales stores missing from master:",
    len(sales_stores - master_stores)
)

print(
    "Inventory stores missing from master:",
    len(inventory_stores - master_stores)
)


# ============================================================
# 17. SALES → CALENDAR VALIDATION
# ============================================================

print("\n[17] SALES → CALENDAR VALIDATION")
print("-" * 80)

sales_dates = set(
    sales["date"]
    .dropna()
)

calendar_dates = set(
    calendar["date"]
    .dropna()
)

missing_dates = (
    sales_dates - calendar_dates
)

print(
    "Sales dates:",
    len(sales_dates)
)

print(
    "Calendar dates:",
    len(calendar_dates)
)

print(
    "Sales dates missing from calendar:",
    len(missing_dates)
)


# ============================================================
# 18. SORT DATASETS
# ============================================================

print("\n[18] SORTING DATA")
print("-" * 80)

if "date" in sales.columns:

    sales.sort_values(
        ["date", "store_id", "sku_id"],
        inplace=True
    )

if "date" in calendar.columns:

    calendar.sort_values(
        "date",
        inplace=True
    )

if "last_restock_date" in inventory.columns:

    inventory.sort_values(
        ["store_id", "sku_id"],
        inplace=True
    )

print("✓ Data sorted")


# ============================================================
# 19. RESET INDEX
# ============================================================

for name, df in datasets.items():

    df.reset_index(
        drop=True,
        inplace=True
    )


# ============================================================
# 20. SAVE CLEANED DATA
# ============================================================

print("\n[19] SAVING PROCESSED DATA")
print("-" * 80)

sales.to_csv(
    PROCESSED_PATH / "sales_daily_clean.csv",
    index=False
)

sku_master.to_csv(
    PROCESSED_PATH / "sku_master_clean.csv",
    index=False
)

calendar.to_csv(
    PROCESSED_PATH / "calendar_clean.csv",
    index=False
)

inventory.to_csv(
    PROCESSED_PATH / "inventory_snapshot_clean.csv",
    index=False
)

store_master.to_csv(
    PROCESSED_PATH / "store_master_clean.csv",
    index=False
)

promotions.to_csv(
    PROCESSED_PATH / "promotions_clean.csv",
    index=False
)

customer_master.to_csv(
    PROCESSED_PATH / "customer_master_clean.csv",
    index=False
)

inventory_flags.to_csv(
    PROCESSED_PATH / "sku_inventory_flags_clean.csv",
    index=False
)

print("✓ sales_daily_clean.csv")
print("✓ sku_master_clean.csv")
print("✓ calendar_clean.csv")
print("✓ inventory_snapshot_clean.csv")
print("✓ store_master_clean.csv")
print("✓ promotions_clean.csv")
print("✓ customer_master_clean.csv")
print("✓ sku_inventory_flags_clean.csv")


# ============================================================
# 21. FINAL QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL PROCESSED DATA QUALITY CHECK")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name.upper()}")

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "Missing cells:",
        df.isnull().sum().sum()
    )

    print(
        "Duplicate rows:",
        df.duplicated().sum()
    )


# ============================================================
# 22. COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("PHASE 2 DATA CLEANING COMPLETED")
print("=" * 80)

print("""
✓ Raw datasets were preserved.
✓ SKU IDs were normalized.
✓ Store IDs were standardized.
✓ Date columns were converted.
✓ Numeric columns were validated.
✓ Duplicate rows were removed.
✓ Dataset relationships were validated.
✓ Cleaned datasets were saved to:

data/processed/

NEXT PHASE:
PHASE 3 → EXPLORATORY DATA ANALYSIS (EDA)
""")

print("=" * 80)