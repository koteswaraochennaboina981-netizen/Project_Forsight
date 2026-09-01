# ============================================================
# PROJECT FORESIGHT
# PHASE 1: DATA UNDERSTANDING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

print("=" * 80)
print("PROJECT FORESIGHT - DATA UNDERSTANDING")
print("=" * 80)

# ------------------------------------------------------------
# 1. DATA PATH
# ------------------------------------------------------------

# FIX (reproducibility): the path used to be Path("../data/raw"), which is
# resolved relative to the CURRENT WORKING DIRECTORY, not this script's
# location. That only worked if you happened to run the script from inside
# ml_part/. Anchoring the path to this file's own location means the script
# runs correctly no matter where it is invoked from (a grader's machine,
# a CI job, etc.), which is what "re-runs end-to-end with a single command"
# requires.
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "data" / "raw"

print("\n[1] DATA PATH")
print("-" * 80)
print(DATA_PATH.resolve())


# ------------------------------------------------------------
# 2. LOAD ALL DATASETS
# ------------------------------------------------------------

print("\n[2] LOADING DATASETS")
print("-" * 80)

sales = pd.read_csv(DATA_PATH / "sales_daily.csv")
sku_master = pd.read_csv(DATA_PATH / "sku_master.csv")
calendar = pd.read_csv(DATA_PATH / "calendar.csv")
inventory = pd.read_csv(DATA_PATH / "inventory_snapshot.csv")
store_master = pd.read_csv(DATA_PATH / "store_master.csv")
promotions = pd.read_csv(DATA_PATH / "promotions.csv")
customer_master = pd.read_csv(DATA_PATH / "customer_master.csv")
inventory_flags = pd.read_csv(DATA_PATH / "sku_inventory_flags.csv")

print("✓ All 8 datasets loaded successfully!")


# ------------------------------------------------------------
# 3. STORE DATASETS IN DICTIONARY
# ------------------------------------------------------------

datasets = {
    "sales_daily": sales,
    "sku_master": sku_master,
    "calendar": calendar,
    "inventory": inventory,
    "store_master": store_master,
    "promotions": promotions,
    "customer_master": customer_master,
    "inventory_flags": inventory_flags
}

print("\nDatasets loaded:")
for name in datasets:
    print("  ✓", name)


# ------------------------------------------------------------
# 4. DATASET SIZE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[3] DATASET SIZE")
print("=" * 80)

for name, df in datasets.items():
    rows, columns = df.shape
    print(
        f"{name:20} : "
        f"{rows:>8,} rows × {columns:>3} columns"
    )


# ------------------------------------------------------------
# 5. COLUMN NAMES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[4] COLUMN NAMES")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name.upper()}")
    print("-" * 60)

    for column in df.columns:
        print("  ", column)


# ------------------------------------------------------------
# 6. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[5] FIRST 5 ROWS")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name.upper()}")
    print("-" * 60)

    display(df.head())


# ------------------------------------------------------------
# 7. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[6] DATA TYPES")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name.upper()}")
    print("-" * 60)

    print(df.dtypes)


# ------------------------------------------------------------
# 8. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[7] MISSING VALUES")
print("=" * 80)

for name, df in datasets.items():

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print(f"\n{name.upper()}")

    if len(missing) == 0:
        print("  ✓ No missing values")
    else:
        print(missing)


# ------------------------------------------------------------
# 9. MISSING VALUE PERCENTAGE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[8] MISSING VALUE PERCENTAGE")
print("=" * 80)

for name, df in datasets.items():

    missing_count = df.isnull().sum()

    missing_percent = (
        missing_count / len(df)
    ) * 100

    result = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent.round(2)
    })

    result = result[
        result["missing_count"] > 0
    ]

    print(f"\n{name.upper()}")

    if result.empty:
        print("  ✓ No missing values")
    else:
        display(
            result.sort_values(
                "missing_percent",
                ascending=False
            )
        )


# ------------------------------------------------------------
# 10. DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[9] DUPLICATE ROWS")
print("=" * 80)

for name, df in datasets.items():

    duplicates = df.duplicated().sum()

    print(
        f"{name:20} : "
        f"{duplicates:,} duplicate rows"
    )


# ------------------------------------------------------------
# 11. DATE-LIKE COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[10] DATE-LIKE COLUMNS")
print("=" * 80)

for name, df in datasets.items():

    date_columns = []

    for column in df.columns:

        if (
            "date" in column.lower()
            or "time" in column.lower()
        ):
            date_columns.append(column)

    print(f"\n{name}:")
    print("  ", date_columns)


# ------------------------------------------------------------
# 12. DATE RANGES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[11] DATE RANGES")
print("=" * 80)

for name, df in datasets.items():

    for column in df.columns:

        if "date" in column.lower():

            temp = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            valid_dates = temp.dropna()

            if len(valid_dates) > 0:

                print(
                    f"{name:20} | "
                    f"{column:20} | "
                    f"{valid_dates.min().date()} → "
                    f"{valid_dates.max().date()}"
                )


# ------------------------------------------------------------
# 13. UNIQUE VALUES
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[12] UNIQUE VALUES")
print("=" * 80)

for name, df in datasets.items():

    print(f"\n{name.upper()}")
    print("-" * 60)

    for column in df.columns:

        print(
            f"{column:30} : "
            f"{df[column].nunique():,} unique values"
        )


# ------------------------------------------------------------
# 14. SALES DATA ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[13] SALES DATA ANALYSIS")
print("=" * 80)

print("Rows:", len(sales))
print("Columns:", len(sales.columns))

print("\nColumns:")
print(list(sales.columns))

if "units_sold" in sales.columns:

    zero_sales = (
        sales["units_sold"] == 0
    ).sum()

    print("\nZero-demand analysis:")
    print("Total records:", len(sales))
    print("Zero-demand records:", zero_sales)

    print(
        "Zero-demand percentage:",
        round(
            zero_sales / len(sales) * 100,
            2
        ),
        "%"
    )


# ------------------------------------------------------------
# 15. SALES NUMERICAL STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[14] SALES NUMERICAL STATISTICS")
print("=" * 80)

numeric_columns = sales.select_dtypes(
    include=np.number
).columns

display(
    sales[numeric_columns].describe().T
)


# ------------------------------------------------------------
# 16. SALES SKU / STORE / DATE COUNTS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[15] SALES DIMENSION COUNTS")
print("=" * 80)

if "sku_id" in sales.columns:

    print(
        "Unique SKUs:",
        sales["sku_id"].nunique()
    )

if "store_id" in sales.columns:

    print(
        "Unique stores:",
        sales["store_id"].nunique()
    )

if "date" in sales.columns:

    print(
        "Unique dates:",
        pd.to_datetime(
            sales["date"],
            errors="coerce"
        ).nunique()
    )


# ------------------------------------------------------------
# 17. SKU MASTER
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[16] SKU MASTER")
print("=" * 80)

print("Rows:", len(sku_master))
print("Columns:", len(sku_master.columns))
print("Columns:", list(sku_master.columns))

display(sku_master.head())


# ------------------------------------------------------------
# 18. STORE MASTER
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[17] STORE MASTER")
print("=" * 80)

print("Rows:", len(store_master))
print("Columns:", len(store_master.columns))
print("Columns:", list(store_master.columns))

display(store_master.head())


# ------------------------------------------------------------
# 19. INVENTORY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[18] INVENTORY")
print("=" * 80)

print("Rows:", len(inventory))
print("Columns:", len(inventory.columns))
print("Columns:", list(inventory.columns))

display(inventory.head())


# ------------------------------------------------------------
# 20. PROMOTIONS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[19] PROMOTIONS")
print("=" * 80)

print("Rows:", len(promotions))
print("Columns:", len(promotions.columns))
print("Columns:", list(promotions.columns))

display(promotions.head())


# ------------------------------------------------------------
# 21. CUSTOMER MASTER
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[20] CUSTOMER MASTER")
print("=" * 80)

print("Rows:", len(customer_master))
print("Columns:", len(customer_master.columns))
print("Columns:", list(customer_master.columns))

display(customer_master.head())


# ------------------------------------------------------------
# 22. INVENTORY FLAGS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[21] INVENTORY FLAGS")
print("=" * 80)

print("Rows:", len(inventory_flags))
print("Columns:", len(inventory_flags.columns))
print("Columns:", list(inventory_flags.columns))

display(inventory_flags.head())


# ------------------------------------------------------------
# 23. FIND ID COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[22] POSSIBLE ID COLUMNS")
print("=" * 80)

for name, df in datasets.items():

    id_columns = [
        column
        for column in df.columns
        if (
            "id" in column.lower()
            or "code" in column.lower()
        )
    ]

    print(f"\n{name}:")
    print("  ", id_columns)


# ------------------------------------------------------------
# 24. SKU COMPATIBILITY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[23] SKU COMPATIBILITY")
print("=" * 80)

if (
    "sku_id" in sales.columns
    and "sku_id" in sku_master.columns
):

    sales_skus = set(
        sales["sku_id"]
        .dropna()
        .astype(str)
    )

    master_skus = set(
        sku_master["sku_id"]
        .dropna()
        .astype(str)
    )

    missing_skus = (
        sales_skus - master_skus
    )

    print(
        "Unique SKUs in sales:",
        len(sales_skus)
    )

    print(
        "Unique SKUs in SKU master:",
        len(master_skus)
    )

    print(
        "Sales SKUs missing from master:",
        len(missing_skus)
    )

    if missing_skus:

        print("\nExamples:")
        print(
            list(missing_skus)[:20]
        )

else:

    print(
        "⚠ sku_id not found in both datasets."
    )


# ------------------------------------------------------------
# 25. STORE COMPATIBILITY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[24] STORE COMPATIBILITY")
print("=" * 80)

if (
    "store_id" in sales.columns
    and "store_id" in store_master.columns
):

    sales_stores = set(
        sales["store_id"]
        .dropna()
        .astype(str)
    )

    master_stores = set(
        store_master["store_id"]
        .dropna()
        .astype(str)
    )

    missing_stores = (
        sales_stores - master_stores
    )

    print(
        "Unique stores in sales:",
        len(sales_stores)
    )

    print(
        "Unique stores in store master:",
        len(master_stores)
    )

    print(
        "Sales stores missing from master:",
        len(missing_stores)
    )

    if missing_stores:

        print("\nExamples:")
        print(
            list(missing_stores)[:20]
        )

else:

    print(
        "⚠ store_id not found in both datasets."
    )


# ------------------------------------------------------------
# 26. SALES → CALENDAR COMPATIBILITY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[25] SALES → CALENDAR COMPATIBILITY")
print("=" * 80)

if (
    "date" in sales.columns
    and "date" in calendar.columns
):

    sales_dates = set(
        pd.to_datetime(
            sales["date"],
            errors="coerce"
        )
        .dropna()
        .dt.date
    )

    calendar_dates = set(
        pd.to_datetime(
            calendar["date"],
            errors="coerce"
        )
        .dropna()
        .dt.date
    )

    missing_calendar_dates = (
        sales_dates - calendar_dates
    )

    print(
        "Unique sales dates:",
        len(sales_dates)
    )

    print(
        "Calendar dates:",
        len(calendar_dates)
    )

    print(
        "Sales dates missing from calendar:",
        len(missing_calendar_dates)
    )

    if missing_calendar_dates:

        print("\nExamples:")
        print(
            list(missing_calendar_dates)[:20]
        )

else:

    print(
        "⚠ date column not found in both datasets."
    )


# ------------------------------------------------------------
# 27. CALENDAR VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[26] CALENDAR VALIDATION")
print("=" * 80)

required_calendar_columns = [
    "date",
    "week",
    "month",
    "season",
    "is_holiday",
    "promo_event"
]

for column in required_calendar_columns:

    if column in calendar.columns:

        print(f"✓ {column}")

    else:

        print(f"✗ {column} MISSING")


# ------------------------------------------------------------
# 28. SALES REQUIRED COLUMN CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[27] SALES REQUIRED COLUMN CHECK")
print("=" * 80)

required_sales_columns = [
    "date",
    "store_id",
    "sku_id",
    "units_sold",
    "revenue",
    "unit_price",
    "promo_flag"
]

for column in required_sales_columns:

    if column in sales.columns:

        print(f"✓ {column}")

    else:

        print(f"✗ {column} MISSING")


# ------------------------------------------------------------
# 29. DATASET SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[28] FINAL DATASET SUMMARY")
print("=" * 80)

summary = []

for name, df in datasets.items():

    summary.append({

        "dataset": name,

        "rows": len(df),

        "columns": len(df.columns),

        "duplicate_rows":
            df.duplicated().sum(),

        "missing_cells":
            df.isnull().sum().sum()
    })

summary_df = pd.DataFrame(summary)

display(summary_df)


# ------------------------------------------------------------
# 30. MEMORY USAGE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[29] MEMORY USAGE")
print("=" * 80)

for name, df in datasets.items():

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / (1024 ** 2)
    )

    print(
        f"{name:20} : "
        f"{memory_mb:.2f} MB"
    )


# ------------------------------------------------------------
# 31. DATASET COMPLETENESS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[30] DATASET COMPLETENESS")
print("=" * 80)

for name, df in datasets.items():

    total_cells = df.size

    missing_cells = (
        df.isnull().sum().sum()
    )

    completeness = (
        (total_cells - missing_cells)
        / total_cells
    ) * 100

    print(
        f"{name:20} : "
        f"{completeness:.2f}% complete"
    )


# ------------------------------------------------------------
# 32. RAW DATA CHECK
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("[31] RAW DATA FILE CHECK")
print("=" * 80)

required_files = [
    "sales_daily.csv",
    "sku_master.csv",
    "calendar.csv",
    "inventory_snapshot.csv",
    "store_master.csv",
    "promotions.csv",
    "customer_master.csv",
    "sku_inventory_flags.csv"
]

for filename in required_files:

    file_path = DATA_PATH / filename

    if file_path.exists():

        print(f"✓ {filename}")

    else:

        print(f"✗ {filename} MISSING")


# ------------------------------------------------------------
# 33. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("PHASE 1 DATA UNDERSTANDING CHECK COMPLETED")
print("=" * 80)

print("""
Next step:
PHASE 2 → DATA CLEANING & DATA ENGINEERING

IMPORTANT:
No raw CSV files were modified by this analysis.
""")

# ============================================================
# SKU ID INVESTIGATION
# Project FORESIGHT
# ============================================================

print("=" * 80)
print("SKU ID FORMAT INVESTIGATION")
print("=" * 80)

# ------------------------------------------------------------
# 1. SHOW SAMPLE SKU IDs
# ------------------------------------------------------------

print("\n[1] SAMPLE SKU IDs FROM SALES")
print("-" * 80)

print(
    sales["sku_id"]
    .drop_duplicates()
    .head(20)
    .tolist()
)

print("\n[2] SAMPLE SKU IDs FROM SKU MASTER")
print("-" * 80)

print(
    sku_master["sku_id"]
    .drop_duplicates()
    .head(20)
    .tolist()
)


# ------------------------------------------------------------
# 2. CHECK SKU ID LENGTH
# ------------------------------------------------------------

print("\n[3] SKU ID LENGTH - SALES")
print("-" * 80)

print(
    sales["sku_id"]
    .astype(str)
    .str.len()
    .value_counts()
    .sort_index()
)

print("\n[4] SKU ID LENGTH - SKU MASTER")
print("-" * 80)

print(
    sku_master["sku_id"]
    .astype(str)
    .str.len()
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 3. NORMALIZE SKU IDs
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("NORMALIZATION TEST")
print("=" * 80)

sales_test = sales.copy()
master_test = sku_master.copy()

# Extract numeric portion of SKU ID
sales_test["sku_number"] = (
    sales_test["sku_id"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

master_test["sku_number"] = (
    master_test["sku_id"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

# Convert numeric portion to integer
sales_test["sku_number"] = pd.to_numeric(
    sales_test["sku_number"],
    errors="coerce"
)

master_test["sku_number"] = pd.to_numeric(
    master_test["sku_number"],
    errors="coerce"
)


# ------------------------------------------------------------
# 4. COMPARE NORMALIZED SKU IDs
# ------------------------------------------------------------

sales_skus = set(
    sales_test["sku_number"]
    .dropna()
    .astype(int)
)

master_skus = set(
    master_test["sku_number"]
    .dropna()
    .astype(int)
)

missing_after_normalization = (
    sales_skus - master_skus
)

matched_after_normalization = (
    sales_skus & master_skus
)


print("\nUnique SKUs in SALES:",
      len(sales_skus))

print("Unique SKUs in SKU MASTER:",
      len(master_skus))

print("SKUs matched after normalization:",
      len(matched_after_normalization))

print("SKUs still missing after normalization:",
      len(missing_after_normalization))


# ------------------------------------------------------------
# 5. SHOW UNMATCHED SKUs
# ------------------------------------------------------------

if len(missing_after_normalization) > 0:

    print("\n[5] UNMATCHED SKU NUMBERS")
    print("-" * 80)

    print(
        sorted(
            missing_after_normalization
        )[:50]
    )

else:

    print("\n✓ ALL SALES SKUs MATCH SKU MASTER AFTER NORMALIZATION")


# ------------------------------------------------------------
# 6. CHECK ORIGINAL SKU MATCHING
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ORIGINAL SKU MATCHING")
print("=" * 80)

original_sales_skus = set(
    sales["sku_id"]
    .dropna()
    .astype(str)
)

original_master_skus = set(
    sku_master["sku_id"]
    .dropna()
    .astype(str)
)

original_matches = (
    original_sales_skus &
    original_master_skus
)

print(
    "Original exact matches:",
    len(original_matches)
)

print(
    "Original sales SKUs:",
    len(original_sales_skus)
)

print(
    "Original master SKUs:",
    len(original_master_skus)
)


# ------------------------------------------------------------
# 7. FINAL DIAGNOSIS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SKU INVESTIGATION RESULT")
print("=" * 80)

if len(missing_after_normalization) == 0:

    print("""
✓ SKU IDs are compatible after normalization.

The apparent mismatch is caused by SKU ID formatting.
We can safely handle this during Phase 2 Data Cleaning.
""")

elif len(matched_after_normalization) > 0:

    print("""
⚠ PARTIAL SKU MATCH FOUND.

Some sales SKUs exist in the SKU master after normalization,
but some genuinely unmatched SKUs remain.

We need to investigate those unmatched SKUs before cleaning.
""")

else:

    print("""
❌ SKU IDs still do not match after normalization.

The sales and SKU master datasets may use different SKU systems.
We need to investigate the dataset structure before Phase 2.
""")

print("=" * 80)