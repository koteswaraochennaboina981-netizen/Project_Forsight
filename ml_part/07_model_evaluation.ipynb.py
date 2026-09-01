# ============================================================
# PROJECT FORESIGHT - PHASE 7: MODEL EVALUATION
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ============================================================
# [1] PATHS
# ============================================================

print("\n" + "=" * 80)
print("PROJECT FORESIGHT - PHASE 7: MODEL EVALUATION")
print("=" * 80)

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

INVENTORY_PATH = os.path.join(
    DATA_PATH,
    "inventory_optimization"
)

EVALUATION_PATH = os.path.join(
    DATA_PATH,
    "model_evaluation"
)

REPORT_PATH = os.path.join(
    BASE_PATH,
    "reports",
    "model_evaluation"
)

os.makedirs(
    EVALUATION_PATH,
    exist_ok=True
)

os.makedirs(
    REPORT_PATH,
    exist_ok=True
)

print("\n[1] PATHS")
print("-" * 80)

print(f"Project root       : {BASE_PATH}")
print(f"Forecast input     : {FORECAST_PATH}")
print(f"Inventory input    : {INVENTORY_PATH}")
print(f"Evaluation output  : {EVALUATION_PATH}")
print(f"Reports            : {REPORT_PATH}")


# ============================================================
# [2] CHECKING INPUT FILES
# ============================================================

print("\n[2] CHECKING INPUT FILES")
print("-" * 80)

forecast_file = os.path.join(
    FORECAST_PATH,
    "demand_forecasts.csv"
)

forecast_metrics_file = os.path.join(
    FORECAST_PATH,
    "forecast_metrics.csv"
)

model_comparison_file = os.path.join(
    FORECAST_PATH,
    "model_comparison.csv"
)

forecast_summary_file = os.path.join(
    FORECAST_PATH,
    "forecast_summary.csv"
)

inventory_recommendations_file = os.path.join(
    INVENTORY_PATH,
    "inventory_recommendations.csv"
)

inventory_risk_file = os.path.join(
    INVENTORY_PATH,
    "inventory_risk_analysis.csv"
)

inventory_metrics_file = os.path.join(
    INVENTORY_PATH,
    "inventory_metrics.csv"
)

inventory_summary_file = os.path.join(
    INVENTORY_PATH,
    "inventory_summary.csv"
)

required_files = {
    "demand_forecasts.csv": forecast_file,
    "forecast_metrics.csv": forecast_metrics_file,
    "model_comparison.csv": model_comparison_file,
    "forecast_summary.csv": forecast_summary_file,
    "inventory_recommendations.csv": inventory_recommendations_file,
    "inventory_risk_analysis.csv": inventory_risk_file,
    "inventory_metrics.csv": inventory_metrics_file,
    "inventory_summary.csv": inventory_summary_file
}

for file_name, file_path in required_files.items():

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{file_name} not found:\n{file_path}"
        )

    print(f"✓ {file_name} found")


# ============================================================
# [3] LOADING PHASE 5 OUTPUTS
# ============================================================

print("\n[3] LOADING FORECASTING OUTPUTS")
print("-" * 80)

forecast_df = pd.read_csv(
    forecast_file
)

forecast_metrics_df = pd.read_csv(
    forecast_metrics_file
)

model_comparison_df = pd.read_csv(
    model_comparison_file
)

forecast_summary_df = pd.read_csv(
    forecast_summary_file
)

print("✓ Demand forecasts loaded")
print("✓ Forecast metrics loaded")
print("✓ Model comparison loaded")
print("✓ Forecast summary loaded")

print()
print(f"Forecast rows       : {len(forecast_df):,}")
print(f"Forecast columns    : {len(forecast_df.columns):,}")


# ============================================================
# [4] LOADING PHASE 6 OUTPUTS
# ============================================================

print("\n[4] LOADING INVENTORY OPTIMIZATION OUTPUTS")
print("-" * 80)

inventory_recommendations_df = pd.read_csv(
    inventory_recommendations_file
)

inventory_risk_df = pd.read_csv(
    inventory_risk_file
)

inventory_metrics_df = pd.read_csv(
    inventory_metrics_file
)

inventory_summary_df = pd.read_csv(
    inventory_summary_file
)

print("✓ Inventory recommendations loaded")
print("✓ Inventory risk analysis loaded")
print("✓ Inventory metrics loaded")
print("✓ Inventory summary loaded")

print()
print(
    f"Inventory recommendation rows : "
    f"{len(inventory_recommendations_df):,}"
)


# ============================================================
# [5] VALIDATING FORECAST DATA
# ============================================================

print("\n[5] VALIDATING FORECAST DATA")
print("-" * 80)

required_forecast_columns = [
    "date",
    "sku_id",
    "store_id",
    "actual_units_sold",
    "predicted_units_sold",
    "forecast_error",
    "absolute_error"
]

missing_forecast_columns = [
    column
    for column in required_forecast_columns
    if column not in forecast_df.columns
]

if missing_forecast_columns:

    raise ValueError(
        "Missing forecast columns: "
        + ", ".join(missing_forecast_columns)
    )

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"],
    errors="coerce"
)

if forecast_df["date"].isna().any():

    raise ValueError(
        "Invalid dates found in forecast data."
    )

numeric_forecast_columns = [
    "actual_units_sold",
    "predicted_units_sold",
    "forecast_error",
    "absolute_error"
]

for column in numeric_forecast_columns:

    forecast_df[column] = pd.to_numeric(
        forecast_df[column],
        errors="coerce"
    )

print("✓ Forecast columns validated")
print("✓ Forecast dates validated")
print("✓ Forecast numeric values validated")


# ============================================================
# [6] FORECAST DATA QUALITY VALIDATION
# ============================================================

print("\n[6] FORECAST DATA QUALITY VALIDATION")
print("-" * 80)

missing_forecast_values = (
    forecast_df[
        required_forecast_columns
    ]
    .isna()
    .sum()
    .sum()
)

negative_predictions = (
    forecast_df["predicted_units_sold"] < 0
).sum()

duplicate_forecast_rows = (
    forecast_df
    .duplicated(
        subset=[
            "date",
            "sku_id",
            "store_id"
        ]
    )
    .sum()
)

print(
    f"Missing forecast values : "
    f"{missing_forecast_values:,}"
)

print(
    f"Negative predictions    : "
    f"{negative_predictions:,}"
)

print(
    f"Duplicate forecast rows : "
    f"{duplicate_forecast_rows:,}"
)

if missing_forecast_values == 0:

    print("✓ No missing forecast values")

if negative_predictions == 0:

    print("✓ No negative predictions")

if duplicate_forecast_rows == 0:

    print("✓ No duplicate forecast records")


# ============================================================
# [7] RE-CALCULATING FORECAST ERRORS
# ============================================================

print("\n[7] CALCULATING FORECAST ERRORS")
print("-" * 80)

forecast_df["calculated_error"] = (
    forecast_df["actual_units_sold"]
    -
    forecast_df["predicted_units_sold"]
)

forecast_df["calculated_absolute_error"] = (
    np.abs(
        forecast_df["calculated_error"]
    )
)

forecast_df["squared_error"] = (
    forecast_df["calculated_error"] ** 2
)

print("✓ Forecast error calculated")
print("✓ Absolute error calculated")
print("✓ Squared error calculated")


# ============================================================
# [8] DEFINING EVALUATION METRICS
# ============================================================

print("\n[8] DEFINING EVALUATION METRICS")
print("-" * 80)


def calculate_smape(
    actual_values,
    predicted_values
):

    actual_values = np.asarray(
        actual_values,
        dtype=float
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=float
    )

    denominator = (
        np.abs(actual_values)
        +
        np.abs(predicted_values)
    )

    mask = denominator != 0

    if not np.any(mask):

        return 0.0

    return (
        100
        *
        np.mean(
            2
            *
            np.abs(
                predicted_values[mask]
                -
                actual_values[mask]
            )
            /
            denominator[mask]
        )
    )


def calculate_wape(
    actual_values,
    predicted_values
):

    actual_values = np.asarray(
        actual_values,
        dtype=float
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=float
    )

    denominator = np.sum(
        np.abs(actual_values)
    )

    if denominator == 0:

        return 0.0

    return (
        100
        *
        np.sum(
            np.abs(
                actual_values
                -
                predicted_values
            )
        )
        /
        denominator
    )


def evaluate_forecast(
    actual_values,
    predicted_values
):

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    smape = calculate_smape(
        actual_values,
        predicted_values
    )

    wape = calculate_wape(
        actual_values,
        predicted_values
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE_percent": smape,
        "WAPE_percent": wape
    }


print("✓ MAE")
print("✓ RMSE")
print("✓ sMAPE")
print("✓ WAPE")


# ============================================================
# [9] OVERALL FORECAST EVALUATION
# ============================================================

print("\n[9] OVERALL FORECAST EVALUATION")
print("-" * 80)

overall_metrics = evaluate_forecast(
    forecast_df["actual_units_sold"],
    forecast_df["predicted_units_sold"]
)

overall_forecast_evaluation = pd.DataFrame(
    [
        {
            "evaluation_scope": "Overall Test Forecast",
            "forecast_rows": len(forecast_df),
            "unique_skus": forecast_df["sku_id"].nunique(),
            "unique_stores": forecast_df["store_id"].nunique(),
            "start_date": forecast_df["date"].min(),
            "end_date": forecast_df["date"].max(),
            "MAE": overall_metrics["MAE"],
            "RMSE": overall_metrics["RMSE"],
            "sMAPE_percent": overall_metrics["sMAPE_percent"],
            "WAPE_percent": overall_metrics["WAPE_percent"]
        }
    ]
)

print(
    f"MAE   : "
    f"{overall_metrics['MAE']:.4f}"
)

print(
    f"RMSE  : "
    f"{overall_metrics['RMSE']:.4f}"
)

print(
    f"sMAPE : "
    f"{overall_metrics['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE  : "
    f"{overall_metrics['WAPE_percent']:.2f}%"
)


# ============================================================
# [10] SKU-LEVEL FORECAST EVALUATION
# ============================================================

print("\n[10] SKU-LEVEL FORECAST EVALUATION")
print("-" * 80)

sku_evaluation_rows = []

for sku_id, group in forecast_df.groupby("sku_id"):

    actual_values = group[
        "actual_units_sold"
    ]

    predicted_values = group[
        "predicted_units_sold"
    ]

    metrics = evaluate_forecast(
        actual_values,
        predicted_values
    )

    sku_evaluation_rows.append(
        {
            "sku_id": sku_id,
            "forecast_rows": len(group),
            "actual_demand": actual_values.sum(),
            "predicted_demand": predicted_values.sum(),
            "absolute_error": np.abs(
                actual_values - predicted_values
            ).sum(),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "sMAPE_percent": metrics["sMAPE_percent"],
            "WAPE_percent": metrics["WAPE_percent"]
        }
    )

sku_forecast_evaluation = pd.DataFrame(
    sku_evaluation_rows
)

sku_forecast_evaluation = (
    sku_forecast_evaluation
    .sort_values(
        "MAE",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    f"SKU evaluations : "
    f"{len(sku_forecast_evaluation):,}"
)

print("✓ SKU-level evaluation created")


# ============================================================
# [11] STORE-LEVEL FORECAST EVALUATION
# ============================================================

print("\n[11] STORE-LEVEL FORECAST EVALUATION")
print("-" * 80)

store_evaluation_rows = []

for store_id, group in forecast_df.groupby("store_id"):

    actual_values = group[
        "actual_units_sold"
    ]

    predicted_values = group[
        "predicted_units_sold"
    ]

    metrics = evaluate_forecast(
        actual_values,
        predicted_values
    )

    store_evaluation_rows.append(
        {
            "store_id": store_id,
            "forecast_rows": len(group),
            "actual_demand": actual_values.sum(),
            "predicted_demand": predicted_values.sum(),
            "absolute_error": np.abs(
                actual_values - predicted_values
            ).sum(),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "sMAPE_percent": metrics["sMAPE_percent"],
            "WAPE_percent": metrics["WAPE_percent"]
        }
    )

store_forecast_evaluation = pd.DataFrame(
    store_evaluation_rows
)

store_forecast_evaluation = (
    store_forecast_evaluation
    .sort_values(
        "MAE",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    f"Store evaluations : "
    f"{len(store_forecast_evaluation):,}"
)

print("✓ Store-level evaluation created")


# ============================================================
# [12] FORECAST ERROR ANALYSIS
# ============================================================

print("\n[12] FORECAST ERROR ANALYSIS")
print("-" * 80)

forecast_error_analysis = forecast_df[
    [
        "date",
        "sku_id",
        "store_id",
        "actual_units_sold",
        "predicted_units_sold",
        "forecast_error",
        "absolute_error"
    ]
].copy()

forecast_error_analysis[
    "error_percentage"
] = np.where(
    forecast_error_analysis[
        "actual_units_sold"
    ] != 0,

    100
    *
    forecast_error_analysis[
        "forecast_error"
    ]
    /
    forecast_error_analysis[
        "actual_units_sold"
    ],

    np.nan
)

forecast_error_analysis[
    "absolute_error_percentage"
] = np.where(
    forecast_error_analysis[
        "actual_units_sold"
    ] != 0,

    100
    *
    forecast_error_analysis[
        "absolute_error"
    ]
    /
    forecast_error_analysis[
        "actual_units_sold"
    ],

    np.nan
)

forecast_error_analysis[
    "error_direction"
] = np.where(
    forecast_error_analysis[
        "forecast_error"
    ] > 0,

    "UNDER_FORECAST",

    np.where(
        forecast_error_analysis[
            "forecast_error"
        ] < 0,

        "OVER_FORECAST",

        "EXACT"
    )
)

forecast_error_analysis = (
    forecast_error_analysis
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    f"Error records : "
    f"{len(forecast_error_analysis):,}"
)

print("✓ Forecast error analysis created")


# ============================================================
# [13] FORECAST ERROR SUMMARY
# ============================================================

print("\n[13] FORECAST ERROR SUMMARY")
print("-" * 80)

under_forecast_count = (
    forecast_error_analysis[
        "error_direction"
    ]
    .eq("UNDER_FORECAST")
    .sum()
)

over_forecast_count = (
    forecast_error_analysis[
        "error_direction"
    ]
    .eq("OVER_FORECAST")
    .sum()
)

exact_forecast_count = (
    forecast_error_analysis[
        "error_direction"
    ]
    .eq("EXACT")
    .sum()
)

error_summary = pd.DataFrame(
    [
        {
            "error_category": "UNDER_FORECAST",
            "record_count": under_forecast_count
        },
        {
            "error_category": "OVER_FORECAST",
            "record_count": over_forecast_count
        },
        {
            "error_category": "EXACT",
            "record_count": exact_forecast_count
        }
    ]
)

print(
    f"Under-forecast records : "
    f"{under_forecast_count:,}"
)

print(
    f"Over-forecast records  : "
    f"{over_forecast_count:,}"
)

print(
    f"Exact forecast records : "
    f"{exact_forecast_count:,}"
)


# ============================================================
# [14] VALIDATING INVENTORY DATA
# ============================================================

print("\n[14] VALIDATING INVENTORY DATA")
print("-" * 80)

required_inventory_columns = [
    "sku_id",
    "store_id",
    "inventory_status",
    "replenishment_priority"
]

missing_inventory_columns = [
    column
    for column in required_inventory_columns
    if column not in inventory_recommendations_df.columns
]

if missing_inventory_columns:

    raise ValueError(
        "Missing inventory columns: "
        + ", ".join(missing_inventory_columns)
    )

print("✓ Inventory recommendation columns validated")


# ============================================================
# [15] INVENTORY STATUS EVALUATION
# ============================================================

print("\n[15] INVENTORY STATUS EVALUATION")
print("-" * 80)

inventory_status_evaluation = (
    inventory_recommendations_df[
        "inventory_status"
    ]
    .value_counts()
    .rename_axis("inventory_status")
    .reset_index(
        name="count"
    )
)

inventory_status_evaluation[
    "percentage"
] = (
    100
    *
    inventory_status_evaluation["count"]
    /
    len(inventory_recommendations_df)
)

print(
    inventory_status_evaluation.to_string(
        index=False
    )
)

print("✓ Inventory status evaluation created")


# ============================================================
# [16] REPLENISHMENT PRIORITY EVALUATION
# ============================================================

print("\n[16] REPLENISHMENT PRIORITY EVALUATION")
print("-" * 80)

priority_evaluation = (
    inventory_recommendations_df[
        "replenishment_priority"
    ]
    .value_counts()
    .rename_axis(
        "replenishment_priority"
    )
    .reset_index(
        name="count"
    )
)

priority_evaluation[
    "percentage"
] = (
    100
    *
    priority_evaluation["count"]
    /
    len(inventory_recommendations_df)
)

print(
    priority_evaluation.to_string(
        index=False
    )
)

print("✓ Replenishment priority evaluation created")


# ============================================================
# [17] INVENTORY RISK EVALUATION
# ============================================================

print("\n[17] INVENTORY RISK EVALUATION")
print("-" * 80)

inventory_risk_evaluation = (
    inventory_risk_df.copy()
)

print(
    f"Risk analysis rows : "
    f"{len(inventory_risk_evaluation):,}"
)

print("✓ Inventory risk data loaded")


# ============================================================
# [18] REPLENISHMENT EVALUATION
# ============================================================

print("\n[18] REPLENISHMENT EVALUATION")
print("-" * 80)

replenishment_columns = [
    "recommended_order_quantity",
    "estimated_replenishment_cost"
]

available_replenishment_columns = [
    column
    for column in replenishment_columns
    if column in inventory_recommendations_df.columns
]

replenishment_evaluation = pd.DataFrame(
    [
        {
            "metric": "total_recommended_order_quantity",
            "value":
                inventory_recommendations_df[
                    "recommended_order_quantity"
                ].sum()
                if "recommended_order_quantity"
                in inventory_recommendations_df.columns
                else np.nan
        },
        {
            "metric": "average_recommended_order_quantity",
            "value":
                inventory_recommendations_df[
                    "recommended_order_quantity"
                ].mean()
                if "recommended_order_quantity"
                in inventory_recommendations_df.columns
                else np.nan
        },
        {
            "metric": "total_estimated_replenishment_cost",
            "value":
                inventory_recommendations_df[
                    "estimated_replenishment_cost"
                ].sum()
                if "estimated_replenishment_cost"
                in inventory_recommendations_df.columns
                else np.nan
        }
    ]
)

print(
    replenishment_evaluation.to_string(
        index=False
    )
)

print("✓ Replenishment evaluation created")


# ============================================================
# [19] CREATING COMBINED INVENTORY EVALUATION
# ============================================================

print("\n[19] CREATING COMBINED INVENTORY EVALUATION")
print("-" * 80)

combined_inventory_evaluation = pd.DataFrame({
    "evaluation_metric": [
        "SKU-store combinations",
        "Total stock on hand",
        "Total forecast demand",
        "Recommended order quantity",
        "Stockout-risk combinations"
    ],
    "value": [
        len(inventory_recommendations_df),
        inventory_recommendations_df["stock_on_hand"].sum(),
        inventory_recommendations_df["forecast_total_demand"].sum(),
        inventory_recommendations_df["recommended_order_quantity"].sum(),
        (
            inventory_recommendations_df["stockout_risk_flag"]
            .sum()
        )
    ]
})

print(
    combined_inventory_evaluation.to_string(index=False)
)

print("✓ Combined inventory evaluation created")

# ============================================================
# [20] HIGH-RISK INVENTORY COMBINATIONS
# ============================================================

print("\n[20] IDENTIFYING HIGH-RISK INVENTORY COMBINATIONS")
print("-" * 80)

high_risk_columns = [
    "sku_id",
    "store_id",
    "inventory_status",
    "replenishment_priority"
]

optional_risk_columns = [
    "stock_on_hand",
    "forecast_total_demand",
    "recommended_order_quantity",
    "stockout_risk_flag",
    "overstock_risk_flag",
    "estimated_replenishment_cost"
]

for column in optional_risk_columns:

    if column in inventory_recommendations_df.columns:

        high_risk_columns.append(
            column
        )

high_risk_inventory = (
    inventory_recommendations_df[
        inventory_recommendations_df[
            "replenishment_priority"
        ].isin(
            [
                "URGENT",
                "HIGH"
            ]
        )
    ][
        high_risk_columns
    ]
    .copy()
)

high_risk_inventory = (
    high_risk_inventory
    .sort_values(
        "replenishment_priority"
    )
    .reset_index(drop=True)
)

print(
    f"High-risk combinations : "
    f"{len(high_risk_inventory):,}"
)

print("✓ High-risk inventory list created")


# ============================================================
# [21] MODEL PERFORMANCE SUMMARY
# ============================================================

print("\n[21] CREATING MODEL PERFORMANCE SUMMARY")
print("-" * 80)

best_model = "Unknown"

if (
    "model"
    in model_comparison_df.columns
    and
    "MAE"
    in model_comparison_df.columns
):

    sorted_model_comparison = (
        model_comparison_df
        .sort_values(
            "MAE"
        )
        .reset_index(drop=True)
    )

    if len(sorted_model_comparison) > 0:

        best_model = (
            sorted_model_comparison
            .iloc[0]["model"]
        )

model_performance_summary = pd.DataFrame(
    [
        {
            "best_model": best_model,
            "test_rows": len(forecast_df),
            "test_MAE":
                overall_metrics["MAE"],
            "test_RMSE":
                overall_metrics["RMSE"],
            "test_sMAPE_percent":
                overall_metrics["sMAPE_percent"],
            "test_WAPE_percent":
                overall_metrics["WAPE_percent"],
            "unique_skus":
                forecast_df["sku_id"].nunique(),
            "unique_stores":
                forecast_df["store_id"].nunique()
        }
    ]
)

print(
    model_performance_summary.to_string(
        index=False
    )
)

print("✓ Model performance summary created")


# ============================================================
# [22] SAVING OVERALL FORECAST EVALUATION
# ============================================================

print("\n[22] SAVING OVERALL FORECAST EVALUATION")
print("-" * 80)

overall_file = os.path.join(
    EVALUATION_PATH,
    "overall_forecast_evaluation.csv"
)

overall_forecast_evaluation.to_csv(
    overall_file,
    index=False
)

print(
    "✓ overall_forecast_evaluation.csv"
)


# ============================================================
# [23] SAVING SKU EVALUATION
# ============================================================

print("\n[23] SAVING SKU FORECAST EVALUATION")
print("-" * 80)

sku_file = os.path.join(
    EVALUATION_PATH,
    "sku_forecast_evaluation.csv"
)

sku_forecast_evaluation.to_csv(
    sku_file,
    index=False
)

print(
    "✓ sku_forecast_evaluation.csv"
)


# ============================================================
# [24] SAVING STORE EVALUATION
# ============================================================

print("\n[24] SAVING STORE FORECAST EVALUATION")
print("-" * 80)

store_file = os.path.join(
    EVALUATION_PATH,
    "store_forecast_evaluation.csv"
)

store_forecast_evaluation.to_csv(
    store_file,
    index=False
)

print(
    "✓ store_forecast_evaluation.csv"
)


# ============================================================
# [25] SAVING FORECAST ERROR ANALYSIS
# ============================================================

print("\n[25] SAVING FORECAST ERROR ANALYSIS")
print("-" * 80)

error_file = os.path.join(
    EVALUATION_PATH,
    "forecast_error_analysis.csv"
)

forecast_error_analysis.to_csv(
    error_file,
    index=False
)

print(
    "✓ forecast_error_analysis.csv"
)


# ============================================================
# [26] SAVING ERROR SUMMARY
# ============================================================

print("\n[26] SAVING ERROR SUMMARY")
print("-" * 80)

error_summary_file = os.path.join(
    EVALUATION_PATH,
    "forecast_error_summary.csv"
)

error_summary.to_csv(
    error_summary_file,
    index=False
)

print(
    "✓ forecast_error_summary.csv"
)


# ============================================================
# [27] SAVING INVENTORY EVALUATION
# ============================================================

print("\n[27] SAVING INVENTORY EVALUATION")
print("-" * 80)

inventory_evaluation_file = os.path.join(
    EVALUATION_PATH,
    "inventory_evaluation.csv"
)

combined_inventory_evaluation.to_csv(
    inventory_evaluation_file,
    index=False
)

print(
    "✓ inventory_evaluation.csv"
)


# ============================================================
# [28] SAVING INVENTORY STATUS EVALUATION
# ============================================================

print("\n[28] SAVING INVENTORY STATUS EVALUATION")
print("-" * 80)

inventory_status_file = os.path.join(
    EVALUATION_PATH,
    "inventory_status_evaluation.csv"
)

inventory_status_evaluation.to_csv(
    inventory_status_file,
    index=False
)

print(
    "✓ inventory_status_evaluation.csv"
)


# ============================================================
# [29] SAVING REPLENISHMENT EVALUATION
# ============================================================

print("\n[29] SAVING REPLENISHMENT EVALUATION")
print("-" * 80)

replenishment_file = os.path.join(
    EVALUATION_PATH,
    "replenishment_evaluation.csv"
)

replenishment_evaluation.to_csv(
    replenishment_file,
    index=False
)

print(
    "✓ replenishment_evaluation.csv"
)


# ============================================================
# [30] SAVING HIGH-RISK INVENTORY
# ============================================================

print("\n[30] SAVING HIGH-RISK INVENTORY")
print("-" * 80)

high_risk_file = os.path.join(
    EVALUATION_PATH,
    "high_risk_inventory.csv"
)

high_risk_inventory.to_csv(
    high_risk_file,
    index=False
)

print(
    "✓ high_risk_inventory.csv"
)


# ============================================================
# [31] SAVING MODEL PERFORMANCE SUMMARY
# ============================================================

print("\n[31] SAVING MODEL PERFORMANCE SUMMARY")
print("-" * 80)

model_summary_file = os.path.join(
    EVALUATION_PATH,
    "model_performance_summary.csv"
)

model_performance_summary.to_csv(
    model_summary_file,
    index=False
)

print(
    "✓ model_performance_summary.csv"
)


# ============================================================
# [32] SAVING REPORT COPIES
# ============================================================

print("\n[32] SAVING REPORT COPIES")
print("-" * 80)

report_files = {
    "overall_forecast_evaluation.csv":
        overall_forecast_evaluation,

    "sku_forecast_evaluation.csv":
        sku_forecast_evaluation,

    "store_forecast_evaluation.csv":
        store_forecast_evaluation,

    "forecast_error_analysis.csv":
        forecast_error_analysis,

    "forecast_error_summary.csv":
        error_summary,

    "inventory_evaluation.csv":
        combined_inventory_evaluation,

    "inventory_status_evaluation.csv":
        inventory_status_evaluation,

    "replenishment_evaluation.csv":
        replenishment_evaluation,

    "high_risk_inventory.csv":
        high_risk_inventory,

    "model_performance_summary.csv":
        model_performance_summary
}

for file_name, dataframe in report_files.items():

    report_file = os.path.join(
        REPORT_PATH,
        file_name
    )

    dataframe.to_csv(
        report_file,
        index=False
    )

print(
    f"✓ {len(report_files)} report files saved"
)


# ============================================================
# [33] FINAL VALIDATION
# ============================================================

print("\n[33] FINAL PHASE 7 VALIDATION")
print("-" * 80)

generated_files = [
    overall_file,
    sku_file,
    store_file,
    error_file,
    error_summary_file,
    inventory_evaluation_file,
    inventory_status_file,
    replenishment_file,
    high_risk_file,
    model_summary_file
]

missing_generated_files = [
    file_path
    for file_path in generated_files
    if not os.path.exists(file_path)
]

if len(missing_generated_files) > 0:

    raise FileNotFoundError(
        "Some Phase 7 output files were not generated:\n"
        +
        "\n".join(
            missing_generated_files
        )
    )

print(
    f"Generated evaluation files : "
    f"{len(generated_files)}"
)

print(
    f"Missing output files       : "
    f"{len(missing_generated_files)}"
)

print("✓ All Phase 7 output files validated")


# ============================================================
# [34] FINAL SUMMARY
# ============================================================

print("\n[34] FINAL PHASE 7 SUMMARY")
print("-" * 80)

print(
    f"Best model           : "
    f"{best_model}"
)

print(
    f"Forecast rows        : "
    f"{len(forecast_df):,}"
)

print(
    f"Unique SKUs          : "
    f"{forecast_df['sku_id'].nunique():,}"
)

print(
    f"Unique stores        : "
    f"{forecast_df['store_id'].nunique():,}"
)

print(
    f"Test MAE             : "
    f"{overall_metrics['MAE']:.4f}"
)

print(
    f"Test RMSE            : "
    f"{overall_metrics['RMSE']:.4f}"
)

print(
    f"Test sMAPE           : "
    f"{overall_metrics['sMAPE_percent']:.2f}%"
)

print(
    f"Test WAPE            : "
    f"{overall_metrics['WAPE_percent']:.2f}%"
)

print(
    f"High-risk inventory  : "
    f"{len(high_risk_inventory):,}"
)


# ============================================================
# [35] FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("PHASE 7 MODEL EVALUATION COMPLETED")
print("=" * 80)

print("\nFiles generated:")

for file_path in generated_files:

    print(
        "✓ "
        +
        os.path.basename(file_path)
    )

print("\nEvaluation output directory:")
print(
    EVALUATION_PATH
)

print("\nReports directory:")
print(
    REPORT_PATH
)

print("\nNEXT PHASE:")
print(
    "PHASE 8 → DASHBOARD / FINAL APPLICATION"
)

print("=" * 80)