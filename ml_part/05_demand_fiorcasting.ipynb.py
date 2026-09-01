# ============================================================
# PROJECT FORESIGHT - PHASE 5: DEMAND FORECASTING
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# [1] PATHS
# ============================================================

print("\n" + "=" * 80)
print("PROJECT FORESIGHT - PHASE 5: DEMAND FORECASTING")
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

FEATURE_PATH = os.path.join(
    DATA_PATH,
    "features"
)

FORECAST_PATH = os.path.join(
    DATA_PATH,
    "forecasting"
)

REPORT_PATH = os.path.join(
    BASE_PATH,
    "reports",
    "forecasting"
)

os.makedirs(
    FORECAST_PATH,
    exist_ok=True
)

os.makedirs(
    REPORT_PATH,
    exist_ok=True
)

print("\n[1] PATHS")
print("-" * 80)

print(
    f"Project root      : {BASE_PATH}"
)

print(
    f"Feature input     : {FEATURE_PATH}"
)

print(
    f"Forecast output   : {FORECAST_PATH}"
)

print(
    f"Forecast reports  : {REPORT_PATH}"
)


# ============================================================
# [2] CHECKING INPUT FILE
# ============================================================

print("\n[2] CHECKING INPUT FILE")
print("-" * 80)

feature_file = os.path.join(
    FEATURE_PATH,
    "forecasting_features.csv"
)

if not os.path.exists(feature_file):

    raise FileNotFoundError(
        f"forecasting_features.csv not found:\n"
        f"{feature_file}"
    )

print("✓ forecasting_features.csv found")


# ============================================================
# [3] LOADING FEATURE DATASET
# ============================================================

print("\n[3] LOADING FEATURE DATASET")
print("-" * 80)

df = pd.read_csv(
    feature_file
)

print("✓ Feature dataset loaded")

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns):,}"
)


# ============================================================
# [4] BASIC DATA VALIDATION
# ============================================================

print("\n[4] BASIC DATA VALIDATION")
print("-" * 80)

required_columns = [
    "date",
    "sku_id",
    "store_id",
    "units_sold"
]

missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_required:

    raise ValueError(
        "Required columns missing: "
        + ", ".join(missing_required)
    )

print("✓ Required columns validated")


# ============================================================
# [5] DATE CONVERSION
# ============================================================

print("\n[5] CONVERTING DATE COLUMN")
print("-" * 80)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

if df["date"].isna().any():

    raise ValueError(
        "Invalid dates found in dataset."
    )

df = df.sort_values(
    [
        "date",
        "store_id",
        "sku_id"
    ]
).reset_index(drop=True)

print("✓ Date converted")
print("✓ Dataset sorted chronologically")


# ============================================================
# [6] TARGET VALIDATION
# ============================================================

print("\n[6] TARGET VALIDATION")
print("-" * 80)

target_column = "units_sold"

print(
    f"Target column : {target_column}"
)

print(
    f"Target mean   : "
    f"{df[target_column].mean():.4f}"
)

print(
    f"Target min    : "
    f"{df[target_column].min():.4f}"
)

print(
    f"Target max    : "
    f"{df[target_column].max():.4f}"
)

if df[target_column].isna().any():

    raise ValueError(
        "Target contains missing values."
    )

print("✓ Target validated")


# ============================================================
# [7] DATASET TIME RANGE
# ============================================================

print("\n[7] DATASET TIME RANGE")
print("-" * 80)

min_date = df["date"].min()
max_date = df["date"].max()

unique_dates = (
    df["date"]
    .sort_values()
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"Minimum date : {min_date.date()}"
)

print(
    f"Maximum date : {max_date.date()}"
)

print(
    f"Unique dates : {len(unique_dates):,}"
)


# ============================================================
# [8] TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[8] TIME-BASED TRAIN / VALIDATION / TEST SPLIT")
print("-" * 80)

# Forecasting requires chronological splitting.
#
# Earlier dates -> training
# Middle dates  -> validation
# Latest dates  -> testing

n_dates = len(unique_dates)

train_end_index = int(
    n_dates * 0.70
)

validation_end_index = int(
    n_dates * 0.85
)

train_end_date = (
    unique_dates[
        train_end_index - 1
    ]
)

validation_end_date = (
    unique_dates[
        validation_end_index - 1
    ]
)

test_start_date = (
    unique_dates[
        validation_end_index
    ]
)

train_df = df[
    df["date"] <= train_end_date
].copy()

validation_df = df[
    (
        df["date"] > train_end_date
    )
    &
    (
        df["date"] <= validation_end_date
    )
].copy()

test_df = df[
    df["date"] >= test_start_date
].copy()

print(
    f"Train      : {len(train_df):,} rows"
)

print(
    f"Validation : {len(validation_df):,} rows"
)

print(
    f"Test       : {len(test_df):,} rows"
)

print()

print(
    f"Train dates      : "
    f"{train_df['date'].min().date()} "
    f"→ "
    f"{train_df['date'].max().date()}"
)

print(
    f"Validation dates : "
    f"{validation_df['date'].min().date()} "
    f"→ "
    f"{validation_df['date'].max().date()}"
)

print(
    f"Test dates       : "
    f"{test_df['date'].min().date()} "
    f"→ "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# [9] SELECTING MODEL FEATURES
# ============================================================

print("\n[9] SELECTING MODEL FEATURES")
print("-" * 80)

# Target must never be used as an input feature.

target_column = "units_sold"

# date is excluded because date-derived features
# such as year, month, day_of_week, etc. already exist.
#
# sku_id is excluded because it is a very high-cardinality
# identifier and SKU-level historical features are already
# available.
#
# store_id is retained and handled as a categorical feature.
#
# FIX (critical data leakage): "revenue" for a given SKU-store-day is
# units_sold * unit_price for that SAME row. Feeding it in as an input
# feature lets the model back-solve units_sold almost exactly (divide by
# unit_price), which is why the previous backtest showed an unrealistically
# low ~2% WAPE. This violates the engagement's non-negotiable no-leakage
# rule (Section 07) because it is same-day, target-derived information the
# model would not actually have at forecast time. It is dropped here;
# lagged/rolling revenue features (e.g. sku_avg_revenue, price_lag_1) are
# still allowed because they only use information from before the forecast
# date.

excluded_columns = [
    target_column,
    "date",
    "sku_id",
    "revenue"
]

model_features = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print(
    f"Candidate model features : "
    f"{len(model_features)}"
)

print("\nExcluded columns:")

for column in excluded_columns:

    if column in df.columns:

        print(
            f"✓ {column}"
        )


# ============================================================
# [10] IDENTIFYING NUMERIC AND CATEGORICAL FEATURES
# ============================================================

print("\n[10] IDENTIFYING FEATURE TYPES")
print("-" * 80)

categorical_features = (
    df[
        model_features
    ]
    .select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    )
    .columns
    .tolist()
)

numeric_features = [
    column
    for column in model_features
    if column not in categorical_features
]

print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)

print("\nCategorical columns:")

if len(categorical_features) > 0:

    for column in categorical_features:

        print(
            f"✓ {column}"
        )

else:

    print("None")


# ============================================================
# [11] PREPARING RAW MODEL DATA
# ============================================================

print("\n[11] PREPARING MODEL DATA")
print("-" * 80)

X = df[
    model_features
].copy()

y = df[
    target_column
].copy()

print(
    f"Raw model feature columns : "
    f"{X.shape[1]}"
)

print(
    f"Categorical columns       : "
    f"{len(categorical_features)}"
)

print(
    f"Numeric columns           : "
    f"{len(numeric_features)}"
)


# ============================================================
# [12] CREATING PREPROCESSING PIPELINE
# ============================================================

print("\n[12] CREATING PREPROCESSING PIPELINE")
print("-" * 80)

# Numeric features remain numeric.

numeric_transformer = "passthrough"

# Categorical features are converted into numerical
# one-hot encoded columns.

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# IMPORTANT:
# Use the already-created chronological train/validation/test
# dataframes. Do NOT reference undefined variables such as
# features, train_data, validation_data or test_data.

X_train_raw = train_df[
    model_features
].copy()

X_validation_raw = validation_df[
    model_features
].copy()

X_test_raw = test_df[
    model_features
].copy()

y_train = train_df[
    target_column
].copy()

y_validation = validation_df[
    target_column
].copy()

y_test = test_df[
    target_column
].copy()

print(
    f"X_train raw       : "
    f"{X_train_raw.shape}"
)

print(
    f"X_validation raw  : "
    f"{X_validation_raw.shape}"
)

print(
    f"X_test raw        : "
    f"{X_test_raw.shape}"
)

print(
    "\nFitting preprocessing pipeline "
    "on training data..."
)

# Fit ONLY on training data.

X_train = preprocessor.fit_transform(
    X_train_raw
)

X_validation = preprocessor.transform(
    X_validation_raw
)

X_test = preprocessor.transform(
    X_test_raw
)

print(
    "✓ Preprocessing pipeline fitted"
)

print(
    f"X_train processed      : "
    f"{X_train.shape}"
)

print(
    f"X_validation processed : "
    f"{X_validation.shape}"
)

print(
    f"X_test processed       : "
    f"{X_test.shape}"
)

print(
    "✓ Categorical features encoded"
)

print(
    "✓ Numerical features preserved"
)

print(
    "✓ Unknown categories handled safely"
)


# ============================================================
# [13] BASELINE FORECAST
# ============================================================

print("\n[13] CREATING BASELINE FORECAST")
print("-" * 80)

# Baseline:
# Predict using the historical 7-day rolling demand.

if "rolling_mean_7" in test_df.columns:

    baseline_predictions = (
        test_df[
            "rolling_mean_7"
        ]
        .fillna(
            train_df[
                target_column
            ].mean()
        )
        .values
    )

else:

    baseline_predictions = np.full(
        len(test_df),
        train_df[
            target_column
        ].mean()
    )

baseline_predictions = np.maximum(
    baseline_predictions,
    0
)

print(
    "✓ Baseline predictions generated"
)


# ============================================================
# [14] EVALUATION METRICS
# ============================================================

print("\n[14] DEFINING EVALUATION METRICS")
print("-" * 80)


def calculate_smape(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = (
        np.abs(actual)
        +
        np.abs(predicted)
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
                predicted[mask]
                -
                actual[mask]
            )
            /
            denominator[mask]
        )
    )


def calculate_wape(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:

        return 0.0

    return (
        100
        *
        np.sum(
            np.abs(
                actual
                -
                predicted
            )
        )
        /
        denominator
    )


def evaluate_model(
    model_name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    smape = calculate_smape(
        actual,
        predicted
    )

    wape = calculate_wape(
        actual,
        predicted
    )

    return {
        "model": model_name,
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
# [15] EVALUATE BASELINE
# ============================================================

print("\n[15] EVALUATING BASELINE")
print("-" * 80)

results = []

baseline_result = evaluate_model(
    "Rolling Mean 7-Day Baseline",
    y_test,
    baseline_predictions
)

results.append(
    baseline_result
)

print(
    f"MAE  : "
    f"{baseline_result['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{baseline_result['RMSE']:.4f}"
)

print(
    f"sMAPE: "
    f"{baseline_result['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE : "
    f"{baseline_result['WAPE_percent']:.2f}%"
)


# ============================================================
# [16] RANDOM FOREST MODEL
# ============================================================

print("\n[16] TRAINING RANDOM FOREST")
print("-" * 80)

rf_model = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print(
    "Training Random Forest..."
)

rf_model.fit(
    X_train,
    y_train
)

print(
    "✓ Random Forest trained"
)


# ============================================================
# [17] RANDOM FOREST VALIDATION
# ============================================================

print("\n[17] RANDOM FOREST VALIDATION")
print("-" * 80)

rf_validation_predictions = (
    rf_model.predict(
        X_validation
    )
)

rf_validation_predictions = np.maximum(
    rf_validation_predictions,
    0
)

rf_validation_result = evaluate_model(
    "Random Forest",
    y_validation,
    rf_validation_predictions
)

print(
    f"MAE  : "
    f"{rf_validation_result['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{rf_validation_result['RMSE']:.4f}"
)

print(
    f"sMAPE: "
    f"{rf_validation_result['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE : "
    f"{rf_validation_result['WAPE_percent']:.2f}%"
)


# ============================================================
# [18] HISTOGRAM GRADIENT BOOSTING
# ============================================================

print(
    "\n[18] TRAINING HISTOGRAM GRADIENT BOOSTING"
)

print("-" * 80)

hgb_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

print(
    "Training HistGradientBoosting..."
)

hgb_model.fit(
    X_train,
    y_train
)

print(
    "✓ HistGradientBoosting trained"
)


# ============================================================
# [19] HISTOGRAM GRADIENT BOOSTING VALIDATION
# ============================================================

print(
    "\n[19] HISTOGRAM GRADIENT BOOSTING VALIDATION"
)

print("-" * 80)

hgb_validation_predictions = (
    hgb_model.predict(
        X_validation
    )
)

hgb_validation_predictions = np.maximum(
    hgb_validation_predictions,
    0
)

hgb_validation_result = evaluate_model(
    "HistGradientBoosting",
    y_validation,
    hgb_validation_predictions
)

print(
    f"MAE  : "
    f"{hgb_validation_result['MAE']:.4f}"
)

print(
    f"RMSE : "
    f"{hgb_validation_result['RMSE']:.4f}"
)

print(
    f"sMAPE: "
    f"{hgb_validation_result['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE : "
    f"{hgb_validation_result['WAPE_percent']:.2f}%"
)


# ============================================================
# [20] MODEL COMPARISON
# ============================================================

print("\n[20] MODEL COMPARISON")
print("-" * 80)

# Baseline is included only as a benchmark.
# The final ML model is selected between the two ML models.

validation_results = pd.DataFrame(
    [
        baseline_result,
        rf_validation_result,
        hgb_validation_result
    ]
)

validation_results = (
    validation_results
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)

print(
    validation_results.to_string(
        index=False
    )
)

ml_validation_results = pd.DataFrame(
    [
        rf_validation_result,
        hgb_validation_result
    ]
)

ml_validation_results = (
    ml_validation_results
    .sort_values(
        "MAE"
    )
    .reset_index(
        drop=True
    )
)

best_model_name = (
    ml_validation_results
    .iloc[0]["model"]
)

print(
    f"\n✓ Best ML validation model: "
    f"{best_model_name}"
)


# ============================================================
# [21] SELECT FINAL MODEL
# ============================================================

print("\n[21] SELECTING FINAL MODEL")
print("-" * 80)

if best_model_name == "Random Forest":

    best_model = rf_model

elif best_model_name == "HistGradientBoosting":

    best_model = hgb_model

else:

    raise ValueError(
        "Unexpected model selected."
    )

print(
    f"Selected model: "
    f"{best_model_name}"
)


# ============================================================
# [22] FINAL TEST PREDICTIONS
# ============================================================

print("\n[22] GENERATING FINAL TEST PREDICTIONS")
print("-" * 80)

test_predictions = (
    best_model.predict(
        X_test
    )
)

# Demand cannot be negative.

test_predictions = np.maximum(
    test_predictions,
    0
)

print(
    "✓ Test predictions generated"
)


# ============================================================
# [23] FINAL TEST EVALUATION
# ============================================================

print("\n[23] FINAL TEST EVALUATION")
print("-" * 80)

final_test_result = evaluate_model(
    best_model_name,
    y_test,
    test_predictions
)

print(
    f"Model : "
    f"{final_test_result['model']}"
)

print(
    f"MAE   : "
    f"{final_test_result['MAE']:.4f}"
)

print(
    f"RMSE  : "
    f"{final_test_result['RMSE']:.4f}"
)

print(
    f"sMAPE : "
    f"{final_test_result['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE  : "
    f"{final_test_result['WAPE_percent']:.2f}%"
)


# ============================================================
# [24] CREATE FORECAST OUTPUT
# ============================================================

print("\n[24] CREATING FORECAST OUTPUT")
print("-" * 80)

forecast_output = test_df[
    [
        "date",
        "sku_id",
        "store_id"
    ]
].copy()

forecast_output[
    "actual_units_sold"
] = y_test.values

forecast_output[
    "predicted_units_sold"
] = test_predictions

forecast_output[
    "forecast_error"
] = (
    forecast_output[
        "actual_units_sold"
    ]
    -
    forecast_output[
        "predicted_units_sold"
    ]
)

forecast_output[
    "absolute_error"
] = np.abs(
    forecast_output[
        "forecast_error"
    ]
)

print(
    "✓ Forecast output created"
)


# ============================================================
# [25] SAVE FORECAST PREDICTIONS
# ============================================================

print("\n[25] SAVING FORECAST PREDICTIONS")
print("-" * 80)

forecast_file = os.path.join(
    FORECAST_PATH,
    "demand_forecasts.csv"
)

forecast_output.to_csv(
    forecast_file,
    index=False
)

print(
    "✓ demand_forecasts.csv"
)


# ============================================================
# [26] SAVE MODEL METRICS
# ============================================================

print("\n[26] SAVING MODEL METRICS")
print("-" * 80)

test_results = pd.DataFrame(
    [
        final_test_result
    ]
)

metrics_file = os.path.join(
    FORECAST_PATH,
    "forecast_metrics.csv"
)

test_results.to_csv(
    metrics_file,
    index=False
)

print(
    "✓ forecast_metrics.csv"
)


# ============================================================
# [27] SAVE MODEL COMPARISON
# ============================================================

print("\n[27] SAVING MODEL COMPARISON")
print("-" * 80)

comparison_file = os.path.join(
    FORECAST_PATH,
    "model_comparison.csv"
)

validation_results.to_csv(
    comparison_file,
    index=False
)

print(
    "✓ model_comparison.csv"
)


# ============================================================
# [28] SAVE FEATURE IMPORTANCE
# ============================================================

print("\n[28] SAVING FEATURE IMPORTANCE")
print("-" * 80)

if best_model_name == "Random Forest":

    importance_values = (
        best_model
        .feature_importances_
    )

    # Get the actual feature names after preprocessing.

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_file = os.path.join(
        FORECAST_PATH,
        "feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_file,
        index=False
    )

    print(
        "✓ feature_importance.csv"
    )

else:

    print(
        "Feature importance skipped for "
        "HistGradientBoosting."
    )


# ============================================================
# [29] SAVE PREPROCESSOR
# ============================================================

print("\n[29] SAVING PREPROCESSOR")
print("-" * 80)

preprocessor_file = os.path.join(
    FORECAST_PATH,
    "forecast_preprocessor.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_file
)

print(
    "✓ forecast_preprocessor.pkl"
)


# ============================================================
# [30] SAVE FINAL MODEL
# ============================================================

print("\n[30] SAVING FINAL MODEL")
print("-" * 80)

model_file = os.path.join(
    FORECAST_PATH,
    "best_demand_forecasting_model.pkl"
)

joblib.dump(
    best_model,
    model_file
)

print(
    "✓ best_demand_forecasting_model.pkl"
)


# ============================================================
# [31] SAVE MODEL FEATURE LIST
# ============================================================

print("\n[31] SAVING MODEL FEATURE LIST")
print("-" * 80)

feature_list = pd.DataFrame(
    {
        "feature": model_features
    }
)

feature_list_file = os.path.join(
    FORECAST_PATH,
    "model_features.csv"
)

feature_list.to_csv(
    feature_list_file,
    index=False
)

print(
    "✓ model_features.csv"
)


# ============================================================
# [32] FORECAST SUMMARY
# ============================================================

print("\n[32] CREATING FORECAST SUMMARY")
print("-" * 80)

summary = pd.DataFrame(
    {
        "metric": [
            "training_rows",
            "validation_rows",
            "test_rows",
            "training_dates",
            "validation_dates",
            "test_dates",
            "best_model",
            "test_mae",
            "test_rmse",
            "test_smape_percent",
            "test_wape_percent"
        ],

        "value": [
            len(train_df),
            len(validation_df),
            len(test_df),

            train_df[
                "date"
            ].nunique(),

            validation_df[
                "date"
            ].nunique(),

            test_df[
                "date"
            ].nunique(),

            best_model_name,

            final_test_result[
                "MAE"
            ],

            final_test_result[
                "RMSE"
            ],

            final_test_result[
                "sMAPE_percent"
            ],

            final_test_result[
                "WAPE_percent"
            ]
        ]
    }
)

summary_file = os.path.join(
    FORECAST_PATH,
    "forecast_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    "✓ forecast_summary.csv"
)


# ============================================================
# [33] FINAL FORECAST VALIDATION
# ============================================================

print("\n[33] FINAL FORECAST VALIDATION")
print("-" * 80)

print(
    f"Forecast rows : "
    f"{len(forecast_output):,}"
)

print(
    f"Missing actuals : "
    f"{forecast_output['actual_units_sold'].isna().sum():,}"
)

print(
    f"Missing forecasts : "
    f"{forecast_output['predicted_units_sold'].isna().sum():,}"
)

print(
    f"Negative forecasts : "
    f""
    f"{(forecast_output['predicted_units_sold'] < 0).sum():,}"
)

if (
    forecast_output[
        "predicted_units_sold"
    ]
    .isna()
    .sum()
    == 0
):

    print(
        "✓ No missing forecasts"
    )

else:

    raise ValueError(
        "Missing forecast values detected."
    )

if (
    forecast_output[
        "predicted_units_sold"
    ]
    < 0
).sum() == 0:

    print(
        "✓ No negative forecasts"
    )

else:

    raise ValueError(
        "Negative forecast values detected."
    )


# ============================================================
# [34] FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print(
    "PHASE 5 DEMAND FORECASTING COMPLETED"
)
print("=" * 80)

print("\nBest ML model:")
print(
    best_model_name
)

print("\nFinal test performance:")

print(
    f"MAE   : "
    f"{final_test_result['MAE']:.4f}"
)

print(
    f"RMSE  : "
    f"{final_test_result['RMSE']:.4f}"
)

print(
    f"sMAPE : "
    f"{final_test_result['sMAPE_percent']:.2f}%"
)

print(
    f"WAPE  : "
    f"{final_test_result['WAPE_percent']:.2f}%"
)

print("\nFiles generated:")

print(
    "✓ demand_forecasts.csv"
)

print(
    "✓ forecast_metrics.csv"
)

print(
    "✓ model_comparison.csv"
)

print(
    "✓ model_features.csv"
)

print(
    "✓ forecast_summary.csv"
)

print(
    "✓ forecast_preprocessor.pkl"
)

print(
    "✓ best_demand_forecasting_model.pkl"
)

if best_model_name == "Random Forest":

    print(
        "✓ feature_importance.csv"
    )

print(
    "\nForecast output directory:"
)

print(
    FORECAST_PATH
)

print("\nNEXT PHASE:")
print(
    "PHASE 6 → INVENTORY OPTIMIZATION"
)

print("=" * 80)