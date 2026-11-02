import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURES = [
    "district_code",
    "week_sin",
    "week_cos",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_4",
    "cases_rolling_avg_4",
]

FEATURE_TYPES = [
    "nominal",      # district_code
    "continuous",   # week_sin
    "continuous",   # week_cos
    "continuous",   # cases_lag_1
    "continuous",   # cases_lag_2
    "continuous",   # cases_lag_3
    "continuous",   # cases_lag_4
    "continuous",   # cases_rolling_avg_4
]


def load_dataset(file_path):
    """Load the dengue dataset and prepare its date column."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df["start.date"] = pd.to_datetime(df["start.date"])
    return df


def weekly_dengue_cases(
    df,
    output_path="outputs/total_cases_across_all_districts_per_week.png",
):
    """Save and display total weekly dengue cases across all districts."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    weekly_cases = (
        df.groupby("start.date")["cases"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        weekly_cases["start.date"],
        weekly_cases["cases"],
        label="Weekly Cases",
    )

    plt.title("Total Weekly Dengue Cases in Sri Lanka")
    plt.xlabel("Start Date")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the image before displaying it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {output_path}")

    plt.show()
    plt.close()


def seasonal_dengue_cases(
    df,
    output_path="outputs/seasonal_dengue_cases.png",
):
    """Save and display average dengue cases by week number."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    seasonality = (
        df.groupby("week")["cases"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 4))

    plt.bar(
        seasonality["week"],
        seasonality["cases"],
        color="orange",
    )

    plt.title("Dengue Seasonality by Week Number")
    plt.xlabel("Week Number")
    plt.ylabel("Average Cases")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save the image before displaying it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {output_path}")

    plt.show()
    plt.close()


def print_dataset_summary(df):
    """Print basic information about the dataset."""
    print("Total rows:", len(df))
    print("Number of districts:", df["district"].nunique())
    print("Number of unique weeks:", df["start.date"].nunique())

    print("\nNull values:")
    print(df.isnull().sum())

    duplicates = df.duplicated(
        subset=["district", "start.date"]
    ).sum()
    print("\nDuplicate district-week records:", duplicates)

    total_weeks = df["start.date"].nunique()
    weeks_per_district = df.groupby("district")["start.date"].nunique()

    print("\nMissing weeks per district:")
    print((total_weeks - weeks_per_district).to_string())

    print("\nCase statistics:")
    print("Median:", df["cases"].median())
    print("Mean:", round(df["cases"].mean(), 1))
    print("Maximum:", df["cases"].max())
    
    weekly_dengue_cases(df)
    seasonal_dengue_cases(df)


def fill_missing_weeks(df):
    """
    Add missing district/date combinations for dates already present
    in the dataset. Interpolate missing case values within each district.

    Note: this preserves the original preprocessing approach. Linear
    interpolation can use later observations, so it can introduce
    look-ahead leakage when missing values occur near the train/test
    boundary. Review the missing-data policy before final reporting.
    """
    df = df.copy()
    df["start.date"] = pd.to_datetime(df["start.date"])

    all_dates = sorted(df["start.date"].unique())
    all_districts = df["district"].unique()

    index = pd.MultiIndex.from_product(
        [all_districts, all_dates],
        names=["district", "start.date"],
    )

    df_full = (
        df.set_index(["district", "start.date"])
        .reindex(index)
        .reset_index()
    )

    df_full["cases"] = (
        df_full.groupby("district")["cases"]
        .transform(
            lambda values: values.interpolate(method="linear").bfill().ffill()
        )
    )

    df_full["year"] = df_full["start.date"].dt.year
    df_full["week"] = (
        df_full["start.date"].dt.isocalendar().week.astype(float)
    )

    return df_full


def create_features(df):
    """Create model features and the one-week-ahead target."""
    df = df.copy()
    df = df.sort_values(
        ["district", "start.date"]
    ).reset_index(drop=True)

    district_categories = sorted(df["district"].dropna().unique().tolist())
    district_mapping = {
        district: code
        for code, district in enumerate(district_categories)
    }
    df["district_code"] = df["district"].map(district_mapping).astype(int)

    days_in_year = np.where(
        df["start.date"].dt.is_leap_year, 366, 365
    )
    year_phase = (df["start.date"].dt.dayofyear - 1) / days_in_year

    df["week_sin"] = np.sin(2 * np.pi * year_phase)
    df["week_cos"] = np.cos(2 * np.pi * year_phase)

    for lag in range(1, 5):
        df[f"cases_lag_{lag}"] = (
            df.groupby("district")["cases"].shift(lag - 1)
        )

    df["cases_rolling_avg_4"] = (
        df.groupby("district")["cases"]
        .transform(lambda values: values.rolling(4, min_periods=4).mean())
    )

    df["target"] = (
        df.groupby("district")["cases"].shift(-1)
    )
    df["target_date"] = (
        df.groupby("district")["start.date"].shift(-1)
    )

    df = df.dropna(
        subset=FEATURES + ["target", "target_date"]
    ).reset_index(drop=True)

    return df, district_categories


def prepare_data(file_path):
    """Load, inspect, preprocess, and split the dataset by target year."""
    df = load_dataset(file_path)
    print_dataset_summary(df)

    df = fill_missing_weeks(df)
    print("\nData shape after filling:", df.shape)
    print("Null cases after filling:", df["cases"].isnull().sum())

    df, district_categories = create_features(df)

    max_year = df["target_date"].dt.year.max()

    train_df = df[df["target_date"].dt.year < max_year].copy()
    test_df = df[df["target_date"].dt.year == max_year].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)
    print(
        "Training years:",
        train_df["target_date"].dt.year.min(),
        "-",
        train_df["target_date"].dt.year.max(),
    )
    print("Testing year:", max_year)

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "features": FEATURES,
        "feature_types": FEATURE_TYPES,
        "district_categories": district_categories,
        "test_year": int(max_year),
    }
