import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from model_training import calculate_district_thresholds, load_model


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics for case-count predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\nRegression Evaluation")
    print("---------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def get_district_thresholds(model_data, train_df=None, percentile=0.75):
    """
    Return the district thresholds saved with the model.

    The thresholds are calculated once during training and stored in the
    pickle file, so evaluation and later predictions use exactly the same
    values. If an older model file has no thresholds, they are calculated
    again from the training data.
    """
    thresholds = model_data.get("district_thresholds")

    if thresholds is None:
        if train_df is None:
            raise ValueError(
                "The model file has no district_thresholds and no train_df "
                "was given to calculate them."
            )
        print("\nModel file has no district thresholds. Calculating them from train_df.")
        thresholds = calculate_district_thresholds(train_df, percentile=percentile)

    # Saved as a dict -> use a Series for mapping and printing
    return pd.Series(thresholds, name="threshold")


def classify_risk_and_evaluate(
    test_df,
    y_pred,
    district_thresholds,
    percentile=0.75,
):
    """
    Classify actual and predicted test cases as high risk using the saved
    district thresholds, then evaluate the risk classification.
    """
    risk_df = test_df[["district", "target"]].copy()
    risk_df["predicted_cases"] = np.asarray(y_pred)
    risk_df["threshold"] = risk_df["district"].map(district_thresholds)

    # Exclude districts that do not have a training threshold.
    missing = risk_df.loc[risk_df["threshold"].isna(), "district"].unique()
    if len(missing) > 0:
        print("\nNo threshold for these districts (excluded):", list(missing))
    risk_df = risk_df.dropna(subset=["threshold"]).copy()

    risk_df["actual_risk"] = (
        risk_df["target"] >= risk_df["threshold"]
    ).astype(int)

    risk_df["predicted_risk"] = (
        risk_df["predicted_cases"] >= risk_df["threshold"]
    ).astype(int)

    # A continuous score is required for ROC-AUC.
    risk_df["risk_score"] = (
        risk_df["predicted_cases"] - risk_df["threshold"]
    )

    y_true = risk_df["actual_risk"]
    y_class = risk_df["predicted_risk"]
    y_score = risk_df["risk_score"]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_class),
        "Precision": precision_score(
            y_true, y_class, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_class, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_class, zero_division=0
        ),
    }

    print("\nDengue Risk Classification Evaluation")
    print("-------------------------------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    if y_true.nunique() == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    else:
        metrics["ROC-AUC"] = np.nan
        print("ROC-AUC: Cannot calculate (only one actual class)")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_class))

    print(f"\nDistrict thresholds ({percentile:.0%} percentile, saved with the model):")
    print(district_thresholds.round(2).to_string())

    return risk_df, metrics


def save_actual_vs_predicted_plot(
    y_test,
    y_pred,
    output_path="outputs/actual_vs_predicted.png",
    number_of_records=200,
):
    """Save and display the actual-versus-predicted plot."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = min(number_of_records, len(y_test))
    x_values = range(count)

    plt.figure(figsize=(14, 6))
    plt.plot(
        x_values,
        np.asarray(y_test)[:count],
        label="Actual Cases",
    )
    plt.plot(
        x_values,
        np.asarray(y_pred)[:count],
        label="Predicted Cases",
    )

    plt.title("EBM - Actual vs Predicted Dengue Cases")
    plt.xlabel("Test Records")
    plt.ylabel("Dengue Cases")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the image before displaying it.
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {output_path}")

    plt.show()
    plt.close()


def run_evaluation(
    train_df,
    test_df,
    X_test,
    y_test,
    model_path="srilanka_dengue_ebm_model.pkl",
    plot_path="outputs/actual_vs_predicted.png",
    risk_percentile=0.75,
):
    """Load the model and its thresholds, predict test cases, and run evaluations."""
    model_data = load_model(model_path)
    model = model_data["model"]

    # Use the percentile that was saved with the model, if it is there
    risk_percentile = model_data.get("risk_percentile", risk_percentile)

    district_thresholds = get_district_thresholds(
        model_data,
        train_df=train_df,
        percentile=risk_percentile,
    )
    print("\nModel and district thresholds loaded successfully.")
    if "Colombo" in district_thresholds.index:
        print("Colombo risk threshold:", round(district_thresholds["Colombo"], 2))

    y_pred = model.predict(X_test)

    regression_metrics = evaluate_regression(y_test, y_pred)

    risk_df, classification_metrics = classify_risk_and_evaluate(
        test_df=test_df,
        y_pred=y_pred,
        district_thresholds=district_thresholds,
        percentile=risk_percentile,
    )

    save_actual_vs_predicted_plot(
        y_test=y_test,
        y_pred=y_pred,
        output_path=plot_path,
    )

    return {
        "y_pred": y_pred,
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "risk_df": risk_df,
        "district_thresholds": district_thresholds,
    }