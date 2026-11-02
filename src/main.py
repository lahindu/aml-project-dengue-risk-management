from data_prep import prepare_data
from model_training import train_model
from evaluation import run_evaluation


DATASET_PATH = "srilanka_dengue_weekly_by_district.csv"
MODEL_PATH = "srilanka_dengue_ebm_model.pkl"
PLOT_PATH = "outputs/actual_vs_predicted.png"


def main():
    # Prepare the dataset.
    data = prepare_data(DATASET_PATH)

    # Train and save the model.
    train_model(
        X_train=data["X_train"],
        y_train=data["y_train"],
        train_df=data["train_df"],
        district_categories=data["district_categories"],
        model_path=MODEL_PATH,
        risk_percentile=0.75
    )

    # Evaluate the saved model.
    results = run_evaluation(
        train_df=data["train_df"],
        test_df=data["test_df"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        model_path=MODEL_PATH,
        plot_path=PLOT_PATH,
        risk_percentile=0.75,
    )

    print("\nCompleted.")
    print("Saved model:", MODEL_PATH)
    print("Saved plot:", PLOT_PATH)

    return results


if __name__ == "__main__":
    main()
