import pickle

from interpret.glassbox import ExplainableBoostingRegressor

from data_prep import FEATURES, FEATURE_TYPES


def calculate_district_thresholds(train_df, percentile=0.75):
    return (
        train_df.groupby("district")["target"]
        .quantile(percentile)
        .to_dict()
    )


def train_model(
    X_train,
    y_train,
    train_df,
    district_categories,
    model_path="srilanka_dengue_ebm_model.pkl",
    risk_percentile=0.75
):
    """Train the EBM regression model and save the model metadata."""
    model = ExplainableBoostingRegressor(
        feature_names=FEATURES,
        feature_types=FEATURE_TYPES,
        objective="poisson_deviance",
        interactions=10,
        max_interaction_bins=16,
        outer_bags=8,
        learning_rate=0.015,
        max_rounds=10000,
        early_stopping_rounds=100,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("EBM model training completed.")

    district_thresholds = calculate_district_thresholds(
        train_df,
        percentile=risk_percentile
    )

    model_data = {
        "model": model,
        "features": FEATURES,
        "feature_types": FEATURE_TYPES,
        "district_categories": district_categories,
        "district_thresholds": district_thresholds,
        "risk_percentile": risk_percentile,
        "target": "next_week_cases"
    }

    with open(model_path, "wb") as file:
        pickle.dump(
            model_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(f"Model saved: {model_path}")
    return model


def load_model(model_path="srilanka_dengue_ebm_model.pkl"):
    """Load the saved EBM model and metadata."""
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)

    return model_data
