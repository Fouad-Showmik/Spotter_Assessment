import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models import MedianBaseline, create_lightgbm, create_catboost
from src.features import get_categorical_features

#Return the four expanding walk-forward validation periods
def get_folds():

    return [
        {
            "fold": 1,
            "train_end": "2025-07-01",
            "valid_start": "2025-07-01",
            "valid_end": "2025-08-01",
        },
        {
            "fold": 2,
            "train_end": "2025-08-01",
            "valid_start": "2025-08-01",
            "valid_end": "2025-09-01",
        },
        {
            "fold": 3,
            "train_end": "2025-09-01",
            "valid_start": "2025-09-01",
            "valid_end": "2025-10-01",
        },
        {
            "fold": 4,
            "train_end": "2025-10-01",
            "valid_start": "2025-10-01",
            "valid_end": "2025-11-01",
        },
    ]


#Calculate regression metrics
def calculate_metrics(y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_true, predictions)
    )

    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2

#Train one model and return its validation predictions
def train_and_predict(
    model_name,
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    if model_name == "median_baseline":
        model = MedianBaseline()
        model.fit(X_train, y_train)

    elif model_name == "lightgbm":
        model = create_lightgbm()
        model.fit(X_train, y_train)

    elif model_name == "catboost":
        model = create_catboost()

        categorical_features = get_categorical_features(X_train)

        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_valid, y_valid),
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.predict(X_valid)


#Run all models across the four walk-forward folds
def evaluate_models(
    df,
    X,
    model_names,
):

    results = []

    dates = df["date"]
    y = df["posted_rate"]

    for model_name in model_names:

        for fold in get_folds():

            train_mask = dates < fold["train_end"]

            valid_mask = (
                (dates >= fold["valid_start"])
                & (dates < fold["valid_end"])
            )

            X_train = X.loc[train_mask]
            X_valid = X.loc[valid_mask]

            y_train = y.loc[train_mask]
            y_valid = y.loc[valid_mask]

            predictions = train_and_predict(
                model_name,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )

            mae, rmse, r2 = calculate_metrics(
                y_valid,
                predictions,
            )

            results.append(
                {
                    "model": model_name,
                    "fold": fold["fold"],
                    "train_rows": len(X_train),
                    "valid_rows": len(X_valid),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

            print(
                f"{model_name:<16} "
                f"Fold {fold['fold']} | "
                f"MAE: {mae:,.2f} | "
                f"RMSE: {rmse:,.2f} | "
                f"R²: {r2:.4f}"
            )

    return pd.DataFrame(results)