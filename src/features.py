import pandas as pd


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]


MODEL_FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
    "days_since_start",
    "route",
    "weight_missing",
]


# Create the features used by the models
def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    features = df.copy()

    # Date-derived features.
    features["month"] = (
        features["date"].dt.month
    )

    features["day_of_week"] = (
        features["date"].dt.dayofweek
    )

    features["day_of_year"] = (
        features["date"].dt.dayofyear
    )

    features["days_since_start"] = (
        features["date"]
        - pd.Timestamp("2025-01-01")
    ).dt.days

    # Pickup-delivery route.
    features["route"] = (
        features["pickup"].astype(str)
        + "__"
        + features["delivery"].astype(str)
    )

    # Missingness indicator for weight.
    features["weight_missing"] = (
        features["weight"].isna().astype(int)
    )

    # Remove columns that are not model features.
    features = features.drop(
        columns=[
            "load_id",
            "date",
            "market_index",
            "quote_signal",
            "posted_rate",
            "predicted_rate",
        ],
        errors="ignore",
    )

    # Explicitly enforce the same feature order
    # for training, validation, and December.
    missing_features = [
        column
        for column in MODEL_FEATURES
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    features = features[MODEL_FEATURES]

    return features


# Return the categorical features present in the data
def get_categorical_features(
    df: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
    ]