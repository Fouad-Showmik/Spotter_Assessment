from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]

#Load a training or validation CSV file
def load_data(path: Path) -> pd.DataFrame:

    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"])

    return df

#Apply the cleaning decisions identified during EDA
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Negative weights were identified during EDA.

    negative_weight = df["weight"] < 0

    df.loc[negative_weight, "weight"] = (
        df.loc[negative_weight, "weight"].abs()
    )

    # Keep numerical missing values as NaN.
    # Explicit category for any missing categorical value.
    for column in ["pickup", "delivery", "equipment"]:
        df[column] = df[column].fillna("UNKNOWN")

    return df


#Prepare the fixed December scenario input
def prepare_december_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    return df


#Build a city-to-coordinate lookup from development data.
def build_coordinate_lookup(
    df: pd.DataFrame,
) -> pd.DataFrame:

    pickup_coordinates = (
        df[
            ["pickup", "pickup_lat", "pickup_lon"]
        ]
        .rename(
            columns={
                "pickup": "city",
                "pickup_lat": "lat",
                "pickup_lon": "lon",
            }
        )
    )

    delivery_coordinates = (
        df[
            ["delivery", "delivery_lat", "delivery_lon"]
        ]
        .rename(
            columns={
                "delivery": "city",
                "delivery_lat": "lat",
                "delivery_lon": "lon",
            }
        )
    )

    coordinates = pd.concat(
        [
            pickup_coordinates,
            delivery_coordinates,
        ],
        ignore_index=True,
    )

    # A city should have one consistent coordinate pair.
    coordinate_counts = (
        coordinates
        .groupby("city")[["lat", "lon"]]
        .nunique()
    )

    conflicts = coordinate_counts[
        (coordinate_counts["lat"] > 1)
        | (coordinate_counts["lon"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            "Some cities have conflicting coordinates."
        )

    return (
        coordinates
        .drop_duplicates(subset="city")
        .set_index("city")
    )


#Add missing geographic coordinates using city names
def add_coordinates(
    df: pd.DataFrame,
    coordinate_lookup: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "pickup_lat" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(
            coordinate_lookup["lat"]
        )
        df["pickup_lon"] = df["pickup"].map(
            coordinate_lookup["lon"]
        )

    if "delivery_lat" not in df.columns:
        df["delivery_lat"] = df["delivery"].map(
            coordinate_lookup["lat"]
        )
        df["delivery_lon"] = df["delivery"].map(
            coordinate_lookup["lon"]
        )

    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]

    if df[coordinate_columns].isna().any().any():
        raise ValueError(
            "Could not find coordinates for one or more cities."
        )

    return df