import pickle
from pathlib import Path

import pandas as pd

from src.features import create_features
from src.preprocessing import (
    add_coordinates,
    build_coordinate_lookup,
    clean_data,
    load_data,
    prepare_december_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "train-test.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation.csv"
)

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation-predictions-template.csv"
)

DECEMBER_PATH = (
    PROJECT_ROOT
    / "data"
    / "december-chart-inputs.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_model.pkl"
)

VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "validation_predictions.csv"
)

#Load the final trained CatBoost model
def load_final_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model

#Predict all rows in validation.csv
def predict_validation(model):

    # Load validation data.
    validation_df = load_data(
        VALIDATION_PATH
    )

    # Apply the same cleaning used during training.
    validation_df = clean_data(
        validation_df
    )

    # Create the same 16 model features.
    X_validation = create_features(
        validation_df
    )

    # Predict validation rates.
    predictions = model.predict(
        X_validation
    )

    # Use the supplied template.
    template_df = pd.read_csv(
        TEMPLATE_PATH
    )

    # Check that the template has the expected number of rows.
    if len(template_df) != len(validation_df):
        raise ValueError(
            "Template row count does not match validation data."
        )

    # Check that the load_id order is exactly the same.
    if not template_df["load_id"].equals(
        validation_df["load_id"]
    ):
        raise ValueError(
            "Template load_id values do not match validation data."
        )

    # Fill the supplied template.
    template_df["predicted_rate"] = predictions

    return template_df


#Predict all rows in the fixed December scenario
def predict_december(model):

    # Keep the original December input structure.
    december_input = pd.read_csv(
        DECEMBER_PATH
    )

    # Convert the date column to datetime.
    december_df = prepare_december_data(
        december_input
    )

    # Load development data for the city-coordinate lookup.
    development_df = load_data(
        TRAIN_PATH
    )

    development_df = clean_data(
        development_df
    )

    # Build city -> coordinate mapping.
    coordinate_lookup = build_coordinate_lookup(
        development_df
    )

    # Add pickup and delivery coordinates.
    december_df = add_coordinates(
        december_df,
        coordinate_lookup,
    )

    # Create the same 16 model features.
    X_december = create_features(
        december_df
    )

    # Predict December rates.
    predictions = model.predict(
        X_december
    )

    # Fill only the existing predicted_rate column.
    december_input["predicted_rate"] = predictions

    return december_input


def main():

    print("Loading final model...")

    model = load_final_model()

    # Validation predictions

    print(
        "Generating validation predictions..."
    )

    validation_predictions = predict_validation(
        model
    )

    validation_predictions.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved validation predictions to:"
        f"\n{VALIDATION_OUTPUT_PATH}"
    )

    # December predictions

    print(
        "\nGenerating December predictions..."
    )

    december_predictions = predict_december(
        model
    )

    december_predictions.to_csv(
        DECEMBER_PATH,
        index=False,
    )

    print(
        f"Updated December file:"
        f"\n{DECEMBER_PATH}"
    )


if __name__ == "__main__":
    main()