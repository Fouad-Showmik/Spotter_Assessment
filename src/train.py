import pickle
from pathlib import Path

from catboost import CatBoostRegressor

from src.preprocessing import load_data, clean_data
from src.features import create_features, get_categorical_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "train-test.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "final_model.pkl"

#Train the selected model on all labeled development data
def train_final_model():

    # Load and clean the full labeled development data.
    df = load_data(TRAIN_PATH)
    df = clean_data(df)

    # Separate features and target.
    X = create_features(
        df.drop(columns=["posted_rate"])
    )
    y = df["posted_rate"]

    # CatBoost was selected during walk-forward validation.
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    categorical_features = get_categorical_features(X)

    # Train the final model on all labeled development data.
    model.fit(
        X,
        y,
        cat_features=categorical_features,
    )

    # Save the trained model.
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Final model: CatBoost")
    print(f"Training rows: {len(X):,}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_final_model()