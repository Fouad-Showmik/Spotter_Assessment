import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor


CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

#Predict the median training rate for every load
class MedianBaseline:

    def __init__(self):
        self.median_rate = None

    def fit(self, X, y):
        self.median_rate = float(np.median(y))
        return self

    def predict(self, X):
        if self.median_rate is None:
            raise ValueError("Model has not been fitted yet.")

        return np.full(len(X), self.median_rate)

#LightGBM regression model with training-data-based categorical encoding.

#Numerical missing values are left as NaN because
#LightGBM can handle them natively.


class LightGBMModel:

    def __init__(self):
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        self.category_maps = {}

#Create category mappings using training data only
    def _fit_category_maps(self, X):

        for column in CATEGORICAL_FEATURES:
            if column not in X.columns:
                continue

            categories = pd.Index(
                X[column].astype(str).unique()
            )

            self.category_maps[column] = {
                category: index
                for index, category in enumerate(categories)
            }

#Convert categorical values to training-based integer codes.
    def _transform(self, X):

        X = X.copy()

        for column, mapping in self.category_maps.items():
            X[column] = (
                X[column]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )

        return X

    def fit(self, X, y):
        self._fit_category_maps(X)

        X_encoded = self._transform(X)

        categorical_features = [
            column
            for column in CATEGORICAL_FEATURES
            if column in X_encoded.columns
        ]

        self.model.fit(
            X_encoded,
            y,
            categorical_feature=categorical_features,
        )

        return self

    def predict(self, X):
        X_encoded = self._transform(X)

        return self.model.predict(X_encoded)


#Create the LightGBM model
def create_lightgbm():

    return LightGBMModel()

#Create the CatBoost regression model
def create_catboost():

    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5,
        random_seed=42,
        early_stopping_rounds=100,
        verbose=False,
        allow_writing_files=False,
    )


