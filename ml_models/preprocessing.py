
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(X: pd.DataFrame):
    """
    Prepare numerical and categorical features.

    All preprocessing is fitted within the model pipeline,
    preventing information from the test or validation data
    from leaking into training.
    """

    X = X.copy()

    # Convert infinite numerical values into missing values.
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if numerical_columns:
        X[numerical_columns] = X[numerical_columns].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Identify supported feature types.
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

    transformers = []

    # Numerical preprocessing.
    if numerical_columns:

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    # Categorical preprocessing.
    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        raise ValueError(
            "No supported numerical or categorical "
            "feature columns were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
