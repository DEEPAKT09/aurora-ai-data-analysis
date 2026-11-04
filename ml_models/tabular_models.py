
import math
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from ml_models.preprocessing import prepare_features


# --------------------------------------------------
# 1. Model catalogues
# --------------------------------------------------

def get_regression_models():

    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=5
        )
    }


def get_classification_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }


# --------------------------------------------------
# 2. Leakage and feature warnings
# --------------------------------------------------

def inspect_features(X, y):

    warnings = []

    exact_target_copies = []

    identifier_columns = []

    high_correlation_columns = []

    # Detect exact copies of the target.
    for column in X.columns:

        if X[column].equals(y):
            exact_target_copies.append(str(column))

    if exact_target_copies:

        warnings.append(
            "The following features are exact copies "
            f"of the target: {exact_target_copies}. "
            "Investigate and remove them if they expose "
            "the answer."
        )

    # Detect potentially identifier-like columns.
    for column in X.columns:

        series = X[column]

        unique_ratio = (
            series.nunique(dropna=True) /
            max(len(series), 1)
        )

        column_name = str(column).lower()

        identifier_name = any(
            term in column_name
            for term in [
                "id",
                "uuid",
                "index",
                "identifier",
                "record_key"
            ]
        )

        if unique_ratio >= 0.95 and identifier_name:

            identifier_columns.append(str(column))

    if identifier_columns:

        warnings.append(
            "Potential identifier columns detected: "
            f"{identifier_columns}. "
            "Check whether these columns provide "
            "meaningful predictive information."
        )

    # Detect highly correlated numerical features.
    numerical = X.select_dtypes(
        include=["number"]
    )

    if numerical.shape[1] >= 2:

        correlation = numerical.corr().abs()

        for i in range(len(correlation.columns)):

            for j in range(i + 1, len(correlation.columns)):

                first = correlation.columns[i]
                second = correlation.columns[j]

                value = correlation.iloc[i, j]

                if pd.notna(value) and value >= 0.98:

                    high_correlation_columns.append({
                        "feature_1": str(first),
                        "feature_2": str(second),
                        "correlation": float(value)
                    })

    if high_correlation_columns:

        warnings.append(
            "Highly correlated numerical features detected. "
            "Investigate whether they are redundant or "
            "derived from the target."
        )

    return {
        "warnings": warnings,
        "exact_target_copies": exact_target_copies,
        "identifier_columns": identifier_columns,
        "high_correlation_features": high_correlation_columns
    }


# --------------------------------------------------
# 3. Main training and comparison function
# --------------------------------------------------

def train_and_compare_models(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    selected_models: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5
):

    # ----------------------------------------------
    # Validate inputs
    # ----------------------------------------------

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if task_type not in [
        "regression",
        "classification"
    ]:

        raise ValueError(
            "Task type must be regression or classification."
        )

    if not 0.1 <= test_size <= 0.4:

        raise ValueError(
            "Test size must be between 0.1 and 0.4."
        )

    if not 2 <= cv_folds <= 10:

        raise ValueError(
            "Cross-validation folds must be between 2 and 10."
        )

    # ----------------------------------------------
    # Prepare dataset
    # ----------------------------------------------

    data = df.dropna(
        subset=[target_column]
    ).copy()

    if len(data) < 20:

        raise ValueError(
            "At least 20 usable rows are required."
        )

    X = data.drop(
        columns=[target_column]
    ).copy()

    y = data[target_column].copy()

    # Convert infinite numerical features to missing.
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns

    X[numerical_columns] = X[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove constant features.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=False) <= 1
    ]

    X = X.drop(
        columns=constant_columns
    )

    # Retain supported data types.
    supported_columns = X.select_dtypes(
    include=["number", "object", "str", "category", "bool"]
).columns.tolist()

    unsupported_columns = [
        str(column)
        for column in X.columns
        if column not in supported_columns
    ]

    X = X[supported_columns]

    if X.empty:

        raise ValueError(
            "No supported feature columns remain."
        )

    # ----------------------------------------------
    # Inspect potential leakage
    # ----------------------------------------------

    feature_inspection = inspect_features(X, y)

    # Exact copies of the target cannot be legitimate
    # independent predictors. Exclude them automatically.
    exact_target_copies = (
        feature_inspection["exact_target_copies"]
    )

    if exact_target_copies:

        X = X.drop(
            columns=exact_target_copies
        )

    if X.empty:

        raise ValueError(
            "All available features were exact copies "
            "of the target or otherwise unsupported."
        )

    # ----------------------------------------------
    # Validate target and select models
    # ----------------------------------------------

    if task_type == "regression":

        if not pd.api.types.is_numeric_dtype(y):

            raise ValueError(
                "Regression requires a numerical target."
            )

        if not np.isfinite(
            pd.to_numeric(y, errors="coerce")
        ).all():

            raise ValueError(
                "Regression target contains infinite "
                "or invalid values."
            )

        if y.nunique() < 2:

            raise ValueError(
                "The target must contain at least "
                "two different values."
            )

        model_catalogue = get_regression_models()

    else:

        if y.nunique() < 2:

            raise ValueError(
                "Classification requires at least two classes."
            )

        if y.nunique() > 50:

            raise ValueError(
                "This classification engine supports "
                "up to 50 classes."
            )

        class_counts = y.value_counts()

        if class_counts.min() < 2:

            raise ValueError(
                "Every target class must have at least "
                "two records."
            )

        model_catalogue = get_classification_models()

    if selected_models:

        unknown_models = (
            set(selected_models) - set(model_catalogue)
        )

        if unknown_models:

            raise ValueError(
                f"Unsupported model selections: {unknown_models}"
            )

        model_catalogue = {
            name: model
            for name, model in model_catalogue.items()
            if name in selected_models
        }

    if not model_catalogue:

        raise ValueError(
            "Please select at least one model."
        )

    # ----------------------------------------------
    # Split training and test data
    # ----------------------------------------------

    if task_type == "classification":

        number_of_classes = y.nunique()

        test_rows = math.ceil(
            len(y) * test_size
        )

        if test_rows < number_of_classes:

            raise ValueError(
                "The test split is too small to contain "
                "at least one example of every class. "
                "Increase test_size or collect more data."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    if len(X_train) < 5:

        raise ValueError(
            "Not enough training records."
        )

    # ----------------------------------------------
    # Configure cross-validation
    # ----------------------------------------------

    if task_type == "regression":

        actual_folds = min(
            cv_folds,
            len(X_train)
        )

        cross_validator = KFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=42
        )

        scoring = "neg_root_mean_squared_error"

    else:

        minimum_class_count = int(
            y_train.value_counts().min()
        )

        actual_folds = min(
            cv_folds,
            minimum_class_count
        )

        cross_validator = None

        if actual_folds >= 2:

            cross_validator = StratifiedKFold(
                n_splits=actual_folds,
                shuffle=True,
                random_state=42
            )

        scoring = "f1_weighted"

    # ----------------------------------------------
    # Train and evaluate models
    # ----------------------------------------------

    results = []

    for model_name, estimator in model_catalogue.items():

        try:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        prepare_features(X_train)
                    ),
                    (
                        "model",
                        clone(estimator)
                    )
                ]
            )

            # Cross-validation on training data only.
            cv_metrics = {}

            if cross_validator is not None:

                try:

                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cross_validator,
                        scoring=scoring,
                        n_jobs=1,
                        error_score="raise"
                    )

                    if task_type == "regression":

                        cv_scores = -cv_scores

                    cv_metrics = {
                        "mean": float(np.mean(cv_scores)),
                        "std": float(np.std(cv_scores)),
                        "folds": int(len(cv_scores))
                    }

                except Exception as cv_error:

                    cv_metrics = {
                        "error": str(cv_error)
                    }

            else:

                cv_metrics = {
                    "error": (
                        "Insufficient examples per class "
                        "for stratified cross-validation."
                    )
                }

            # Fit final model on complete training split.
            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            # --------------------------------------
            # Regression metrics
            # --------------------------------------

            if task_type == "regression":

                metrics = {
                    "mae": float(
                        mean_absolute_error(
                            y_test,
                            predictions
                        )
                    ),

                    "rmse": float(
                        mean_squared_error(
                            y_test,
                            predictions
                        ) ** 0.5
                    ),

                    "r2": float(
                        r2_score(
                            y_test,
                            predictions
                        )
                    )
                }

                score = metrics["rmse"]

            # --------------------------------------
            # Classification metrics
            # --------------------------------------

            else:

                class_labels = sorted(
                    y.unique().tolist(),
                    key=str
                )

                metrics = {
                    "accuracy": float(
                        accuracy_score(
                            y_test,
                            predictions
                        )
                    ),

                    "precision_weighted": float(
                        precision_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                    "recall_weighted": float(
                        recall_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                    "f1_weighted": float(
                        f1_score(
                            y_test,
                            predictions,
                            average="weighted",
                            zero_division=0
                        )
                    ),

                    "confusion_matrix": confusion_matrix(
                        y_test,
                        predictions,
                        labels=class_labels
                    ).tolist()
                }

                score = metrics["f1_weighted"]

            # --------------------------------------
            # Store model result
            # --------------------------------------

            model_result = {
                "model": model_name,
                "metrics": metrics,
                "cross_validation": cv_metrics,
                "score": float(score),

                "predictions": [
                    value.item()
                    if hasattr(value, "item")
                    else value
                    for value in predictions.tolist()
                ],

                "actual_values": [
                    value.item()
                    if hasattr(value, "item")
                    else value
                    for value in y_test.tolist()
                ],

                "test_rows": int(len(y_test))
            }

            if task_type == "classification":

                model_result["class_labels"] = [
                    value.item()
                    if hasattr(value, "item")
                    else value
                    for value in class_labels
                ]

            results.append(model_result)

        except Exception as error:

            results.append({
                "model": model_name,
                "error": str(error)
            })

    # ----------------------------------------------
    # Identify successful models
    # ----------------------------------------------

    successful_results = [
        result
        for result in results
        if "error" not in result
    ]

    if not successful_results:

        raise RuntimeError(
            "All selected models failed. "
            f"Details: {results}"
        )

    # Lower RMSE is better.
    # Higher weighted F1 is better.
    successful_results.sort(
        key=lambda item: (
            item["score"]
            if task_type == "regression"
            else -item["score"]
        )
    )

    # ----------------------------------------------
    # Final response
    # ----------------------------------------------

    return {
        "task_type": task_type,
        "target_column": target_column,

        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),

        "cross_validation_folds": (
            actual_folds
            if cross_validator is not None
            else 0
        ),

        "removed_constant_columns": [
            str(column)
            for column in constant_columns
        ],

        "removed_exact_target_copies": exact_target_copies,

        "unsupported_columns": unsupported_columns,

        "feature_warnings": feature_inspection["warnings"],

        "feature_inspection": feature_inspection,

        "best_model": successful_results[0]["model"],

        "models": results
    }
