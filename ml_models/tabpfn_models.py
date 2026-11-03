import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

load_dotenv()

def train_tabpfn(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    cv_folds: int = 5
):

    if target_column not in df.columns:
        raise ValueError("Target column does not exist.")

    data = df.dropna(subset=[target_column]).copy()

    if len(data) < 20:
        raise ValueError("At least 20 usable records are required.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove constant and unsupported feature columns.
    X = X.loc[:, X.nunique(dropna=False) > 1]

    X = X.select_dtypes(
        include=["number", "object", "category", "bool"]
    )

    if X.empty:
        raise ValueError("No usable feature columns found.")

    if task_type == "regression":

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numerical target.")

        if y.nunique() < 2:
            raise ValueError("Target must have at least two values.")

        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    elif task_type == "classification":

        if y.nunique() < 2:
            raise ValueError("At least two target classes are required.")

        if y.nunique() > 10:
            raise ValueError(
                "This initial TabPFN integration supports "
                "up to 10 target classes."
            )

        if y.value_counts().min() < 2:
            raise ValueError("Each class needs at least two records.")

        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    else:
        raise ValueError(
            "Task type must be regression or classification."
        )

    # Keep TabPFN preprocessing separate from the scikit-learn pipeline.
    # TabPFN handles categorical and missing feature values itself.
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if task_type == "regression":

        metrics = {
            "mae": float(
                mean_absolute_error(y_test, predictions)
            ),
            "rmse": float(
                mean_squared_error(y_test, predictions) ** 0.5
            ),
            "r2": float(
                r2_score(y_test, predictions)
            )
        }

    else:

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
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
            )
        }

    # Cross-validation uses only the training partition, matching the
    # scikit-learn workflow. Each fold gets a fresh TabPFN model.
    cv_metrics = {}
    if task_type == "regression":
        actual_folds = min(cv_folds, len(X_train))
        splitter = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []
        for fold_train_idx, fold_valid_idx in splitter.split(X_train):
            from tabpfn import TabPFNRegressor
            fold_model = TabPFNRegressor()
            fold_model.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
            fold_predictions = fold_model.predict(X_train.iloc[fold_valid_idx])
            fold_scores.append(float(mean_squared_error(
                y_train.iloc[fold_valid_idx], fold_predictions
            ) ** 0.5))
        cv_metrics = {
            "mean": float(sum(fold_scores) / len(fold_scores)),
            "std": float(pd.Series(fold_scores).std(ddof=0)),
            "folds": len(fold_scores),
            "metric": "rmse"
        }
        score = metrics["rmse"]
    else:
        min_class_count = int(y_train.value_counts().min())
        actual_folds = min(cv_folds, min_class_count)
        if actual_folds >= 2:
            splitter = StratifiedKFold(
                n_splits=actual_folds, shuffle=True, random_state=42
            )
            fold_scores = []
            for fold_train_idx, fold_valid_idx in splitter.split(X_train, y_train):
                from tabpfn import TabPFNClassifier
                fold_model = TabPFNClassifier()
                fold_model.fit(X_train.iloc[fold_train_idx], y_train.iloc[fold_train_idx])
                fold_predictions = fold_model.predict(X_train.iloc[fold_valid_idx])
                fold_scores.append(float(f1_score(
                    y_train.iloc[fold_valid_idx], fold_predictions,
                    average="weighted", zero_division=0
                )))
            cv_metrics = {
                "mean": float(sum(fold_scores) / len(fold_scores)),
                "std": float(pd.Series(fold_scores).std(ddof=0)),
                "folds": len(fold_scores),
                "metric": "f1_weighted"
            }
        else:
            cv_metrics = {
                "error": "Insufficient examples per class for stratified cross-validation."
            }
        score = metrics["f1_weighted"]

    return {
        "model": "TabPFN",
        "task_type": task_type,
        "target_column": target_column,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metrics": metrics,
        "cross_validation": cv_metrics,
        "score": float(score),
        "predictions": predictions.tolist(),
        "actual_values": y_test.tolist()
    }