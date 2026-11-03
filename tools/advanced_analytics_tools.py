
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


def _validate_numeric_columns(df, columns=None):
    """
    Validate and return numerical columns for analysis.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    if columns is None:
        columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    else:
        missing = [
            column for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Columns not found: {missing}"
            )

        non_numeric = [
            column for column in columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        if non_numeric:
            raise ValueError(
                f"These columns are not numerical: {non_numeric}"
            )

    if not columns:
        raise ValueError(
            "No numerical columns are available for outlier detection."
        )

    return columns


def detect_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range method.

    Lower bound = Q1 - multiplier * IQR
    Upper bound = Q3 + multiplier * IQR
    """

    if multiplier <= 0:
        raise ValueError("The IQR multiplier must be positive.")

    columns = _validate_numeric_columns(df, columns)

    results = {}

    for column in columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        mask = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        mask = mask.fillna(False)

        outlier_indices = df.index[mask].tolist()

        results[column] = {
            "method": "IQR",
            "q1": q1,
            "q3": q3,
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(mask.sum()),
            "outlier_percentage": round(
                float(mask.mean() * 100), 2
            ),
            "outlier_indices": [
                str(index) for index in outlier_indices
            ],
            "outlier_values": [
                float(value)
                for value in df.loc[mask, column].tolist()
            ]
        }

    return {
        "success": True,
        "method": "IQR",
        "total_rows": len(df),
        "results": results
    }


def detect_outliers_zscore(
    df,
    columns=None,
    threshold=3.0
):
    """
    Detect outliers using absolute Z-score.

    A value is flagged when its absolute Z-score
    exceeds the specified threshold.
    """

    if threshold <= 0:
        raise ValueError("The Z-score threshold must be positive.")

    columns = _validate_numeric_columns(df, columns)

    results = {}

    for column in columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid = series.dropna()

        if len(valid) < 2:
            continue

        standard_deviation = float(valid.std(ddof=0))

        if standard_deviation == 0:
            continue

        mean = float(valid.mean())

        z_scores = (series - mean) / standard_deviation

        mask = z_scores.abs() > threshold
        mask = mask.fillna(False)

        results[column] = {
            "method": "Z-score",
            "mean": mean,
            "standard_deviation": standard_deviation,
            "threshold": float(threshold),
            "outlier_count": int(mask.sum()),
            "outlier_percentage": round(
                float(mask.sum() / len(df) * 100), 2
            ),
            "outlier_indices": [
                str(index)
                for index in df.index[mask].tolist()
            ],
            "outlier_values": [
                float(value)
                for value in df.loc[mask, column].tolist()
            ]
        }

    return {
        "success": True,
        "method": "Z-score",
        "total_rows": len(df),
        "results": results
    }


def detect_outliers_isolation_forest(
    df,
    columns=None,
    contamination=0.05,
    random_state=42
):
    """
    Detect multivariate outliers using Isolation Forest.

    The algorithm considers combinations of numerical
    features rather than examining each column separately.
    """

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be greater than 0 and at most 0.5."
        )

    columns = _validate_numeric_columns(df, columns)

    X = df[columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns with no usable values.
    X = X.dropna(axis=1, how="all")

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "No usable numerical features were found."
        )

    # Remove constant columns.
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] == 0:
        raise ValueError(
            "All selected numerical features are constant."
        )

    if len(X) < 10:
        raise ValueError(
            "Isolation Forest requires at least 10 rows "
            "for this application."
        )

    imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(X_imputed)

    anomaly_scores = model.decision_function(X_imputed)

    outlier_mask = predictions == -1

    outlier_indices = df.index[outlier_mask].tolist()

    outlier_records = df.loc[
        outlier_mask,
        X.columns
    ].copy()

    outlier_records.insert(
        0,
        "original_index",
        [str(index) for index in outlier_indices]
    )

    return {
        "success": True,
        "method": "Isolation Forest",
        "total_rows": len(df),
        "features_used": X.columns.tolist(),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_percentage": round(
            float(outlier_mask.mean() * 100), 2
        ),
        "outlier_indices": [
            str(index) for index in outlier_indices
        ],
        "anomaly_scores": [
            float(score)
            for score in anomaly_scores[outlier_mask]
        ],
        "outlier_records": outlier_records.to_dict(
            orient="records"
        )
    }


def run_outlier_analysis(
    df,
    columns=None,
    iqr_multiplier=1.5,
    zscore_threshold=3.0,
    contamination=0.05
):
    """
    Run all three outlier detection methods.

    Returns independent results for IQR, Z-score
    and Isolation Forest.
    """

    return {
        "success": True,
        "dataset_rows": len(df),
        "iqr": detect_outliers_iqr(
            df,
            columns=columns,
            multiplier=iqr_multiplier
        ),
        "zscore": detect_outliers_zscore(
            df,
            columns=columns,
            threshold=zscore_threshold
        ),
        "isolation_forest": detect_outliers_isolation_forest(
            df,
            columns=columns,
            contamination=contamination
        )
    }
