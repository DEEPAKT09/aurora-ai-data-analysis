
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose


def validate_time_series(series):
    """
    Validate and clean a time-series pandas Series.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "Series must have a DatetimeIndex."
        )

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            "Time-series values must be numeric."
        )

    series = series.copy().sort_index()
    series = series.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(series) < 5:
        raise ValueError(
            "At least 5 valid observations are required."
        )

    if series.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps must be aggregated first."
        )

    return series


def detect_rolling_zscore(
    series,
    window=6,
    threshold=3.0
):
    """
    Detect anomalies using a rolling Z-score.

    The baseline uses only preceding observations,
    so the current value is not included in its
    own baseline.
    """

    series = validate_time_series(series)

    if window < 3:
        raise ValueError(
            "Window must be at least 3."
        )

    if len(series) <= window:
        raise ValueError(
            "Series must contain more observations than the window."
        )

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    rolling_mean = (
        series.shift(1)
        .rolling(window=window, min_periods=window)
        .mean()
    )

    rolling_std = (
        series.shift(1)
        .rolling(window=window, min_periods=window)
        .std()
    )

    safe_std = rolling_std.replace(0, np.nan)

    z_scores = (
        (series - rolling_mean) / safe_std
    )

    anomalies = z_scores.abs() >= threshold

    results = pd.DataFrame({
        "actual_value": series,
        "expected_value": rolling_mean,
        "anomaly_score": z_scores,
        "is_anomaly": anomalies.fillna(False)
    })

    return results


def detect_rolling_mad(
    series,
    window=6,
    threshold=3.5
):
    """
    Detect anomalies using rolling Median
    Absolute Deviation (MAD).

    MAD is more robust to extreme values
    than standard deviation.
    """

    series = validate_time_series(series)

    if window < 3:
        raise ValueError(
            "Window must be at least 3."
        )

    if len(series) <= window:
        raise ValueError(
            "Series must contain more observations than the window."
        )

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    previous_values = series.shift(1)

    rolling_median = (
        previous_values
        .rolling(window=window, min_periods=window)
        .median()
    )

    rolling_mad = (
        previous_values
        .rolling(window=window, min_periods=window)
        .apply(
            lambda values: np.median(
                np.abs(values - np.median(values))
            ),
            raw=True
        )
    )

    # 1.4826 makes MAD comparable to standard
    # deviation for normally distributed data.
    robust_std = 1.4826 * rolling_mad

    safe_std = robust_std.replace(0, np.nan)

    anomaly_scores = (
        (series - rolling_median) / safe_std
    )

    anomalies = (
        anomaly_scores.abs() >= threshold
    )

    results = pd.DataFrame({
        "actual_value": series,
        "expected_value": rolling_median,
        "anomaly_score": anomaly_scores,
        "is_anomaly": anomalies.fillna(False)
    })

    return results


def detect_seasonal_residual_anomalies(
    series,
    period=12,
    threshold=3.0,
    model="additive"
):
    """
    Detect anomalies after decomposing a time series
    into trend, seasonal and residual components.

    Requires at least two complete seasonal cycles.
    """

    series = validate_time_series(series)

    if period < 2:
        raise ValueError(
            "Seasonal period must be at least 2."
        )

    if len(series) < period * 2:
        raise ValueError(
            "At least two complete seasonal cycles are required."
        )

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    if model not in ["additive", "multiplicative"]:
        raise ValueError(
            "Model must be additive or multiplicative."
        )

    if model == "multiplicative" and (series <= 0).any():
        raise ValueError(
            "Multiplicative decomposition requires positive values."
        )

    decomposition = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend="period"
    )

    residuals = decomposition.resid

    residual_median = residuals.median()

    residual_mad = np.median(
        np.abs(residuals - residual_median)
    )

    robust_std = 1.4826 * residual_mad

    if robust_std == 0 or not np.isfinite(robust_std):
        raise ValueError(
            "Residual variation is insufficient for anomaly detection."
        )

    residual_scores = (
        (residuals - residual_median) / robust_std
    )

    anomalies = (
        residual_scores.abs() >= threshold
    )

    results = pd.DataFrame({
        "actual_value": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "expected_value": (
            decomposition.trend + decomposition.seasonal
            if model == "additive"
            else decomposition.trend * decomposition.seasonal
        ),
        "residual": residuals,
        "anomaly_score": residual_scores,
        "is_anomaly": anomalies.fillna(False)
    })

    return results


def run_anomaly_detection(
    df,
    date_column,
    target_column,
    method="rolling_mad",
    window=6,
    threshold=3.5,
    period=12,
    model="additive",
    frequency=None,
    aggregation="mean"
):
    """
    Main entry point for time-series anomaly detection.
    """

    if date_column not in df.columns:
        raise ValueError(
            f"Date column '{date_column}' was not found."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if method not in [
        "rolling_zscore",
        "rolling_mad",
        "seasonal_residual"
    ]:
        raise ValueError(
            "Unsupported anomaly detection method."
        )

    if aggregation not in [
        "mean", "sum", "median", "min", "max"
    ]:
        raise ValueError(
            "Unsupported aggregation method."
        )

    working_df = df[
        [date_column, target_column]
    ].copy()

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce"
    )

    working_df[target_column] = pd.to_numeric(
        working_df[target_column],
        errors="coerce"
    )

    working_df = working_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if working_df.empty:
        raise ValueError(
            "No valid date and numeric observations were found."
        )

    working_df = working_df.sort_values(date_column)

    # Aggregate duplicate timestamps.
    series = (
        working_df
        .groupby(date_column)[target_column]
        .agg(aggregation)
        .sort_index()
    )

    # Optional regular-frequency resampling.
    if frequency:
        series = series.resample(frequency).agg(aggregation)

    series = validate_time_series(series)

    if method == "rolling_zscore":
        results = detect_rolling_zscore(
            series,
            window=window,
            threshold=threshold
        )

    elif method == "rolling_mad":
        results = detect_rolling_mad(
            series,
            window=window,
            threshold=threshold
        )

    else:
        results = detect_seasonal_residual_anomalies(
            series,
            period=period,
            threshold=threshold,
            model=model
        )

    anomaly_rows = results[
        results["is_anomaly"]
    ].copy()

    anomaly_rows = anomaly_rows.reset_index()

    results = results.reset_index()

    anomaly_count = len(anomaly_rows)

    return {
        "success": True,
        "method": method,
        "target_column": target_column,
        "date_column": date_column,
        "total_observations": len(results),
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(
            anomaly_count / len(results) * 100, 2
        ),
        "results": results,
        "anomalies": anomaly_rows,
        "message": (
            f"Detected {anomaly_count} anomalies "
            f"out of {len(results)} observations "
            f"using {method}."
        )
    }
