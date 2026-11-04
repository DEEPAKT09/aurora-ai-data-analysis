
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    frequency: str | None = None,
):
    """Validate, aggregate, and regularize a univariate time series."""

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' was not found.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    data = data.dropna(subset=[date_column, target_column])

    data = data.replace([np.inf, -np.inf], np.nan)

    data = data.dropna(subset=[target_column])

    if data.empty:
        raise ValueError(
            "No valid date and numeric target observations were found."
        )

    # Aggregate repeated timestamps rather than rejecting the dataset.
    data = (
        data.groupby(date_column, as_index=False)[target_column]
        .mean()
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    if len(data) < 8:
        raise ValueError(
            "Holt-Winters forecasting requires at least 8 valid "
            "historical observations."
        )

    # Create the time series before attempting frequency inference.
    indexed = data.set_index(date_column)[target_column]

    # Automatically infer frequency when one is not provided.
    if frequency is None:

        inferred_frequency = pd.infer_freq(indexed.index)

        frequency = (
            inferred_frequency.freqstr
            if hasattr(inferred_frequency, "freqstr")
            else inferred_frequency
        )

    if frequency is None:

        # Infer a practical frequency from the median time difference.
        differences = indexed.index.to_series().diff().dropna()

        if differences.empty:
            raise ValueError("Could not infer the time frequency.")

        median_days = differences.median().total_seconds() / 86400

        if median_days <= 1.5:
            frequency = "D"

        elif median_days <= 8:
            frequency = "W"

        elif median_days <= 32:
            frequency = "MS"

        elif median_days <= 100:
            frequency = "QS"

        else:
            frequency = "YS"

    try:

        indexed = indexed.resample(frequency).mean()

    except Exception as error:

        raise ValueError(
            f"Unable to create a regular time series using "
            f"frequency '{frequency}': {error}"
        ) from error

    # Interpolate gaps so the model receives a continuous series.
    indexed = indexed.interpolate(method="time").ffill().bfill()

    if indexed.isna().any() or not np.isfinite(indexed.to_numpy()).all():
        raise ValueError(
            "The target series contains invalid numeric values."
        )

    if len(indexed) < 8:
        raise ValueError(
            "The regularized time series has fewer than 8 observations."
        )

    return indexed.astype(float), frequency


def _seasonal_period_for_frequency(frequency: str, row_count: int):
    """Choose a conservative seasonal period when enough history exists."""

    freq = frequency.upper()

    if freq.startswith(("M",)):
        candidate = 12
    elif freq.startswith(("Q",)):
        candidate = 4
    elif freq.startswith(("W",)):
        candidate = 52
    elif freq.startswith(("D",)):
        candidate = 7
    else:
        candidate = None

    # Require at least two complete seasonal cycles.
    if candidate and row_count >= candidate * 2:
        return candidate

    return None


def _fit_holt_winters(series: pd.Series, frequency: str):
    """Fit a lightweight Holt-Winters model with safe fallbacks."""

    seasonal_periods = _seasonal_period_for_frequency(
        frequency,
        len(series),
    )

    configurations = []

    if seasonal_periods:
        configurations.append(
            {
                "trend": "add",
                "damped_trend": True,
                "seasonal": "add",
                "seasonal_periods": seasonal_periods,
            }
        )

    configurations.extend(
        [
            {
                "trend": "add",
                "damped_trend": True,
                "seasonal": None,
                "seasonal_periods": None,
            },
            {
                "trend": None,
                "damped_trend": False,
                "seasonal": None,
                "seasonal_periods": None,
            },
        ]
    )

    errors = []

    for config in configurations:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ExponentialSmoothing(
                    series,
                    trend=config["trend"],
                    damped_trend=config["damped_trend"],
                    seasonal=config["seasonal"],
                    seasonal_periods=config["seasonal_periods"],
                    initialization_method="estimated",
                )

                fitted = model.fit(optimized=True)

            return fitted, config

        except Exception as error:
            errors.append(str(error))

    raise RuntimeError(
        "Holt-Winters could not fit this time series. "
        + " | ".join(errors[-2:])
    )


def forecast_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    prediction_length: int = 6,
    frequency: str | None = None,
    evaluation_size: int | None = None,
):
    """
    Forecast future observations with Holt-Winters Exponential Smoothing.

    The most recent observations are held out for chronological evaluation.
    The final forecast is then fitted using all available observations.
    """

    if prediction_length < 1 or prediction_length > 100:
        raise ValueError("Prediction length must be between 1 and 100.")

    series, frequency = prepare_time_series(
        df=df,
        date_column=date_column,
        target_column=target_column,
        frequency=frequency,
    )

    if evaluation_size is None:
        evaluation_size = min(
            max(1, int(round(len(series) * 0.2))),
            12,
            len(series) - 5,
        )

    if evaluation_size < 0:
        raise ValueError("Evaluation size cannot be negative.")

    evaluation = None

    if evaluation_size > 0:
        if evaluation_size >= len(series) - 4:
            raise ValueError(
                "Not enough historical observations remain for evaluation "
                "and model training."
            )

        train_series = series.iloc[:-evaluation_size]
        actual_values = series.iloc[-evaluation_size:].to_numpy()

        evaluation_model, evaluation_config = _fit_holt_winters(
            train_series,
            frequency,
        )

        predicted_values = np.asarray(
            evaluation_model.forecast(evaluation_size),
            dtype=float,
        )

        evaluation = {
            "mae": float(
                mean_absolute_error(actual_values, predicted_values)
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(actual_values, predicted_values)
                )
            ),
            "evaluation_rows": int(evaluation_size),
        }

    final_model, final_config = _fit_holt_winters(
        series,
        frequency,
    )

    forecast_values = np.asarray(
        final_model.forecast(prediction_length),
        dtype=float,
    )

    future_dates = pd.date_range(
        start=series.index[-1],
        periods=prediction_length + 1,
        freq=frequency,
    )[1:]

    # Approximate prediction bounds using historical residual variation.
    residuals = np.asarray(final_model.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    residual_std = (
        float(np.std(residuals, ddof=1))
        if len(residuals) > 1
        else 0.0
    )

    forecast_rows = []

    for step, (forecast_date, forecast_value) in enumerate(
        zip(future_dates, forecast_values),
        start=1,
    ):
        interval = 1.96 * residual_std * np.sqrt(step)

        forecast_rows.append(
            {
                "date": forecast_date,
                "forecast": float(forecast_value),
                "lower_bound": float(forecast_value - interval),
                "upper_bound": float(forecast_value + interval),
            }
        )

    return {
        "model": "Holt-Winters Exponential Smoothing",
        "task_type": "time_series_forecasting",
        "target_column": target_column,
        "date_column": date_column,
        "frequency": frequency,
        "historical_rows": int(len(series)),
        "prediction_length": int(prediction_length),
        "evaluation": evaluation,
        "model_configuration": final_config,
        "forecast": forecast_rows,
    }
