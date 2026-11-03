
from functools import lru_cache

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


@lru_cache(maxsize=1)
def load_chronos_model():
    """
    Load Chronos-2 once and reuse it for subsequent forecasts.
    CPU mode is used because this project currently has no CUDA.
    """
    return Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu"
    )


def prepare_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    frequency: str | None = None
):
    """Validate and prepare a univariate time-series dataset."""

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' was not found.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.dropna().sort_values(date_column).reset_index(drop=True)

    if len(data) < 20:
        raise ValueError(
            "Chronos-2 requires more historical observations. "
            "Please provide at least 20 valid rows."
        )

    if data[date_column].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. Aggregate them before forecasting."
        )

    if not np.isfinite(data[target_column]).all():
        raise ValueError("Target column contains infinite values.")

    if frequency is None:
        frequency = pd.infer_freq(data[date_column])

    if frequency is None:
        raise ValueError(
            "Could not identify a regular time frequency. "
            "Please check for missing dates or specify a frequency."
        )

    return data, frequency


def _run_chronos_prediction(
    model,
    data,
    date_column,
    target_column,
    prediction_length,
    frequency
):
    """Run Chronos-2 for one time series."""

    context = pd.DataFrame({
        "item_id": "series_1",
        "timestamp": data[date_column],
        "target": data[target_column]
    })

    forecasts = model.predict_df(
        context,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        freq=frequency
    )

    # Rename the output columns into consistent names.
    quantile_columns = {}

    for column in forecasts.columns:
        column_name = str(column)

        if column_name in ("0.1", "0.1"):
            quantile_columns[column] = "lower_bound"
        elif column_name in ("0.5", "0.5"):
            quantile_columns[column] = "forecast"
        elif column_name in ("0.9", "0.9"):
            quantile_columns[column] = "upper_bound"

    forecasts = forecasts.rename(columns=quantile_columns)

    required_columns = [
        "timestamp",
        "predictions",
        "lower_bound",
        "forecast",
        "upper_bound"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in forecasts.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Unexpected Chronos output. Missing columns: {missing_columns}. "
            f"Received: {forecasts.columns.tolist()}"
        )

    forecasts = forecasts.rename(
        columns={
            "timestamp": "date",
            "predictions": "mean_forecast"
        }
    )

    return forecasts[
        [
            "date",
            "mean_forecast",
            "lower_bound",
            "forecast",
            "upper_bound"
        ]
    ]


def forecast_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    prediction_length: int = 6,
    frequency: str | None = None,
    evaluation_size: int = 6
):
    """
    Generate future forecasts and evaluate the model using
    a chronological holdout split.
    """

    if prediction_length < 1:
        raise ValueError("Prediction length must be at least 1.")

    data, frequency = prepare_time_series(
        df,
        date_column,
        target_column,
        frequency
    )

    if evaluation_size < 0:
        raise ValueError("Evaluation size cannot be negative.")

    if evaluation_size >= len(data) - 12:
        raise ValueError(
            "Evaluation size is too large. Keep at least 12 "
            "observations for training."
        )

    model = load_chronos_model()

    evaluation = None

    # Evaluate using the most recent historical observations.
    if evaluation_size > 0:
        training_data = data.iloc[:-evaluation_size].copy()
        actual_values = data.iloc[-evaluation_size:][
            target_column
        ].to_numpy()

        evaluation_forecasts = _run_chronos_prediction(
            model,
            training_data,
            date_column,
            target_column,
            evaluation_size,
            frequency
        )

        predicted_values = evaluation_forecasts[
            "forecast"
        ].to_numpy()

        evaluation = {
            "mae": float(
                mean_absolute_error(actual_values, predicted_values)
            ),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(actual_values, predicted_values)
                )
            ),
            "evaluation_rows": int(evaluation_size)
        }

    # Generate the actual future forecast using all available history.
    future_forecasts = _run_chronos_prediction(
        model,
        data,
        date_column,
        target_column,
        prediction_length,
        frequency
    )

    return {
        "model": "Chronos-2",
        "task_type": "time_series_forecasting",
        "target_column": target_column,
        "date_column": date_column,
        "frequency": frequency,
        "historical_rows": len(data),
        "prediction_length": prediction_length,
        "evaluation": evaluation,
        "forecast": future_forecasts
    }
