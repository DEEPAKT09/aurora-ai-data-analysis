
import numpy as np
import pandas as pd

from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose


def prepare_time_series(
    df,
    date_column,
    target_column,
    frequency=None,
    aggregation="mean"
):
    """
    Prepare a time series by parsing dates, sorting observations,
    aggregating duplicate dates and optionally resampling.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    if date_column not in df.columns:
        raise ValueError(f"Date column not found: {date_column}")

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(
            f"Target column '{target_column}' must be numerical."
        )

    if aggregation not in ["mean", "sum", "median", "min", "max"]:
        raise ValueError(
            "Aggregation must be mean, sum, median, min or max."
        )

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if data.empty:
        raise ValueError(
            "No valid date and numerical observations were found."
        )

    data = data.sort_values(date_column)

    # Aggregate duplicate timestamps.
    data = data.groupby(
        date_column,
        as_index=False
    )[target_column].agg(aggregation)

    series = data.set_index(date_column)[target_column]

    # Resample only when the user explicitly provides a frequency.
    if frequency:
        series = series.resample(frequency).agg(aggregation)

    if series.empty:
        raise ValueError("No observations remain after preparation.")

    return series


def analyze_trend(series):
    """
    Calculate the linear trend and its statistical significance.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("Expected a Pandas Series.")

    series = series.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(series) < 3:
        raise ValueError(
            "At least three observations are required for trend analysis."
        )

    x = np.arange(len(series))
    y = series.to_numpy(dtype=float)

    result = linregress(x, y)

    slope = float(result.slope)
    p_value = float(result.pvalue)

    average = float(np.mean(y))

    if average != 0:
        percentage_change_per_period = (
            slope / abs(average)
        ) * 100
    else:
        percentage_change_per_period = None

    if p_value < 0.05 and slope > 0:
        direction = "Increasing"
    elif p_value < 0.05 and slope < 0:
        direction = "Decreasing"
    else:
        direction = "No statistically significant linear trend"

    return {
        "success": True,
        "observations": len(series),
        "direction": direction,
        "slope_per_period": slope,
        "percentage_change_per_period": (
            float(percentage_change_per_period)
            if percentage_change_per_period is not None
            else None
        ),
        "r_squared": float(result.rvalue ** 2),
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
        "first_value": float(y[0]),
        "last_value": float(y[-1]),
        "total_change": float(y[-1] - y[0]),
        "mean": average,
        "minimum": float(np.min(y)),
        "maximum": float(np.max(y))
    }


def analyze_seasonality(
    series,
    period,
    model="additive"
):
    """
    Decompose a time series into trend, seasonal and residual components.

    At least two complete seasonal cycles are required.
    """

    if not isinstance(series, pd.Series):
        raise TypeError("Expected a Pandas Series.")

    if not isinstance(period, int) or period < 2:
        raise ValueError(
            "Period must be an integer greater than or equal to 2."
        )

    if model not in ["additive", "multiplicative"]:
        raise ValueError(
            "Model must be additive or multiplicative."
        )

    series = series.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(series) < 2 * period:
        raise ValueError(
            f"At least {2 * period} observations are required "
            f"for period {period}."
        )

    if model == "multiplicative" and (series <= 0).any():
        raise ValueError(
            "Multiplicative decomposition requires strictly "
            "positive observations."
        )

    decomposition = seasonal_decompose(
        series,
        model=model,
        period=period,
        extrapolate_trend="period"
    )

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    seasonal_strength = None

    valid_residual = residual.dropna()
    valid_trend = trend.dropna()

    if len(valid_residual) > 1 and len(valid_trend) > 1:

        residual_variance = float(valid_residual.var())
        combined_variance = float(
            (valid_residual + seasonal.loc[
                valid_residual.index
            ]).var()
        )

        if combined_variance > 0:
            seasonal_strength = max(
                0.0,
                min(
                    1.0,
                    1 - residual_variance / combined_variance
                )
            )

    components = pd.DataFrame({
        "observed": series,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual
    })

    return {
        "success": True,
        "model": model,
        "period": period,
        "observations": len(series),
        "seasonal_strength": seasonal_strength,
        "components": components.replace(
            [np.inf, -np.inf],
            np.nan
        ).to_dict(orient="index")
    }


def summarize_time_period(
    series,
    period_type="month"
):
    """
    Summarize average and total values by calendar period.

    Supports month, weekday, quarter and year.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "The time series must have a DatetimeIndex."
        )

    period_type = period_type.lower()

    if period_type == "month":
        labels = series.index.month_name()
        sort_values = series.index.month

    elif period_type == "weekday":
        labels = series.index.day_name()
        sort_values = series.index.dayofweek

    elif period_type == "quarter":
        labels = (
            "Q" + series.index.quarter.astype(str)
        )
        sort_values = series.index.quarter

    elif period_type == "year":
        labels = series.index.year.astype(str)
        sort_values = series.index.year

    else:
        raise ValueError(
            "period_type must be month, weekday, quarter or year."
        )

    summary = pd.DataFrame({
        "period": labels,
        "sort_order": sort_values,
        "value": series.to_numpy()
    })

    result = summary.groupby(
        ["sort_order", "period"],
        as_index=False
    ).agg(
        average=("value", "mean"),
        total=("value", "sum"),
        count=("value", "count")
    )

    result = result.sort_values(
        "sort_order"
    ).drop(columns=["sort_order"])

    highest = result.loc[
        result["average"].idxmax()
    ]

    lowest = result.loc[
        result["average"].idxmin()
    ]

    return {
        "success": True,
        "period_type": period_type,
        "summary": result.to_dict(orient="records"),
        "highest_average_period": str(highest["period"]),
        "highest_average": float(highest["average"]),
        "lowest_average_period": str(lowest["period"]),
        "lowest_average": float(lowest["average"])
    }


def run_time_series_analysis(
    df,
    date_column,
    target_column,
    frequency=None,
    aggregation="mean",
    seasonal_period=None,
    decomposition_model="additive",
    summary_period="month"
):
    """
    Run trend, optional seasonality and period summary analysis.
    """

    series = prepare_time_series(
        df,
        date_column,
        target_column,
        frequency=frequency,
        aggregation=aggregation
    )

    trend_result = analyze_trend(series)

    summary_result = summarize_time_period(
        series,
        period_type=summary_period
    )

    seasonality_result = None
    seasonality_error = None

    if seasonal_period is not None:

        try:
            seasonality_result = analyze_seasonality(
                series,
                period=seasonal_period,
                model=decomposition_model
            )

        except ValueError as error:
            seasonality_error = str(error)

    return {
        "success": True,
        "date_column": date_column,
        "target_column": target_column,
        "frequency": frequency,
        "aggregation": aggregation,
        "observations": len(series),
        "start_date": str(series.index.min()),
        "end_date": str(series.index.max()),
        "trend": trend_result,
        "seasonality": seasonality_result,
        "seasonality_error": seasonality_error,
        "period_summary": summary_result
    }
