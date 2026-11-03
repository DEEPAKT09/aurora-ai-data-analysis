
import numpy as np
import pandas as pd

from ml_models.forecasting_models import forecast_time_series


def create_sample_dataset():
    """Generate 60 months of sample sales data."""

    np.random.seed(42)

    dates = pd.date_range(
        start="2021-01-01",
        periods=60,
        freq="MS"
    )

    trend = np.arange(60) * 12
    seasonality = 100 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise = np.random.normal(0, 15, 60)

    sales = 500 + trend + seasonality + noise

    return pd.DataFrame({
        "Date": dates,
        "Sales": sales.round(2)
    })


def test_chronos_forecasting():
    print("\nCreating sample time-series dataset...")

    df = create_sample_dataset()

    print("\nSample data:")
    print(df.head())

    print("\nRunning Chronos-2 forecasting...")

    result = forecast_time_series(
        df=df,
        date_column="Date",
        target_column="Sales",
        prediction_length=6,
        frequency="MS",
        evaluation_size=6
    )

    print("\nCHRONOS-2 FORECASTING RESULTS")

    print("Model:", result["model"])
    print("Target:", result["target_column"])
    print("Frequency:", result["frequency"])
    print("Historical rows:", result["historical_rows"])
    print("Forecast periods:", result["prediction_length"])

    print("\nEvaluation metrics:")
    print(result["evaluation"])

    print("\nFuture forecasts:")
    print(result["forecast"].to_string(index=False))


if __name__ == "__main__":
    test_chronos_forecasting()
