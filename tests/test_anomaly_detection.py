
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.anomaly_detection_tools import (
    detect_rolling_zscore,
    detect_rolling_mad,
    detect_seasonal_residual_anomalies,
    run_anomaly_detection
)


def create_anomaly_sample():
    """
    Generate 60 monthly sales observations
    with two deliberately injected anomalies.
    """

    np.random.seed(42)

    dates = pd.date_range(
        start="2021-01-01",
        periods=60,
        freq="MS"
    )

    trend = np.arange(60) * 10

    seasonality = (
        100 * np.sin(
            2 * np.pi * np.arange(60) / 12
        )
    )

    noise = np.random.normal(0, 8, 60)

    sales = 1000 + trend + seasonality + noise

    # Inject an unusually high sales value.
    sales[30] += 500

    # Inject an unusually low sales value.
    sales[45] -= 450

    return pd.DataFrame({
        "Date": dates,
        "Sales": sales
    })


def test_rolling_zscore():

    df = create_anomaly_sample()

    series = df.set_index("Date")["Sales"]

    result = detect_rolling_zscore(
        series,
        window=6,
        threshold=3.0
    )

    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].dtype == bool

    print("\nROLLING Z-SCORE")
    print(result[result["is_anomaly"]])


def test_rolling_mad():

    df = create_anomaly_sample()

    series = df.set_index("Date")["Sales"]

    result = detect_rolling_mad(
        series,
        window=6,
        threshold=3.5
    )

    assert "is_anomaly" in result.columns
    assert result["is_anomaly"].dtype == bool

    print("\nROLLING MAD")
    print(result[result["is_anomaly"]])


def test_seasonal_residual():

    df = create_anomaly_sample()

    series = df.set_index("Date")["Sales"]

    result = detect_seasonal_residual_anomalies(
        series,
        period=12,
        threshold=3.0
    )

    assert "trend" in result.columns
    assert "seasonal" in result.columns
    assert "residual" in result.columns

    print("\nSEASONAL RESIDUAL")
    print(result[result["is_anomaly"]])


def test_complete_anomaly_analysis():

    df = create_anomaly_sample()

    result = run_anomaly_detection(
        df=df,
        date_column="Date",
        target_column="Sales",
        method="rolling_mad",
        window=6,
        threshold=3.5
    )

    assert result["success"] is True
    assert result["total_observations"] == 60
    assert result["anomaly_count"] >= 1

    print("\nCOMPLETE ANOMALY ANALYSIS")
    print(result["message"])
    print(result["anomalies"])


if __name__ == "__main__":

    test_rolling_zscore()
    test_rolling_mad()
    test_seasonal_residual()
    test_complete_anomaly_analysis()

    print("\nAll anomaly detection tests completed.")
