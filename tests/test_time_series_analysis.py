
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from tools.time_series_analysis import (
    prepare_time_series,
    analyze_trend,
    analyze_seasonality,
    summarize_time_period,
    run_time_series_analysis
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_datasets"
    / "time_series_sample.csv"
)


def load_data():
    return pd.read_csv(DATA_PATH)


def test_prepare_time_series():

    df = load_data()

    series = prepare_time_series(
        df,
        date_column="Date",
        target_column="Sales"
    )

    assert len(series) == 60
    assert isinstance(series.index, pd.DatetimeIndex)

    print("\nTIME SERIES PREPARATION")
    print(series.head())


def test_trend_analysis():

    df = load_data()

    series = prepare_time_series(
        df,
        "Date",
        "Sales"
    )

    result = analyze_trend(series)

    assert result["success"] is True
    assert result["slope_per_period"] > 0

    print("\nTREND ANALYSIS")
    print(result)


def test_seasonality():

    df = load_data()

    series = prepare_time_series(
        df,
        "Date",
        "Sales"
    )

    result = analyze_seasonality(
        series,
        period=12
    )

    assert result["success"] is True
    assert result["period"] == 12

    print("\nSEASONALITY ANALYSIS")
    print("Seasonal strength:", result["seasonal_strength"])


def test_period_summary():

    df = load_data()

    series = prepare_time_series(
        df,
        "Date",
        "Sales"
    )

    result = summarize_time_period(
        series,
        period_type="month"
    )

    assert result["success"] is True
    assert len(result["summary"]) == 12

    print("\nMONTHLY SUMMARY")
    print(pd.DataFrame(result["summary"]))


def test_complete_analysis():

    df = load_data()

    result = run_time_series_analysis(
        df,
        date_column="Date",
        target_column="Sales",
        seasonal_period=12,
        summary_period="month"
    )

    assert result["success"] is True
    assert result["trend"]["success"] is True
    assert result["seasonality"]["success"] is True

    print("\nCOMPLETE TIME-SERIES ANALYSIS")
    print("Trend:", result["trend"]["direction"])
    print("Slope:", result["trend"]["slope_per_period"])
    print("Seasonal strength:", result["seasonality"]["seasonal_strength"])
    print("Highest month:", result["period_summary"]["highest_average_period"])
    print("Lowest month:", result["period_summary"]["lowest_average_period"])


if __name__ == "__main__":

    test_prepare_time_series()
    test_trend_analysis()
    test_seasonality()
    test_period_summary()
    test_complete_analysis()

    print("\nAll time-series analysis tests passed!")
