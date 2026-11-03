
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.insight_generation_tools import (
    generate_data_quality_insights,
    generate_outlier_insights,
    generate_correlation_insights,
    generate_hypothesis_insights,
    generate_trend_insights,
    generate_anomaly_insights,
    generate_automated_insights
)


def create_sample_dataset():

    np.random.seed(42)

    dates = pd.date_range(
        start="2021-01-01",
        periods=60,
        freq="MS"
    )

    sales = (
        1000
        + np.arange(60) * 10
        + 100 * np.sin(
            2 * np.pi * np.arange(60) / 12
        )
        + np.random.normal(0, 8, 60)
    )

    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Profit": sales * 0.25,
        "Region": np.random.choice(
            ["India", "USA", "UK"],
            size=60
        )
    })

    # Add a missing value and a duplicate row
    # to test data-quality insights.
    df.loc[5, "Profit"] = np.nan
    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True
    )

    return df


def test_data_quality_insights():

    df = create_sample_dataset()

    insights = generate_data_quality_insights(df)

    assert len(insights) >= 2

    print("\nDATA QUALITY INSIGHTS")

    for insight in insights:
        print(insight["title"])


def test_outlier_insights():

    result = {
        "outlier_count": 5,
        "outlier_percentage": 8.33,
        "method": "IQR"
    }

    insights = generate_outlier_insights(result)

    assert len(insights) == 1
    assert insights[0]["category"] == "outlier"

    print("\nOUTLIER INSIGHTS")
    print(insights)


def test_correlation_insights():

    result = {
        "significant_pairs": [
            {
                "column_1": "Sales",
                "column_2": "Profit",
                "correlation": 0.95,
                "p_value": 0.001
            }
        ]
    }

    insights = generate_correlation_insights(result)

    assert len(insights) == 1

    print("\nCORRELATION INSIGHTS")
    print(insights)


def test_hypothesis_insights():

    result = {
        "test_name": "Independent t-test",
        "p_value": 0.002
    }

    insights = generate_hypothesis_insights(result)

    assert len(insights) == 1

    assert (
        insights[0]["evidence"]["statistically_significant"]
        is True
    )

    print("\nHYPOTHESIS TESTING INSIGHTS")
    print(insights)


def test_trend_insights():

    result = {
        "trend": {
            "direction": "increasing",
            "slope": 10.2,
            "p_value": 0.001,
            "r_squared": 0.91,
            "total_change": 500
        },
        "seasonality": {
            "seasonal_strength": 0.75
        }
    }

    insights = generate_trend_insights(result)

    assert len(insights) == 2

    print("\nTREND AND SEASONALITY INSIGHTS")

    for insight in insights:
        print(insight["title"])


def test_anomaly_insights():

    result = {
        "anomaly_count": 3,
        "total_observations": 60,
        "anomaly_percentage": 5.0,
        "method": "rolling_mad"
    }

    insights = generate_anomaly_insights(result)

    assert len(insights) == 1

    print("\nANOMALY INSIGHTS")
    print(insights)


def test_complete_insight_generation():

    df = create_sample_dataset()

    result = generate_automated_insights(
        df=df,

        outlier_result={
            "outlier_count": 4,
            "outlier_percentage": 6.67,
            "method": "IQR"
        },

        correlation_result={
            "significant_pairs": [
                {
                    "column_1": "Sales",
                    "column_2": "Profit",
                    "correlation": 0.95,
                    "p_value": 0.001
                }
            ]
        },

        hypothesis_result={
            "test_name": "Independent t-test",
            "p_value": 0.002
        },

        trend_result={
            "trend": {
                "direction": "increasing",
                "slope": 10.2,
                "p_value": 0.001,
                "r_squared": 0.91,
                "total_change": 500
            },
            "seasonality": {
                "seasonal_strength": 0.75
            }
        },

        anomaly_result={
            "anomaly_count": 3,
            "total_observations": 60,
            "anomaly_percentage": 5.0,
            "method": "rolling_mad"
        }
    )

    assert result["success"] is True
    assert result["total_insights"] > 0
    assert result["high_priority_count"] >= 0

    # High-priority insights must appear first.
    priorities = [
        item["priority"]
        for item in result["insights"]
    ]

    priority_values = [
        {"high": 0, "medium": 1, "low": 2}[p]
        for p in priorities
    ]

    assert priority_values == sorted(priority_values)

    print("\nCOMPLETE AUTOMATED INSIGHT REPORT")

    print("Total insights:", result["total_insights"])
    print("High priority:", result["high_priority_count"])
    print("Categories:", result["category_counts"])

    for index, insight in enumerate(
        result["insights"], start=1
    ):

        print(f"\nInsight {index}")
        print("Category:", insight["category"])
        print("Title:", insight["title"])
        print("Description:", insight["description"])
        print("Priority:", insight["priority"])
        print("Evidence:", insight["evidence"])


if __name__ == "__main__":

    test_data_quality_insights()
    test_outlier_insights()
    test_correlation_insights()
    test_hypothesis_insights()
    test_trend_insights()
    test_anomaly_insights()
    test_complete_insight_generation()

    print("\nAll automated insight tests completed successfully.")
