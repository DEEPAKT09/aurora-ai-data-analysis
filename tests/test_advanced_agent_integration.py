import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from agents.advanced_analytics_agent import run_advanced_analytics_agent


def sample_df():
    rng = np.random.default_rng(17)
    dates = pd.date_range("2021-01-01", periods=60, freq="MS")
    sales = 1000 + np.arange(60) * 8 + 80 * np.sin(
        2 * np.pi * np.arange(60) / 12
    ) + rng.normal(0, 7, 60)
    sales[30] += 450
    regions = np.where(np.arange(60) % 2 == 0, "North", "South")
    return pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Profit": sales * 0.22 + rng.normal(0, 3, 60),
        "Region": regions
    })


def test_outlier_integration():
    result = run_advanced_analytics_agent(
        "Find outliers", sample_df(),
        config={"operation": "outliers", "columns": ["Sales"]}
    )
    assert result["success"]
    assert "outliers" in result["results"]
    assert result["insights"]["total_insights"] >= 1


def test_correlation_integration():
    result = run_advanced_analytics_agent(
        "Find significant correlations", sample_df(),
        config={"operation": "correlation", "columns": ["Sales", "Profit"]}
    )
    assert result["success"]
    assert "pearson" in result["results"]["correlation"]
    assert isinstance(result["insights"]["insights"], list)


def test_trend_integration():
    result = run_advanced_analytics_agent(
        "Analyze sales trend", sample_df(),
        config={
            "operation": "trend",
            "date_column": "Date",
            "target_column": "Sales",
            "seasonal_period": 12
        }
    )
    assert result["success"]
    assert "trend" in result["results"]["trend"]
    assert result["insights"]["total_insights"] >= 1


def test_hypothesis_integration():
    result = run_advanced_analytics_agent(
        "Compare sales between regions", sample_df(),
        config={
            "operation": "hypothesis",
            "test_type": "independent_t",
            "numeric_column": "Sales",
            "group_column": "Region",
            "group_a": "North",
            "group_b": "South"
        }
    )
    assert result["success"]
    assert "p_value" in result["results"]["hypothesis"]
    assert result["insights"]["total_insights"] >= 1


if __name__ == "__main__":
    test_outlier_integration()
    test_correlation_integration()
    test_trend_integration()
    test_hypothesis_integration()
    print("All Phase 8 advanced-agent integration tests passed.")
