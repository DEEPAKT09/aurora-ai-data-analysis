
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.correlation_tools import (
    calculate_correlation_matrix,
    calculate_correlation_significance,
    generate_correlation_insights,
    run_correlation_analysis
)


def create_sample_data():

    np.random.seed(42)

    advertising = np.arange(1, 51, dtype=float)

    sales = (
        100 + advertising * 20
        + np.random.normal(0, 15, 50)
    )

    profit = (
        sales * 0.30
        + np.random.normal(0, 5, 50)
    )

    quantity = np.random.randint(10, 100, 50)

    return pd.DataFrame({
        "Advertising": advertising,
        "Sales": sales,
        "Profit": profit,
        "Quantity": quantity
    })


def test_pearson_correlation():

    df = create_sample_data()

    result = calculate_correlation_matrix(
        df,
        method="pearson"
    )

    assert result["success"] is True
    assert "Sales" in result["columns"]

    correlation = result["correlation_matrix"][
        "Advertising"
    ]["Sales"]

    assert correlation > 0.9

    print("\nPEARSON CORRELATION")
    print(correlation)


def test_spearman_correlation():

    df = create_sample_data()

    result = calculate_correlation_matrix(
        df,
        method="spearman"
    )

    assert result["success"] is True

    correlation = result["correlation_matrix"][
        "Advertising"
    ]["Sales"]

    assert correlation > 0.9

    print("\nSPEARMAN CORRELATION")
    print(correlation)


def test_significance():

    df = create_sample_data()

    result = calculate_correlation_significance(
        df,
        method="pearson",
        alpha=0.05
    )

    assert result["success"] is True
    assert result["total_pairs"] > 0

    print("\nCORRELATION SIGNIFICANCE")
    print(pd.DataFrame(result["pairs"]).head())


def test_correlation_insights():

    df = create_sample_data()

    matrix = calculate_correlation_matrix(
        df,
        method="pearson"
    )

    significance = calculate_correlation_significance(
        df,
        method="pearson"
    )

    result = generate_correlation_insights(
        matrix,
        significance
    )

    assert result["success"] is True
    assert result["insight_count"] >= 1

    print("\nCORRELATION INSIGHTS")

    for insight in result["insights"]:
        print(insight["description"])


def test_complete_analysis():

    df = create_sample_data()

    result = run_correlation_analysis(df)

    assert result["success"] is True
    assert "pearson" in result
    assert "spearman" in result

    print("\nCOMPLETE CORRELATION ANALYSIS")

    print(
        "Pearson significant pairs:",
        result["pearson"]["significance"]["significant_pairs"]
    )

    print(
        "Spearman significant pairs:",
        result["spearman"]["significance"]["significant_pairs"]
    )


def test_missing_values():

    df = create_sample_data()

    df.loc[0, "Sales"] = np.nan
    df.loc[1, "Profit"] = np.nan

    result = run_correlation_analysis(df)

    assert result["success"] is True

    print("\nMISSING VALUE TEST PASSED")


if __name__ == "__main__":

    test_pearson_correlation()
    test_spearman_correlation()
    test_significance()
    test_correlation_insights()
    test_complete_analysis()
    test_missing_values()

    print("\nAll correlation tests passed!")
