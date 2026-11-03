import sys
from pathlib import Path

# Add the project root directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from tools.advanced_analytics_tools import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    run_outlier_analysis
)


def create_sample_data():

    return pd.DataFrame({
        "Sales": [
            100, 110, 105, 115, 120,
            108, 112, 118, 109, 5000,
            102, 106, 111, 114, 117,
            103, 107, 113, 116, 119
        ],
        "Profit": [
            10, 12, 11, 13, 14,
            11, 12, 13, 10, 900,
            10, 11, 12, 13, 14,
            9, 10, 12, 13, 14
        ],
        "Quantity": [
            2, 3, 2, 4, 3,
            2, 3, 4, 2, 100,
            3, 2, 4, 3, 2,
            3, 4, 2, 3, 4
        ]
    })


def test_iqr():

    df = create_sample_data()

    result = detect_outliers_iqr(
        df,
        columns=["Sales"]
    )

    assert result["success"] is True
    assert result["results"]["Sales"]["outlier_count"] >= 1

    print("\nIQR RESULTS")
    print(result)


def test_zscore():

    df = create_sample_data()

    result = detect_outliers_zscore(
        df,
        columns=["Sales"]
    )

    assert result["success"] is True
    assert result["results"]["Sales"]["outlier_count"] >= 1

    print("\nZ-SCORE RESULTS")
    print(result)


def test_isolation_forest():

    df = create_sample_data()

    result = detect_outliers_isolation_forest(
        df,
        contamination=0.05
    )

    assert result["success"] is True
    assert result["outlier_count"] >= 1

    print("\nISOLATION FOREST RESULTS")
    print(result)


def test_complete_outlier_analysis():

    df = create_sample_data()

    result = run_outlier_analysis(df)

    assert result["success"] is True
    assert "iqr" in result
    assert "zscore" in result
    assert "isolation_forest" in result

    print("\nCOMPLETE OUTLIER ANALYSIS")
    print(result)


if __name__ == "__main__":

    test_iqr()
    test_zscore()
    test_isolation_forest()
    test_complete_outlier_analysis()

    print("\nAll outlier detection tests passed!")
