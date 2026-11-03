
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tools.hypothesis_testing_tools import (
    independent_t_test,
    paired_t_test,
    one_way_anova,
    chi_square_test,
    normality_test
)


def create_sample_data():

    np.random.seed(42)

    return pd.DataFrame({
        "Department": (
            ["Sales"] * 30
            + ["Engineering"] * 30
            + ["Marketing"] * 30
        ),

        "Salary": (
            np.random.normal(40000, 3000, 30).tolist()
            + np.random.normal(60000, 4000, 30).tolist()
            + np.random.normal(50000, 3500, 30).tolist()
        ),

        "Before": np.random.normal(100, 10, 90),

        "After": np.random.normal(110, 10, 90),

        "Gender": np.random.choice(
            ["Male", "Female"], 90
        ),

        "Purchase": np.random.choice(
            ["Yes", "No"], 90
        )
    })


def test_independent_t_test():

    df = create_sample_data()

    result = independent_t_test(
        df,
        numeric_column="Salary",
        group_column="Department",
        group_a="Sales",
        group_b="Engineering"
    )

    assert result["success"] is True
    assert "p_value" in result
    assert "cohen_d" in result

    print("\nINDEPENDENT T-TEST")
    print(result)


def test_paired_t_test():

    df = create_sample_data()

    result = paired_t_test(
        df,
        column_a="Before",
        column_b="After"
    )

    assert result["success"] is True
    assert "p_value" in result

    print("\nPAIRED T-TEST")
    print(result)


def test_one_way_anova():

    df = create_sample_data()

    result = one_way_anova(
        df,
        numeric_column="Salary",
        group_column="Department"
    )

    assert result["success"] is True
    assert result["number_of_groups"] == 3
    assert "eta_squared" in result

    print("\nONE-WAY ANOVA")
    print(result)


def test_chi_square():

    df = create_sample_data()

    result = chi_square_test(
        df,
        column_a="Gender",
        column_b="Purchase"
    )

    assert result["success"] is True
    assert "cramers_v" in result

    print("\nCHI-SQUARE TEST")
    print(result)


def test_normality():

    df = create_sample_data()

    result = normality_test(
        df,
        numeric_column="Salary"
    )

    assert result["success"] is True
    assert "p_value" in result

    print("\nNORMALITY TEST")
    print(result)


if __name__ == "__main__":

    test_independent_t_test()
    test_paired_t_test()
    test_one_way_anova()
    test_chi_square()
    test_normality()

    print("\nAll hypothesis testing tests passed!")
