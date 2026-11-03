
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import (
    ttest_ind,
    ttest_rel,
    f_oneway,
    chi2_contingency,
    shapiro
)


def _validate_numeric_series(series, column_name):
    """Return finite numerical observations."""

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(values) < 2:
        raise ValueError(
            f"{column_name} must contain at least two valid observations."
        )

    return values.astype(float)


def _interpret_p_value(p_value, alpha):
    """Interpret a p-value against the selected significance level."""

    significant = bool(p_value < alpha)

    if significant:
        interpretation = (
            "Reject the null hypothesis. "
            "The result is statistically significant."
        )
    else:
        interpretation = (
            "Fail to reject the null hypothesis. "
            "There is insufficient evidence of a statistically "
            "significant difference or association."
        )

    return {
        "alpha": float(alpha),
        "statistically_significant": significant,
        "interpretation": interpretation
    }


def independent_t_test(
    df,
    numeric_column,
    group_column,
    group_a,
    group_b,
    equal_var=False,
    alpha=0.05
):
    """
    Compare the means of two independent groups.

    Welch's t-test is used by default because it does not
    assume equal population variances.
    """

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if numeric_column not in df.columns:
        raise ValueError(f"Column not found: {numeric_column}")

    if group_column not in df.columns:
        raise ValueError(f"Column not found: {group_column}")

    if group_a == group_b:
        raise ValueError("Please select two different groups.")

    selected = df[
        df[group_column].isin([group_a, group_b])
    ]

    values_a = _validate_numeric_series(
        selected.loc[
            selected[group_column] == group_a,
            numeric_column
        ],
        str(group_a)
    )

    values_b = _validate_numeric_series(
        selected.loc[
            selected[group_column] == group_b,
            numeric_column
        ],
        str(group_b)
    )

    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError(
            "Each group must have at least two valid observations."
        )

    statistic, p_value = ttest_ind(
        values_a,
        values_b,
        equal_var=equal_var
    )

    # Cohen's d using pooled standard deviation.
    n_a = len(values_a)
    n_b = len(values_b)

    pooled_variance = (
        ((n_a - 1) * values_a.var(ddof=1))
        + ((n_b - 1) * values_b.var(ddof=1))
    ) / (n_a + n_b - 2)

    pooled_std = np.sqrt(pooled_variance)

    cohen_d = (
        float((values_a.mean() - values_b.mean()) / pooled_std)
        if pooled_std > 0 else None
    )

    return {
        "success": True,
        "test": "Independent samples t-test",
        "method": "Welch's t-test" if not equal_var else "Student's t-test",
        "numeric_column": numeric_column,
        "group_column": group_column,
        "group_a": str(group_a),
        "group_b": str(group_b),
        "group_a_count": n_a,
        "group_b_count": n_b,
        "group_a_mean": float(values_a.mean()),
        "group_b_mean": float(values_b.mean()),
        "mean_difference": float(values_a.mean() - values_b.mean()),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cohen_d": cohen_d,
        **_interpret_p_value(p_value, alpha)
    }


def paired_t_test(
    df,
    column_a,
    column_b,
    alpha=0.05
):
    """
    Compare two related measurements from the same observations.

    Rows with missing values in either measurement are excluded.
    """

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if column_a not in df.columns or column_b not in df.columns:
        raise ValueError("One or both selected columns do not exist.")

    paired = df[[column_a, column_b]].copy()

    paired[column_a] = pd.to_numeric(
        paired[column_a], errors="coerce"
    )

    paired[column_b] = pd.to_numeric(
        paired[column_b], errors="coerce"
    )

    paired = paired.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(paired) < 2:
        raise ValueError(
            "At least two complete paired observations are required."
        )

    values_a = paired[column_a].astype(float)
    values_b = paired[column_b].astype(float)

    differences = values_a - values_b

    if np.isclose(differences.std(ddof=1), 0):
        raise ValueError(
            "Paired differences have zero variance; "
            "the t-statistic cannot be meaningfully calculated."
        )

    statistic, p_value = ttest_rel(
        values_a,
        values_b
    )

    standard_deviation = differences.std(ddof=1)

    cohen_dz = (
        float(differences.mean() / standard_deviation)
    )

    return {
        "success": True,
        "test": "Paired samples t-test",
        "column_a": column_a,
        "column_b": column_b,
        "paired_observations": len(paired),
        "column_a_mean": float(values_a.mean()),
        "column_b_mean": float(values_b.mean()),
        "mean_difference": float(differences.mean()),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cohen_dz": cohen_dz,
        **_interpret_p_value(p_value, alpha)
    }


def one_way_anova(
    df,
    numeric_column,
    group_column,
    alpha=0.05
):
    """
    Compare means across three or more independent groups.
    """

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if numeric_column not in df.columns:
        raise ValueError(f"Column not found: {numeric_column}")

    if group_column not in df.columns:
        raise ValueError(f"Column not found: {group_column}")

    groups = []

    group_details = []

    for group_name, group_df in df.groupby(
        group_column,
        observed=True
    ):

        values = _validate_numeric_series(
            group_df[numeric_column],
            str(group_name)
        )

        if len(values) < 2:
            raise ValueError(
                f"Group {group_name} must have at least "
                "two valid observations."
            )

        groups.append(values.to_numpy())

        group_details.append({
            "group": str(group_name),
            "count": len(values),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1))
        })

    if len(groups) < 3:
        raise ValueError(
            "One-way ANOVA requires at least three groups."
        )

    statistic, p_value = f_oneway(*groups)

    # Eta-squared effect size.
    all_values = np.concatenate(groups)

    grand_mean = np.mean(all_values)

    between_group_ss = sum(
        len(group) * (np.mean(group) - grand_mean) ** 2
        for group in groups
    )

    total_ss = np.sum(
        (all_values - grand_mean) ** 2
    )

    eta_squared = (
        float(between_group_ss / total_ss)
        if total_ss > 0 else None
    )

    return {
        "success": True,
        "test": "One-way ANOVA",
        "numeric_column": numeric_column,
        "group_column": group_column,
        "number_of_groups": len(groups),
        "groups": group_details,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
        **_interpret_p_value(p_value, alpha),
        "follow_up_note": (
            "A significant ANOVA indicates that at least one "
            "group mean differs. A post-hoc test is needed to "
            "identify which groups differ."
        )
    }


def chi_square_test(
    df,
    column_a,
    column_b,
    alpha=0.05
):
    """
    Test whether two categorical variables are associated.

    Reports Cramer's V as an effect-size measure.
    """

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if column_a not in df.columns or column_b not in df.columns:
        raise ValueError("One or both selected columns do not exist.")

    selected = df[[column_a, column_b]].dropna()

    if len(selected) == 0:
        raise ValueError("No complete observations are available.")

    contingency_table = pd.crosstab(
        selected[column_a],
        selected[column_b]
    )

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError(
            "Both categorical columns must contain at least "
            "two distinct categories."
        )

    statistic, p_value, degrees_of_freedom, expected = (
        chi2_contingency(contingency_table)
    )

    sample_size = contingency_table.to_numpy().sum()

    minimum_dimension = min(contingency_table.shape) - 1

    cramers_v = (
        float(np.sqrt(
            statistic / (sample_size * minimum_dimension)
        ))
        if sample_size > 0 and minimum_dimension > 0
        else None
    )

    return {
        "success": True,
        "test": "Chi-square test of independence",
        "column_a": column_a,
        "column_b": column_b,
        "observations": int(sample_size),
        "degrees_of_freedom": int(degrees_of_freedom),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "contingency_table": contingency_table.to_dict(),
        "expected_frequencies": expected.tolist(),
        "minimum_expected_frequency": float(expected.min()),
        "expected_cells_below_5": int((expected < 5).sum()),
        **_interpret_p_value(p_value, alpha),
        "assumption_warning": (
            "Some expected cell frequencies are below 5. "
            "Interpret the chi-square approximation cautiously."
            if (expected < 5).any()
            else None
        )
    }


def normality_test(
    df,
    numeric_column,
    alpha=0.05
):
    """
    Assess normality using the Shapiro-Wilk test.

    For large datasets, a reproducible sample of 5000 values
    is used because very large samples can make the test
    excessively sensitive.
    """

    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")

    if numeric_column not in df.columns:
        raise ValueError(f"Column not found: {numeric_column}")

    values = _validate_numeric_series(
        df[numeric_column],
        numeric_column
    )

    if len(values) < 3:
        raise ValueError(
            "Normality testing requires at least three observations."
        )

    if values.nunique() < 2:
        raise ValueError(
            "Normality cannot be assessed for a constant column."
        )

    original_count = len(values)

    if len(values) > 5000:
        values = values.sample(
            n=5000,
            random_state=42
        )

    statistic, p_value = shapiro(values)

    result = {
        "success": True,
        "test": "Shapiro-Wilk normality test",
        "column": numeric_column,
        "original_observations": original_count,
        "tested_observations": len(values),
        "statistic": float(statistic),
        "p_value": float(p_value),
        **_interpret_p_value(p_value, alpha)
    }

    result["normality_interpretation"] = (
        "The test did not find sufficient evidence against "
        "normality."
        if p_value >= alpha
        else
        "The test provides evidence that the data may not "
        "follow a normal distribution."
    )

    return result
