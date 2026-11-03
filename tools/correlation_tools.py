
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr


def _get_numeric_data(df, columns=None):
    """
    Validate the dataset and return selected numerical columns.
    """

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    if columns is None:
        columns = df.select_dtypes(
            include=["number"]
        ).columns.tolist()

    else:
        missing = [
            column for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Columns not found: {missing}"
            )

        non_numeric = [
            column for column in columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

        if non_numeric:
            raise ValueError(
                f"These columns are not numerical: {non_numeric}"
            )

    if len(columns) < 2:
        raise ValueError(
            "At least two numerical columns are required."
        )

    numeric_df = df[columns].copy()

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove columns with no usable observations.
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Remove constant columns because their correlation
    # cannot be calculated meaningfully.
    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=True) > 1
    ]

    if numeric_df.shape[1] < 2:
        raise ValueError(
            "At least two non-constant numerical columns "
            "with usable values are required."
        )

    return numeric_df


def calculate_correlation_matrix(
    df,
    columns=None,
    method="pearson"
):
    """
    Calculate a Pearson or Spearman correlation matrix.

    Pairwise complete observations are used for each
    correlation calculation.
    """

    method = method.lower()

    if method not in ["pearson", "spearman"]:
        raise ValueError(
            "Method must be either 'pearson' or 'spearman'."
        )

    numeric_df = _get_numeric_data(df, columns)

    correlation_matrix = numeric_df.corr(
        method=method,
        min_periods=3
    )

    return {
        "success": True,
        "method": method,
        "columns": numeric_df.columns.tolist(),
        "correlation_matrix": (
            correlation_matrix.round(4).to_dict()
        )
    }


def calculate_correlation_significance(
    df,
    columns=None,
    method="pearson",
    alpha=0.05
):
    """
    Calculate pairwise correlations and p-values.

    alpha is the statistical significance threshold.
    """

    method = method.lower()

    if method not in ["pearson", "spearman"]:
        raise ValueError(
            "Method must be either 'pearson' or 'spearman'."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "Alpha must be between 0 and 1."
        )

    numeric_df = _get_numeric_data(df, columns)

    pairs = []

    column_names = numeric_df.columns.tolist()

    for i in range(len(column_names)):

        for j in range(i + 1, len(column_names)):

            column_a = column_names[i]
            column_b = column_names[j]

            pair_df = numeric_df[
                [column_a, column_b]
            ].dropna()

            observations = len(pair_df)

            if observations < 3:
                continue

            values_a = pair_df[column_a]
            values_b = pair_df[column_b]

            if values_a.nunique() < 2 or values_b.nunique() < 2:
                continue

            if method == "pearson":

                coefficient, p_value = pearsonr(
                    values_a,
                    values_b
                )

            else:

                coefficient, p_value = spearmanr(
                    values_a,
                    values_b
                )

            coefficient = float(coefficient)
            p_value = float(p_value)

            if not np.isfinite(coefficient) or not np.isfinite(p_value):
                continue

            if coefficient >= 0:
                direction = "Positive"
            else:
                direction = "Negative"

            absolute_correlation = abs(coefficient)

            if absolute_correlation >= 0.7:
                strength = "Strong"
            elif absolute_correlation >= 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"

            pairs.append({
                "feature_1": column_a,
                "feature_2": column_b,
                "correlation": round(coefficient, 4),
                "p_value": round(p_value, 6),
                "observations": observations,
                "direction": direction,
                "strength": strength,
                "statistically_significant": p_value < alpha
            })

    pairs.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True
    )

    return {
        "success": True,
        "method": method,
        "alpha": alpha,
        "total_pairs": len(pairs),
        "significant_pairs": sum(
            pair["statistically_significant"]
            for pair in pairs
        ),
        "pairs": pairs
    }


def generate_correlation_insights(
    correlation_result,
    significance_result,
    strong_threshold=0.7
):
    """
    Generate deterministic, evidence-based correlation insights.

    These are statistical observations, not causal conclusions.
    """

    if not correlation_result.get("success"):
        raise ValueError(
            "A successful correlation result is required."
        )

    if not significance_result.get("success"):
        raise ValueError(
            "A successful significance result is required."
        )

    pairs = significance_result.get("pairs", [])

    insights = []

    for pair in pairs:

        coefficient = pair["correlation"]
        significant = pair["statistically_significant"]

        if abs(coefficient) < strong_threshold:
            continue

        if not significant:
            continue

        direction = pair["direction"].lower()

        insights.append({
            "type": "strong_correlation",
            "feature_1": pair["feature_1"],
            "feature_2": pair["feature_2"],
            "correlation": coefficient,
            "p_value": pair["p_value"],
            "observations": pair["observations"],
            "description": (
                f"{pair['feature_1']} and {pair['feature_2']} "
                f"have a statistically significant strong "
                f"{direction} correlation "
                f"(r = {coefficient})."
            ),
            "interpretation": (
                "This indicates an association, not proof "
                "that one variable causes changes in the other."
            )
        })

    if not insights:
        insights.append({
            "type": "no_strong_significant_relationship",
            "description": (
                "No strong statistically significant "
                "correlations were identified using the "
                "selected method and significance threshold."
            )
        })

    return {
        "success": True,
        "insight_count": len(insights),
        "insights": insights
    }


def run_correlation_analysis(
    df,
    columns=None,
    alpha=0.05
):
    """
    Run Pearson and Spearman correlation analysis,
    statistical significance testing and insight generation.
    """

    pearson_matrix = calculate_correlation_matrix(
        df,
        columns=columns,
        method="pearson"
    )

    spearman_matrix = calculate_correlation_matrix(
        df,
        columns=columns,
        method="spearman"
    )

    pearson_significance = calculate_correlation_significance(
        df,
        columns=columns,
        method="pearson",
        alpha=alpha
    )

    spearman_significance = calculate_correlation_significance(
        df,
        columns=columns,
        method="spearman",
        alpha=alpha
    )

    pearson_insights = generate_correlation_insights(
        pearson_matrix,
        pearson_significance
    )

    spearman_insights = generate_correlation_insights(
        spearman_matrix,
        spearman_significance
    )

    return {
        "success": True,
        "pearson": {
            "matrix": pearson_matrix,
            "significance": pearson_significance,
            "insights": pearson_insights
        },
        "spearman": {
            "matrix": spearman_matrix,
            "significance": spearman_significance,
            "insights": spearman_insights
        }
    }
