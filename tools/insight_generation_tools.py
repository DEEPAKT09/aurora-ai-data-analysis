
import math
import pandas as pd
import numpy as np


def _safe_number(value):
    """Convert a value to a finite Python float."""

    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def _make_insight(
    category,
    title,
    description,
    evidence,
    priority="medium",
    recommendation=None
):
    """Create a consistent insight object."""

    return {
        "category": category,
        "title": title,
        "description": description,
        "evidence": evidence,
        "priority": priority,
        "recommendation": recommendation
    }


def _get_dataframe(result, key="results"):
    """Extract a DataFrame from an analytical result."""

    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, dict):
        value = result.get(key)

        if isinstance(value, pd.DataFrame):
            return value

    return None


def generate_data_quality_insights(df):
    """Generate insights about dataset quality."""

    insights = []

    if df is None or not isinstance(df, pd.DataFrame):
        return insights

    total_rows, total_columns = df.shape

    if total_rows == 0:
        return insights

    missing_cells = int(df.isna().sum().sum())

    total_cells = total_rows * total_columns

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells else 0
    )

    duplicate_rows = int(df.duplicated().sum())

    if missing_cells > 0:

        priority = (
            "high"
            if missing_percentage >= 20
            else "medium"
        )

        insights.append(
            _make_insight(
                category="data_quality",
                title="Missing values detected",
                description=(
                    f"The dataset contains {missing_cells} "
                    f"missing cells across {total_cells} cells."
                ),
                evidence={
                    "missing_cells": missing_cells,
                    "total_cells": total_cells,
                    "missing_percentage": round(
                        missing_percentage, 2
                    )
                },
                priority=priority,
                recommendation=(
                    "Review missing-value patterns and choose "
                    "an appropriate imputation or removal strategy."
                )
            )
        )

    if duplicate_rows > 0:

        insights.append(
            _make_insight(
                category="data_quality",
                title="Duplicate records detected",
                description=(
                    f"The dataset contains {duplicate_rows} "
                    "duplicate rows."
                ),
                evidence={
                    "duplicate_rows": duplicate_rows,
                    "total_rows": total_rows
                },
                priority="medium",
                recommendation=(
                    "Check whether duplicate records are valid "
                    "repeated transactions or accidental duplicates."
                )
            )
        )

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:

        insights.append(
            _make_insight(
                category="data_quality",
                title="Constant columns detected",
                description=(
                    "Some columns contain only one unique value "
                    "and may provide little analytical information."
                ),
                evidence={
                    "columns": constant_columns
                },
                priority="low",
                recommendation=(
                    "Review whether these columns are necessary "
                    "for further analysis."
                )
            )
        )

    return insights


def generate_outlier_insights(outlier_result):
    """Generate insights from general outlier detection results."""

    insights = []

    if not isinstance(outlier_result, dict):
        return insights

    outlier_count = outlier_result.get(
        "total_outliers",
        outlier_result.get("outlier_count")
    )

    if outlier_count is None:
        return insights

    try:
        outlier_count = int(outlier_count)
    except (TypeError, ValueError):
        return insights

    if outlier_count == 0:
        return insights

    percentage = _safe_number(
        outlier_result.get("outlier_percentage")
    )

    evidence = {
        "outlier_count": outlier_count,
        "outlier_percentage": percentage,
        "method": outlier_result.get("method")
    }

    insights.append(
        _make_insight(
            category="outlier",
            title="Unusual data points detected",
            description=(
                f"The outlier analysis identified "
                f"{outlier_count} unusual observations."
            ),
            evidence=evidence,
            priority="high" if percentage and percentage >= 10 else "medium",
            recommendation=(
                "Inspect the flagged observations before deciding "
                "whether to retain, correct or remove them."
            )
        )
    )

    return insights


def generate_correlation_insights(correlation_result):
    """Generate insights from correlation analysis."""

    insights = []

    if not isinstance(correlation_result, dict):
        return insights

    # Preferred input: list of pairwise correlation records.
    pairs = correlation_result.get("significant_pairs", [])

    if not pairs:
        pairs = correlation_result.get("correlations", [])

    if not isinstance(pairs, list):
        return insights

    for pair in pairs:

        if not isinstance(pair, dict):
            continue

        column_1 = pair.get(
            "column_1",
            pair.get("feature_1", pair.get("column_x"))
        )

        column_2 = pair.get(
            "column_2",
            pair.get("feature_2", pair.get("column_y"))
        )

        correlation = _safe_number(
            pair.get(
                "correlation",
                pair.get("correlation_coefficient", pair.get("r"))
            )
        )

        if column_1 is None or column_2 is None:
            continue

        if correlation is None:
            continue

        if abs(correlation) < 0.7:
            continue

        p_value = _safe_number(pair.get("p_value"))

        # If a p-value is available, require statistical significance.
        if p_value is not None and p_value >= 0.05:
            continue

        direction = (
            "positive"
            if correlation > 0
            else "negative"
        )

        strength = (
            "very strong"
            if abs(correlation) >= 0.9
            else "strong"
        )

        insights.append(
            _make_insight(
                category="correlation",
                title=(
                    f"{strength.capitalize()} {direction} "
                    "correlation detected"
                ),
                description=(
                    f"{column_1} and {column_2} show a "
                    f"{strength} {direction} correlation."
                ),
                evidence={
                    "column_1": column_1,
                    "column_2": column_2,
                    "correlation": correlation,
                    "p_value": p_value
                },
                priority="medium",
                recommendation=(
                    "Investigate the relationship further. "
                    "Correlation does not establish causation."
                )
            )
        )

    return insights


def generate_hypothesis_insights(hypothesis_result):
    """Generate insights from statistical hypothesis tests."""

    insights = []

    if not isinstance(hypothesis_result, dict):
        return insights

    p_value = _safe_number(
        hypothesis_result.get(
            "p_value",
            hypothesis_result.get("pvalue")
        )
    )

    if p_value is None:
        return insights

    test_name = hypothesis_result.get(
        "test_name",
        hypothesis_result.get(
            "test",
            "Statistical hypothesis test"
        )
    )

    significant = p_value < 0.05

    if significant:

        description = (
            f"The test found statistically significant evidence "
            f"of a difference or association "
            f"(p = {p_value:.5f})."
        )

        recommendation = (
            "Review the group-level results and effect size "
            "to understand the practical importance."
        )

        priority = "medium"

    else:

        description = (
            f"The test did not find sufficient statistical evidence "
            f"of a difference or association "
            f"(p = {p_value:.5f})."
        )

        recommendation = (
            "Do not interpret this as proof that no difference "
            "exists. Consider sample size and statistical power."
        )

        priority = "low"

    insights.append(
        _make_insight(
            category="hypothesis_testing",
            title=(
                "Statistically significant result"
                if significant
                else "No statistically significant result"
            ),
            description=description,
            evidence={
                "test_name": test_name,
                "p_value": p_value,
                "significance_level": 0.05,
                "statistically_significant": significant
            },
            priority=priority,
            recommendation=recommendation
        )
    )

    return insights


def generate_trend_insights(trend_result):
    """Generate insights from trend and seasonality analysis."""

    insights = []

    if not isinstance(trend_result, dict):
        return insights

    trend = trend_result.get("trend")

    if isinstance(trend, dict):

        direction = trend.get("direction")

        slope = _safe_number(trend.get("slope"))

        p_value = _safe_number(trend.get("p_value"))

        if direction and slope is not None:

            if p_value is not None and p_value >= 0.05:
                description = (
                    f"The estimated trend is {direction}, "
                    "but it is not statistically significant "
                    f"(p = {p_value:.5f})."
                )

                priority = "low"

            else:
                description = (
                    f"The time series shows an estimated "
                    f"{direction} trend with a slope of "
                    f"{slope:.4f} per period."
                )

                priority = "medium"

            insights.append(
                _make_insight(
                    category="trend",
                    title=f"{direction.capitalize()} trend detected",
                    description=description,
                    evidence={
                        "direction": direction,
                        "slope": slope,
                        "p_value": p_value,
                        "r_squared": _safe_number(
                            trend.get("r_squared")
                        ),
                        "total_change": _safe_number(
                            trend.get("total_change")
                        )
                    },
                    priority=priority,
                    recommendation=(
                        "Review the trend alongside seasonal patterns "
                        "and relevant business events."
                    )
                )
            )

    seasonality = trend_result.get("seasonality")

    if isinstance(seasonality, dict):

        strength = _safe_number(
            seasonality.get("seasonal_strength")
        )

        if strength is not None:

            if strength >= 0.6:

                insights.append(
                    _make_insight(
                        category="seasonality",
                        title="Strong seasonal pattern detected",
                        description=(
                            "The time series shows a relatively "
                            "strong recurring seasonal pattern."
                        ),
                        evidence={
                            "seasonal_strength": strength
                        },
                        priority="medium",
                        recommendation=(
                            "Consider seasonal patterns when "
                            "planning inventory, staffing or forecasts."
                        )
                    )
                )

            elif strength <= 0.2:

                insights.append(
                    _make_insight(
                        category="seasonality",
                        title="Weak seasonal pattern detected",
                        description=(
                            "The measured seasonal component "
                            "is relatively weak."
                        ),
                        evidence={
                            "seasonal_strength": strength
                        },
                        priority="low"
                    )
                )

    return insights


def generate_anomaly_insights(anomaly_result):
    """Generate insights from time-series anomaly detection."""

    insights = []

    if not isinstance(anomaly_result, dict):
        return insights

    anomaly_count = anomaly_result.get("anomaly_count")

    if anomaly_count is None:
        return insights

    try:
        anomaly_count = int(anomaly_count)
    except (TypeError, ValueError):
        return insights

    total = anomaly_result.get("total_observations", 0)

    try:
        total = int(total)
    except (TypeError, ValueError):
        total = 0

    if anomaly_count == 0:
        return insights

    percentage = _safe_number(
        anomaly_result.get("anomaly_percentage")
    )

    insights.append(
        _make_insight(
            category="anomaly",
            title="Time-series anomalies detected",
            description=(
                f"The time-series analysis identified "
                f"{anomaly_count} unusual observations."
            ),
            evidence={
                "anomaly_count": anomaly_count,
                "total_observations": total,
                "anomaly_percentage": percentage,
                "method": anomaly_result.get("method")
            },
            priority=(
                "high"
                if percentage is not None and percentage >= 10
                else "medium"
            ),
            recommendation=(
                "Investigate the flagged dates and compare them "
                "with promotions, holidays, operational changes "
                "or data collection issues."
            )
        )
    )

    return insights


def generate_automated_insights(
    df=None,
    outlier_result=None,
    correlation_result=None,
    hypothesis_result=None,
    trend_result=None,
    anomaly_result=None
):
    """
    Combine available analytical results into
    a single evidence-backed insight report.
    """

    insights = []

    insights.extend(
        generate_data_quality_insights(df)
    )

    insights.extend(
        generate_outlier_insights(outlier_result)
    )

    insights.extend(
        generate_correlation_insights(correlation_result)
    )

    insights.extend(
        generate_hypothesis_insights(hypothesis_result)
    )

    insights.extend(
        generate_trend_insights(trend_result)
    )

    insights.extend(
        generate_anomaly_insights(anomaly_result)
    )

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2
    }

    insights.sort(
        key=lambda item: priority_order.get(
            item["priority"], 3
        )
    )

    category_counts = {}

    for insight in insights:
        category = insight["category"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    return {
        "success": True,
        "total_insights": len(insights),
        "high_priority_count": sum(
            1 for item in insights
            if item["priority"] == "high"
        ),
        "category_counts": category_counts,
        "insights": insights
    }
