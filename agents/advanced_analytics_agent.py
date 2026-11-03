from typing import Literal, Optional
import math
import pandas as pd
from pydantic import BaseModel, Field

from agents.query_planner import get_gemini_model
from tools.advanced_analytics_tools import run_outlier_analysis
from tools.correlation_tools import run_correlation_analysis
from tools.hypothesis_testing_tools import (
    independent_t_test, paired_t_test, one_way_anova,
    chi_square_test, normality_test
)
from tools.time_series_analysis import run_time_series_analysis
from tools.anomaly_detection_tools import run_anomaly_detection
from tools.insight_generation_tools import generate_automated_insights


class AdvancedAnalyticsPlan(BaseModel):
    operation: Literal[
        "outliers", "correlation", "hypothesis",
        "trend", "anomaly", "comprehensive"
    ] = Field(description="Advanced analysis operation.")
    columns: list[str] = Field(default_factory=list)
    date_column: Optional[str] = None
    target_column: Optional[str] = None
    frequency: Optional[str] = None
    aggregation: Literal["mean", "sum", "median", "min", "max"] = "mean"
    seasonal_period: Optional[int] = None
    summary_period: Literal["month", "weekday", "quarter", "year"] = "month"
    anomaly_method: Literal[
        "rolling_zscore", "rolling_mad", "seasonal_residual"
    ] = "rolling_mad"
    window: int = 6
    threshold: float = 3.5
    test_type: Literal[
        "independent_t", "paired_t", "anova", "chi_square", "normality"
    ] = "independent_t"
    numeric_column: Optional[str] = None
    group_column: Optional[str] = None
    group_a: Optional[str] = None
    group_b: Optional[str] = None
    column_a: Optional[str] = None
    column_b: Optional[str] = None


def create_advanced_analytics_plan(question: str, df: pd.DataFrame):
    model = get_gemini_model().with_structured_output(
        AdvancedAnalyticsPlan, method="json_schema"
    )
    schema = [
        {"name": str(c), "dtype": str(df[c].dtype),
         "sample_values": [str(v) for v in df[c].dropna().head(3)]}
        for c in df.columns
    ]
    prompt = f"""
You are a planning component for an advanced data analytics application.
Choose the correct operation and exact column names from the dataset.
Operations:
- outliers: general numeric outlier analysis
- correlation: Pearson/Spearman correlations
- hypothesis: statistical tests
- trend: time trend and optional seasonality
- anomaly: time-series anomaly detection
- comprehensive: run data quality and all applicable analyses where inputs are clear

For hypothesis choose test_type:
independent_t (numeric_column + group_column + group_a + group_b),
paired_t (column_a + column_b, both numeric),
anova (numeric_column + group_column),
chi_square (column_a + column_b, categorical),
normality (numeric_column).
For trend/anomaly specify date_column and target_column.
For seasonal analysis use seasonal_period only when a period is reasonably supported by the request/data.
Use only exact column names. Do not guess group values: choose observed values.
Dataset schema and sample values: {schema}
User request: {question}
Return only the structured plan.
"""
    plan = model.invoke(prompt)
    valid_columns = set(map(str, df.columns))
    for field in ("columns",):
        setattr(plan, field, [c for c in getattr(plan, field) if c in valid_columns])
    for field in ("date_column", "target_column", "numeric_column",
                  "group_column", "column_a", "column_b"):
        value = getattr(plan, field)
        if value is not None and value not in valid_columns:
            setattr(plan, field, None)
    return plan


def _json_safe(value):
    if isinstance(value, pd.DataFrame):
        return value.replace([float("inf"), float("-inf")], float("nan")).where(
            pd.notna(value), None
        ).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_hypothesis(df, plan):
    alpha = 0.05
    if plan.test_type == "independent_t":
        if not all([plan.numeric_column, plan.group_column,
                    plan.group_a is not None, plan.group_b is not None]):
            raise ValueError("Independent t-test requires a numeric column, group column and two group values.")
        observed = df[plan.group_column].dropna().unique().tolist()
        lookup = {str(value): value for value in observed}
        if plan.group_a not in lookup or plan.group_b not in lookup:
            raise ValueError(
                f"Choose two observed values from {plan.group_column}: "
                f"{[str(value) for value in observed[:20]]}"
            )
        return independent_t_test(
            df, plan.numeric_column, plan.group_column,
            lookup[plan.group_a], lookup[plan.group_b], alpha=alpha
        )
    if plan.test_type == "paired_t":
        if not plan.column_a or not plan.column_b:
            raise ValueError("Paired t-test requires two numeric columns.")
        return paired_t_test(df, plan.column_a, plan.column_b, alpha=alpha)
    if plan.test_type == "anova":
        if not plan.numeric_column or not plan.group_column:
            raise ValueError("ANOVA requires a numeric column and a group column.")
        return one_way_anova(df, plan.numeric_column, plan.group_column, alpha=alpha)
    if plan.test_type == "chi_square":
        if not plan.column_a or not plan.column_b:
            raise ValueError("Chi-square requires two categorical columns.")
        return chi_square_test(df, plan.column_a, plan.column_b, alpha=alpha)
    if not plan.numeric_column:
        raise ValueError("Normality testing requires a numeric column.")
    return normality_test(df, plan.numeric_column, alpha=alpha)


def run_advanced_analytics_agent(question: str, df: pd.DataFrame, config=None):
    if df is None or df.empty:
        raise ValueError("Upload a non-empty dataset before running advanced analytics.")

    plan = AdvancedAnalyticsPlan(**config) if config else create_advanced_analytics_plan(question, df)
    raw = {}
    errors = {}
    operation = plan.operation

    if operation in ("trend", "anomaly") and not (
        plan.date_column and plan.target_column
    ):
        raise ValueError(
            "Time-series analysis requires valid date_column and target_column fields."
        )

    def execute(name, function):
        try:
            raw[name] = function()
        except Exception as exc:
            if operation != "comprehensive":
                raise
            errors[name] = str(exc)

    if operation in ("outliers", "comprehensive"):
        execute("outliers", lambda: run_outlier_analysis(
            df, columns=plan.columns or None
        ))

    if operation in ("correlation", "comprehensive"):
        execute("correlation", lambda: run_correlation_analysis(
            df, columns=plan.columns or None
        ))

    if operation == "hypothesis":
        execute("hypothesis", lambda: _run_hypothesis(df, plan))

    if operation in ("trend", "comprehensive"):
        if plan.date_column and plan.target_column:
            execute("trend", lambda: run_time_series_analysis(
                df, plan.date_column, plan.target_column,
                frequency=plan.frequency,
                aggregation=plan.aggregation,
                seasonal_period=plan.seasonal_period,
                summary_period=plan.summary_period
            ))

    if operation in ("anomaly", "comprehensive"):
        if plan.date_column and plan.target_column:
            execute("anomaly", lambda: run_anomaly_detection(
                df, plan.date_column, plan.target_column,
                method=plan.anomaly_method,
                window=plan.window,
                threshold=plan.threshold,
                period=plan.seasonal_period or 12,
                frequency=plan.frequency,
                aggregation=plan.aggregation
            ))

    
    # Normalize outlier results from all three methods.
    outlier_for_insights = None

    if "outliers" in raw:

        outlier_results = raw["outliers"]

        flagged_indices = set()
        methods_detected = []
        method_counts = {}

        # Collect IQR and Z-score results.
        for method_name in ["iqr", "zscore"]:

            method_result = outlier_results.get(
                method_name, {}
            )

            column_results = method_result.get(
                "results", {}
            )

            method_indices = set()

            for column_result in column_results.values():

                if isinstance(column_result, dict):

                    indices = column_result.get(
                        "outlier_indices", []
                    )

                    method_indices.update(
                        str(index) for index in indices
                    )

            if method_indices:
                methods_detected.append(method_name)
                method_counts[method_name] = len(method_indices)

            flagged_indices.update(method_indices)

        # Collect Isolation Forest results.
        isolation_result = outlier_results.get(
            "isolation_forest", {}
        )

        isolation_indices = set(
            str(index)
            for index in isolation_result.get(
                "outlier_indices", []
            )
        )

        if isolation_indices:
            methods_detected.append("isolation_forest")
            method_counts["isolation_forest"] = len(
                isolation_indices
            )

        flagged_indices.update(isolation_indices)

        # Count unique observations, avoiding double counting
        # when multiple methods flag the same row.
        total_outliers = len(flagged_indices)

        outlier_for_insights = {
            "outlier_count": total_outliers,
            "outlier_percentage": round(
                total_outliers / max(len(df), 1) * 100,
                2
            ),
            "method": ", ".join(methods_detected),
            "methods_detected": methods_detected,
            "method_counts": method_counts
        }


    correlation_for_insights = None
    if "correlation" in raw:
        pairs = raw["correlation"].get("pearson", {}).get("significance", {}).get("pairs", [])
        correlation_for_insights = {
            "significant_pairs": [
                {
                    "column_1": p.get("feature_1"),
                    "column_2": p.get("feature_2"),
                    "correlation": p.get("correlation"),
                    "p_value": p.get("p_value")
                }
                for p in pairs
            ]
        }

    trend_for_insights = None
    if "trend" in raw:
        trend_data = raw["trend"].get("trend", {})
        season_data = raw["trend"].get("seasonality")
        trend_for_insights = {
            "trend": {
                "direction": trend_data.get("direction"),
                "slope": trend_data.get("slope_per_period"),
                "p_value": trend_data.get("p_value"),
                "r_squared": trend_data.get("r_squared"),
                "total_change": trend_data.get("total_change")
            },
            "seasonality": season_data
        }

    insights = generate_automated_insights(
        df=df,
        outlier_result=outlier_for_insights,
        correlation_result=correlation_for_insights,
        hypothesis_result=raw.get("hypothesis"),
        trend_result=trend_for_insights,
        anomaly_result=raw.get("anomaly")
    )

    return {
        "success": True,
        "operation": operation,
        "plan": _json_safe(plan.model_dump()),
        "results": _json_safe(raw),
        "errors": errors,
        "insights": _json_safe(insights),
        "answer": (
            f"Advanced {operation.replace('_', ' ')} analysis completed. "
            f"Generated {insights['total_insights']} evidence-backed insights."
        )
    }
