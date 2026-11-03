
import logging
from typing import Literal

from pydantic import BaseModel, Field

from agents.query_planner import get_gemini_model


logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    route: Literal[
        "profiling",
        "analysis",
        "advanced_analysis",
        "visualization",
        "prediction",
        "unsupported",
    ] = Field(
        description="The specialist agent that should handle the request."
    )

    reason: str = Field(
        description="A brief explanation for choosing this agent."
    )


def fallback_route(user_question: str):
    """
    Deterministic routing for common requests when Gemini routing
    is unavailable.

    This is deliberately conservative. It does not attempt to
    understand complex or ambiguous natural-language requests.
    """

    question = user_question.lower().strip()

    if not question:
        return None

    # Visualization requests.
    visualization_terms = [
        "chart",
        "graph",
        "plot",
        "visualize",
        "visualise",
        "histogram",
        "heatmap",
        "heat map",
        "box plot",
        "scatter plot",
        "bar chart",
        "line chart",
        "pie chart",
    ]

    if any(term in question for term in visualization_terms):
        return SupervisorDecision(
            route="visualization",
            reason="Deterministic fallback: visualization keyword detected.",
        )

    # Prediction and forecasting requests.
    prediction_terms = [
        "predict",
        "prediction",
        "forecast",
        "forecasting",
        "future value",
        "train a model",
        "machine learning model",
        "classification model",
        "regression model",
        "model comparison",
    ]

    if any(term in question for term in prediction_terms):
        return SupervisorDecision(
            route="prediction",
            reason="Deterministic fallback: predictive analytics keyword detected.",
        )

    # Advanced analytics requests.
    advanced_terms = [
        "outlier",
        "anomaly",
        "anomalies",
        "hypothesis test",
        "hypothesis testing",
        "t-test",
        "ttest",
        "anova",
        "chi-square",
        "chi square",
        "statistical significance",
        "p-value",
        "p value",
        "seasonality",
        "time series",
        "time-series",
        "trend analysis",
        "automated insights",
        "key insights",
    ]

    if any(term in question for term in advanced_terms):
        return SupervisorDecision(
            route="advanced_analysis",
            reason="Deterministic fallback: advanced analytics keyword detected.",
        )

    # Dataset profiling requests.
    profiling_terms = [
        "dataset overview",
        "data overview",
        "profile",
        "profiling",
        "schema",
        "data quality",
        "missing values",
        "missing data",
        "duplicate rows",
        "duplicates",
        "column types",
        "dataset summary",
    ]

    if any(term in question for term in profiling_terms):
        return SupervisorDecision(
            route="profiling",
            reason="Deterministic fallback: dataset profiling keyword detected.",
        )

    # Ordinary analysis requests.
    analysis_terms = [
        "average",
        "mean",
        "median",
        "sum",
        "total",
        "count",
        "maximum",
        "minimum",
        "max",
        "min",
        "correlation",
        "compare",
        "comparison",
        "group by",
        "groupby",
        "top ",
        "highest",
        "lowest",
        "percentage",
        "distribution",
        "statistics",
        "calculate",
    ]

    if any(term in question for term in analysis_terms):
        return SupervisorDecision(
            route="analysis",
            reason="Deterministic fallback: standard analysis keyword detected.",
        )

    return None


def route_user_question(
    user_question: str,
    df,
) -> SupervisorDecision:
    """
    Route a user question using Gemini structured output.

    If Gemini routing fails, try a conservative deterministic
    fallback. If the question remains ambiguous, raise a controlled
    exception so the supervisor node can return a friendly error.
    """

    if not isinstance(user_question, str) or not user_question.strip():
        return SupervisorDecision(
            route="unsupported",
            reason="The user question is empty.",
        )

    try:
        model = get_gemini_model()

        structured_model = model.with_structured_output(
            SupervisorDecision,
            method="json_schema",
        )

        column_information = [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
            }
            for column in df.columns
        ]

        prompt = f"""
You supervise a multi-agent AI Data Analyst.
Select exactly one specialist agent.

Available agents:

profiling:
Dataset overview, schema, missing values, duplicates,
column information and data quality.

analysis:
Ordinary calculations, averages, totals, aggregations,
comparisons, descriptive statistics and simple correlations.

advanced_analysis:
Outlier detection, advanced Pearson/Spearman correlation
with significance, hypothesis tests (t-test, paired t-test,
ANOVA, chi-square, normality), trend analysis, seasonality,
time-series anomaly detection, and automated insight reports.

visualization:
The user explicitly asks to create or display a chart or graph.

prediction:
Future forecasting, regression/classification, machine-learning
training and model comparison.

unsupported:
Requests unrelated to the uploaded dataset or application.

Routing rules:

1. Prefer advanced_analysis for statistical inference,
   outliers, anomaly detection, time-series trend/seasonality,
   and automated insights.

2. Do not route ordinary average, total, count or group-by
   questions to advanced_analysis.

3. Route explicit chart requests to visualization.

4. Route machine-learning training and forecasting to prediction.

5. Do not answer the question or invent dataset columns.

Dataset columns:
{column_information}

User question:
{user_question}

Return only the structured route and reason.
"""

        decision = structured_model.invoke(prompt)

        if not isinstance(decision, SupervisorDecision):
            decision = SupervisorDecision.model_validate(decision)

        return decision

    except Exception as error:
        # Log the technical failure without logging the question,
        # dataset contents or API credentials.
        logger.warning(
            "Gemini supervisor routing failed: %s",
            type(error).__name__,
        )

        fallback = fallback_route(user_question)

        if fallback is not None:
            logger.info(
                "Using deterministic fallback route: %s",
                fallback.route,
            )
            return fallback

        # Let supervisor_node in graph/nodes.py classify this
        # failure and return a friendly user-facing message.
        raise RuntimeError(
            "Supervisor routing service unavailable."
        ) from error
