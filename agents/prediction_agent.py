
from typing import Literal, Optional, Any

import pandas as pd
from pydantic import BaseModel, Field

from agents.query_planner import get_gemini_model


class PredictionPlan(BaseModel):
    task_type: Literal[
        "regression",
        "classification",
        "forecasting",
    ] = Field(
        description="The prediction task requested by the user."
    )

    target_column: str = Field(
        description="Exact target column name from the dataset."
    )

    date_column: Optional[str] = Field(
        default=None,
        description="Date/time column for forecasting. Null for tabular tasks.",
    )

    selected_models: list[str] = Field(
        default_factory=list,
        description=(
            "Exact scikit-learn model names to run. "
            "Use an empty list to compare all available models."
        ),
    )

    prediction_length: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of future periods for forecasting.",
    )

    frequency: Optional[str] = Field(
        default=None,
        description=(
            "Pandas frequency such as D, W, MS or ME. "
            "Use null when it should be inferred."
        ),
    )


def create_prediction_plan(
    user_question: str,
    df: pd.DataFrame,
) -> PredictionPlan:
    """Use Gemini to understand the user's prediction request."""

    model = get_gemini_model()

    structured_model = model.with_structured_output(
        PredictionPlan,
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
You are the planning component of an AI Data Analyst.

Identify the correct predictive analytics task and dataset columns.

TASK TYPES:

1. regression:
Predict a continuous numerical target, such as salary, revenue,
sales, profit or house price.

2. classification:
Predict a category or class, such as purchased/not purchased,
approved/rejected or customer segment.

3. forecasting:
Predict future values of a numerical variable using historical
time-series observations.

TABULAR MODEL SELECTION:

Available scikit-learn models:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN

If the user names specific scikit-learn models, return their exact
names in selected_models.

If the user requests a general comparison, leave selected_models
empty so all available scikit-learn models can be compared.

FORECASTING:

Forecasting uses Holt-Winters Exponential Smoothing.

Identify the exact date column and numerical target column.

Interpret forecast horizons such as:
- next 7 days
- next 3 months
- next 12 periods

Use a suitable pandas frequency:
- Daily: D
- Weekly: W
- Monthly: MS
- Quarterly: QS
- Yearly: YS

If the frequency is unclear, return null.

DATASET COLUMNS:
{column_information}

USER QUESTION:
{user_question}

RULES:
- Use only exact column names from the dataset.
- Never invent a target or date column.
- Do not confuse statistical analysis with prediction.
- Choose forecasting only when future time-based values are requested.
- Return only the structured prediction plan.
"""

    plan = structured_model.invoke(prompt)

    if plan.target_column not in df.columns:
        raise ValueError(
            f"The selected target column '{plan.target_column}' "
            "does not exist in the dataset."
        )

    if plan.task_type == "forecasting":
        if not plan.date_column:
            raise ValueError(
                "A date/time column is required for forecasting."
            )

        if plan.date_column not in df.columns:
            raise ValueError(
                f"Date column '{plan.date_column}' "
                "does not exist in the dataset."
            )

        if plan.target_column == plan.date_column:
            raise ValueError(
                "The date column cannot also be the forecast target."
            )

    return plan


def make_json_safe(value: Any):
    """Convert model outputs into JSON-serializable values."""

    if isinstance(value, pd.DataFrame):
        return [
            make_json_safe(record)
            for record in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return [
            make_json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return make_json_safe(value.item())

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def run_prediction_agent(
    user_question,
    df,
    prediction_config=None,
):
    """Execute predictive analysis using the configured prediction plan."""

    try:
        # 1. Create or receive prediction plan.
        if prediction_config:
            plan = PredictionPlan(**prediction_config)

            if plan.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{plan.target_column}' "
                    "does not exist in the dataset."
                )

            if plan.task_type == "forecasting":
                if not plan.date_column:
                    raise ValueError(
                        "A date column is required for forecasting."
                    )

                if plan.date_column not in df.columns:
                    raise ValueError(
                        f"Date column '{plan.date_column}' "
                        "does not exist in the dataset."
                    )

                if plan.date_column == plan.target_column:
                    raise ValueError(
                        "Target and date columns must be different."
                    )

            if plan.task_type in ["regression", "classification"]:
                if not plan.selected_models:
                    raise ValueError(
                        "Select at least one machine learning model."
                    )
        else:
            plan = create_prediction_plan(user_question, df)

        # 2. Execute Holt-Winters forecasting.
        if plan.task_type == "forecasting":
            from ml_models.forecasting_models import forecast_time_series

            result = forecast_time_series(
                df=df,
                date_column=plan.date_column,
                target_column=plan.target_column,
                prediction_length=plan.prediction_length,
                frequency=plan.frequency,
            )

            result = make_json_safe(result)

            return {
                "success": True,
                "task_type": "forecasting",
                "target_column": plan.target_column,
                "date_column": plan.date_column,
                "engines": ["Holt-Winters"],
                "results": {
                    "Holt-Winters": result,
                },
                "answer": (
                    "Holt-Winters Exponential Smoothing successfully "
                    f"generated {plan.prediction_length} future forecasts "
                    f"for '{plan.target_column}'."
                ),
            }

        # 3. Execute scikit-learn models.
        from ml_models.tabular_models import train_and_compare_models

        sklearn_result = train_and_compare_models(
            df=df,
            target_column=plan.target_column,
            task_type=plan.task_type,
            selected_models=(
                plan.selected_models
                if plan.selected_models
                else None
            ),
        )

        sklearn_result = make_json_safe(sklearn_result)

        best_model = sklearn_result.get("best_model")

        return {
            "success": True,
            "task_type": plan.task_type,
            "target_column": plan.target_column,
            "engines": ["scikit-learn"],
            "results": {
                "scikit-learn": sklearn_result,
            },
            "answer": (
                f"Predictive modelling completed for "
                f"'{plan.target_column}'. "
                f"The best scikit-learn model was {best_model}."
            ),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "answer": (
                "I couldn't complete the predictive analysis. "
                f"Reason: {error}"
            ),
        }
