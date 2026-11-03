
from typing import Literal, Optional, Any

import pandas as pd
from pydantic import BaseModel, Field

from agents.query_planner import get_gemini_model

from ml_models.tabular_models import train_and_compare_models
from ml_models.tabpfn_models import train_tabpfn
from ml_models.forecasting_models import forecast_time_series


class PredictionPlan(BaseModel):
    task_type: Literal[
        "regression",
        "classification",
        "forecasting"
    ] = Field(
        description="The prediction task requested by the user."
    )

    target_column: str = Field(
        description="Exact target column name from the dataset."
    )

    date_column: Optional[str] = Field(
        default=None,
        description="Date/time column for forecasting. Null for tabular tasks."
    )

    selected_models: list[str] = Field(
        default_factory=list,
        description=(
            "Exact scikit-learn model names to run. "
            "Use an empty list to compare all available models."
        )
    )

    use_tabpfn: bool = Field(
        default=False,
        description=(
            "Set true if the user explicitly requests TabPFN "
            "or comparison with TabPFN."
        )
    )

    prediction_length: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of future periods for forecasting."
    )

    frequency: Optional[str] = Field(
        default=None,
        description=(
            "Pandas frequency such as D, W, MS or ME. "
            "Use null when it should be inferred."
        )
    )


def create_prediction_plan(
    user_question: str,
    df: pd.DataFrame
) -> PredictionPlan:
    """Use Gemini to understand the user's prediction request."""

    model = get_gemini_model()

    structured_model = model.with_structured_output(
        PredictionPlan,
        method="json_schema"
    )

    column_information = [
        {
            "name": str(column),
            "dtype": str(df[column].dtype)
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

MODEL SELECTION:

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

Set use_tabpfn=true only if the user explicitly asks for TabPFN,
a TabPFN comparison, or a comparison including all prediction engines.

FORECASTING:

For forecasting, identify the exact date column and numerical
target column.

Interpret forecast horizons such as:
- next 7 days
- next 3 months
- next 12 periods

Use a suitable pandas frequency:
- Daily: D
- Weekly: W
- Monthly: MS
- Quarterly: QS
- Yearly: Y

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

    # Validate that Gemini selected real dataset columns.
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

    if pd.isna(value):
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)



def run_prediction_agent(
    user_question,
    df,
    prediction_config=None
):
    """
    Execute predictive analysis.

    If prediction_config is provided, use the exact
    configuration supplied by the Streamlit interface.

    Otherwise, use Gemini to generate the prediction plan.
    """

    try:

        # --------------------------------------------------
        # 1. Create or receive prediction plan
        # --------------------------------------------------

        if prediction_config:

            plan = PredictionPlan(
                **prediction_config
            )

            # Validate the supplied configuration.
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

            if plan.task_type in [
                "regression",
                "classification"
            ]:

                if not plan.selected_models:
                    raise ValueError(
                        "Select at least one machine learning model."
                    )

        else:

            # Preserve Gemini-based planning for chat requests.
            plan = create_prediction_plan(
                user_question,
                df
            )

        # --------------------------------------------------
        # 2. Execute forecasting
        # --------------------------------------------------

        if plan.task_type == "forecasting":

            result = forecast_time_series(
                df=df,
                date_column=plan.date_column,
                target_column=plan.target_column,
                prediction_length=plan.prediction_length,
                frequency=plan.frequency
            )

            result = make_json_safe(result)

            return {
                "success": True,
                "task_type": "forecasting",
                "target_column": plan.target_column,
                "date_column": plan.date_column,
                "engines": ["Chronos-2"],
                "results": {
                    "Chronos-2": result
                },
                "answer": (
                    f"Chronos-2 successfully generated "
                    f"{plan.prediction_length} future forecasts "
                    f"for '{plan.target_column}'."
                )
            }

        # --------------------------------------------------
        # 3. Execute scikit-learn models
        # --------------------------------------------------

        results = {}
        engines = []

        sklearn_result = train_and_compare_models(
            df=df,
            target_column=plan.target_column,
            task_type=plan.task_type,
            selected_models=(
                plan.selected_models
                if plan.selected_models
                else None
            )
        )

        results["scikit-learn"] = make_json_safe(
            sklearn_result
        )

        engines.append("scikit-learn")

        # --------------------------------------------------
        # 4. Optional TabPFN execution
        # --------------------------------------------------

        if plan.use_tabpfn:

            try:

                tabpfn_result = train_tabpfn(
                    df=df,
                    target_column=plan.target_column,
                    task_type=plan.task_type
                )

                results["TabPFN"] = make_json_safe(
                    tabpfn_result
                )

                engines.append("TabPFN")

            except Exception as error:

                results["TabPFN"] = {
                    "error": str(error)
                }

        # --------------------------------------------------
        # 5. Prepare response
        # --------------------------------------------------

        best_model = sklearn_result.get("best_model")
        overall_model = best_model

        if "TabPFN" in engines:
            tab_metrics = results["TabPFN"].get("metrics", {})
            sklearn_candidates = [
                model for model in sklearn_result.get("models", [])
                if not model.get("error")
            ]
            candidates = []
            for model_result in sklearn_candidates:
                metric_values = model_result.get("metrics", {})
                if plan.task_type == "regression" and "rmse" in metric_values:
                    candidates.append((metric_values["rmse"], model_result["model"]))
                elif plan.task_type == "classification" and "f1_weighted" in metric_values:
                    candidates.append((-metric_values["f1_weighted"], model_result["model"]))

            if plan.task_type == "regression" and "rmse" in tab_metrics:
                candidates.append((tab_metrics["rmse"], "TabPFN"))
            elif plan.task_type == "classification" and "f1_weighted" in tab_metrics:
                candidates.append((-tab_metrics["f1_weighted"], "TabPFN"))

            if candidates:
                overall_model = min(candidates)[1]

        answer = (
            f"Predictive modelling completed for "
            f"'{plan.target_column}'. "
            f"The best scikit-learn model was {best_model}."
        )

        if "TabPFN" in engines:
            answer += (
                f" The best overall model on the shared holdout set was "
                f"{overall_model}. TabPFN cross-validation results are included."
            )

        elif plan.use_tabpfn:

            answer += (
                " TabPFN execution failed. "
                "See the model error details."
            )

        return {
            "success": True,
            "task_type": plan.task_type,
            "target_column": plan.target_column,
            "engines": engines,
            "results": results,
            "answer": answer
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
            "answer": (
                "I couldn't complete the predictive analysis. "
                f"Reason: {error}"
            )
        }
