
from typing import Any, Literal
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # User input and routing
    user_question: str

    route: Literal[
        "profiling",
        "analysis",
        "advanced_analysis",
        "visualization",
        "prediction",
        "unsupported",
    ]

    # Agent results
    analysis_result: dict[str, Any]
    advanced_result: dict[str, Any]
    advanced_config: dict[str, Any]

    prediction_result: dict[str, Any]
    prediction_config: dict[str, Any]

    # Response and visualization
    answer: str
    chart_json: str
    chart_plan: dict[str, Any]

    # Standardized error information
    error: str
    error_code: str
    error_message: str


class GraphContext(TypedDict):
    dataframe: Any
