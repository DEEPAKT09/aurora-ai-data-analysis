
import logging

from langgraph.runtime import Runtime

from agents.visualization_agent import generate_visualization
from agents.supervisor import route_user_question
from agents.ai_analyst import ask_data_analyst
from agents.prediction_agent import run_prediction_agent
from agents.advanced_analytics_agent import run_advanced_analytics_agent

from graph.state import GraphState, GraphContext


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# ERROR CLASSIFICATION
# ---------------------------------------------------------

def classify_error(error):
    """
    Convert technical exceptions into safe, understandable
    user-facing error categories.
    """

    message = str(error).lower()

    if any(term in message for term in [
        "api key",
        "api_key",
        "authentication",
        "unauthorized",
        "permission denied",
        "invalid credentials",
    ]):
        return (
            "API_AUTHENTICATION_ERROR",
            "The AI service could not authenticate your request. "
            "Please check the application configuration."
        )

    if any(term in message for term in [
        "rate limit",
        "resource exhausted",
        "quota",
        "429",
    ]):
        return (
            "API_RATE_LIMIT",
            "The AI service is receiving too many requests. "
            "Please wait a moment and try again."
        )

    if any(term in message for term in [
        "timeout",
        "timed out",
        "deadline exceeded",
    ]):
        return (
            "REQUEST_TIMEOUT",
            "The request took too long to complete. "
            "Please try again with a smaller or simpler request."
        )

    if any(term in message for term in [
        "connection",
        "network",
        "unavailable",
        "503",
        "502",
    ]):
        return (
            "SERVICE_UNAVAILABLE",
            "The AI service is temporarily unavailable. "
            "Please try again shortly."
        )

    if any(term in message for term in [
        "column",
        "keyerror",
        "not found",
        "does not exist",
    ]):
        return (
            "COLUMN_OR_DATA_ERROR",
            "The requested column or data could not be found. "
            "Please check the column names in your uploaded dataset."
        )

    if any(term in message for term in [
        "memory",
        "out of memory",
    ]):
        return (
            "MEMORY_ERROR",
            "The operation requires more memory than is currently "
            "available. Try a smaller dataset or a simpler operation."
        )

    return (
        "AGENT_EXECUTION_ERROR",
        "The requested operation could not be completed. "
        "Please review your question and dataset, then try again."
    )


def build_error_response(error, agent_name):
    """
    Log technical details for server-side debugging while
    returning a safe message to the user.
    """

    logger.exception(
        "Error in %s agent",
        agent_name,
        exc_info=error,
    )

    error_code, user_message = classify_error(error)

    return {
        "error": user_message,
        "error_code": error_code,
        "error_message": user_message,
        "answer": user_message,
    }


# ---------------------------------------------------------
# SUPERVISOR NODE
# ---------------------------------------------------------

def supervisor_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Route the user request to the appropriate specialist."""

    try:
        df = runtime.context["dataframe"]

        decision = route_user_question(
            state["user_question"],
            df
        )

        return {
            "route": decision.route,
            "answer": f"Request routed to {decision.route} agent.",
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return {
            "route": "unsupported",
            **build_error_response(error, "supervisor"),
        }


# ---------------------------------------------------------
# PROFILING NODE
# ---------------------------------------------------------

def profiling_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Generate a basic dataset profile."""

    try:
        df = runtime.context["dataframe"]

        result = {
            "total_records": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": [
                str(column) for column in df.columns
            ],
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_types": {
                str(column): str(df[column].dtype)
                for column in df.columns
            }
        }

        return {
            "analysis_result": result,
            "answer": (
                f"Dataset profiling completed. The dataset contains "
                f"{result['total_records']} records and "
                f"{result['total_columns']} columns, with "
                f"{result['missing_values']} missing values and "
                f"{result['duplicate_rows']} duplicate rows."
            ),
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return build_error_response(error, "profiling")


# ---------------------------------------------------------
# ANALYSIS NODE
# ---------------------------------------------------------

def analysis_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Run the standard AI data analyst."""

    try:
        df = runtime.context["dataframe"]

        result = ask_data_analyst(
            state["user_question"],
            df
        )

        return {
            "analysis_result": result,
            "answer": result.get(
                "answer",
                "The analysis was completed, but no explanation was returned."
            ),
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return build_error_response(error, "analysis")


# ---------------------------------------------------------
# VISUALIZATION NODE
# ---------------------------------------------------------

def visualization_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Generate a visualization from the user's request."""

    try:
        df = runtime.context["dataframe"]

        result = generate_visualization(
            state["user_question"],
            df
        )

        chart_plan = {
            "chart_type": result["chart_type"],
            "x_column": result["x_column"],
            "y_column": result["y_column"],
            "color_column": result["color_column"],
            "aggregation": result["aggregation"],
            "explanation": result["explanation"],
        }

        return {
            "analysis_result": chart_plan,
            "chart_plan": chart_plan,
            "chart_json": result["chart_json"],
            "answer": (
                f"{result['explanation']}\n\n"
                f"Chart type: {result['chart_type']}"
            ),
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return build_error_response(error, "visualization")


# ---------------------------------------------------------
# PREDICTION NODE
# ---------------------------------------------------------

def prediction_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """
    Execute predictive analytics using either dashboard
    configuration or AI-generated configuration.
    """

    try:
        df = runtime.context["dataframe"]

        result = run_prediction_agent(
            user_question=state["user_question"],
            df=df,
            prediction_config=state.get("prediction_config")
        )

        if not result.get("success", False):
                original_error = result.get("error", "")

                error_code, user_message = classify_error(
                    original_error or "Prediction could not be completed."
                )

                # Store only safe information in the graph state.
                safe_result = {
                    "success": False,
                    "answer": user_message,
                    "error": user_message,
                    "error_code": error_code,
                }

                return {
                    "prediction_result": safe_result,
                    "analysis_result": safe_result,
                    "answer": user_message,
                    "error": user_message,
                    "error_code": error_code,
                    "error_message": user_message,
                }

        return {
            "prediction_result": result,
            "analysis_result": result,
            "answer": result.get(
                "answer",
                "Predictive analysis completed."
            ),
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return {
            "prediction_result": {
                "success": False,
                "answer": "Prediction could not be completed.",
            },
            "analysis_result": {},
            **build_error_response(error, "prediction"),
        }


# ---------------------------------------------------------
# ADVANCED ANALYSIS NODE
# ---------------------------------------------------------

def advanced_analysis_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Run advanced analytics and evidence-based insights."""

    try:
        df = runtime.context["dataframe"]

        result = run_advanced_analytics_agent(
            state["user_question"],
            df,
            config=state.get("advanced_config")
        )

        return {
            "advanced_result": result,
            "analysis_result": result,
            "answer": result.get(
                "answer",
                "Advanced analysis completed."
            ),
            "error": "",
            "error_code": "",
            "error_message": "",
        }

    except Exception as error:
        return build_error_response(
            error,
            "advanced_analysis"
        )


# ---------------------------------------------------------
# UNSUPPORTED REQUEST NODE
# ---------------------------------------------------------

def unsupported_node(
    state: GraphState,
    runtime: Runtime[GraphContext]
):
    """Respond to unsupported requests or preserve routing errors."""

    if state.get("error_code"):
        return {
            "answer": state.get(
                "error_message",
                "The request could not be processed."
            )
        }

    return {
        "answer": (
            "I can help with dataset profiling, statistical analysis, "
            "visualizations, advanced analytics and predictive analytics. "
            "Please ask a question related to your uploaded dataset."
        ),
        "error": "",
        "error_code": "UNSUPPORTED_REQUEST",
        "error_message": (
            "This request is outside the supported data-analysis tasks."
        ),
    }