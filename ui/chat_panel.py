import logging

import uuid

import streamlit as st

@st.cache_resource

def get_compiled_graph():

    """

    Build the graph once and reuse it across Streamlit reruns.

    """
    from graph.workflow import build_graph

    return build_graph()

def initialize_chat_state():

    """

    Initialize the conversation-related session state.

    """

    if "messages" not in st.session_state:

        st.session_state["messages"] = []

    if "recent_queries" not in st.session_state:

        st.session_state["recent_queries"] = []

    if "thread_id" not in st.session_state:

        st.session_state["thread_id"] = str(uuid.uuid4())

def render_prediction_results(prediction_result,  key_prefix="prediction"):

    """

    Display predictive analytics results, model comparisons,

    cross-validation, feature diagnostics and forecasts.

    """
    import pandas as pd
    import plotly.graph_objects as go

    if not prediction_result:

        return

    if not prediction_result.get("success"):

        st.error(

            prediction_result.get(

                "error",

                "Prediction could not be completed."

            )

        )

        return

    task_type = prediction_result.get("task_type")

    target = prediction_result.get("target_column")

    engines = prediction_result.get("engines", [])

    st.divider()

    st.markdown("### Predictive Analytics Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Task", str(task_type).title())

    col2.metric("Target", target or "N/A")

    col3.metric("Model Engines", len(engines))

    if engines:

        st.caption("Engines used: " + ", ".join(engines))

    results = prediction_result.get("results", {})

    if not results:
        st.info(
            "The prediction process completed, but no model results "
            "were returned. Check the target column, available features "
            "and dataset size, then try again."
        )
        return

    best_result = None

    # ==================================================

    # Scikit-learn model comparison

    # ==================================================

    sklearn_result = results.get("scikit-learn")

    if sklearn_result:

        st.markdown("#### Model Comparison")

        best_model = sklearn_result.get("best_model")

        if best_model:

            st.info(f"Best scikit-learn model: {best_model}")

        # ----------------------------------------------

        # Training and testing information

        # ----------------------------------------------

        st.markdown("##### Dataset Evaluation Details")

        col1, col2, col3 = st.columns(3)

        col1.metric(

            "Training Rows",

            f'{sklearn_result.get("training_rows", "N/A"):,}'

            if isinstance(sklearn_result.get("training_rows"), int)

            else "N/A"

        )

        col2.metric(

            "Testing Rows",

            f'{sklearn_result.get("testing_rows", "N/A"):,}'

            if isinstance(sklearn_result.get("testing_rows"), int)

            else "N/A"

        )

        col3.metric(

            "Cross-validation Folds",

            sklearn_result.get("cross_validation_folds", "N/A")

        )

        # ----------------------------------------------

        # Build model comparison table

        # ----------------------------------------------

        model_rows = []

        for model in sklearn_result.get("models", []):

            row = {

                "Model": model.get("model", "Unknown")

            }

            metrics = model.get("metrics", {}).copy()

            metrics.pop("confusion_matrix", None)

            row.update(metrics)

            cv = model.get("cross_validation", {})

            if cv:

                row["CV Mean"] = cv.get("mean")

                row["CV Std"] = cv.get("std")

                row["CV Folds"] = cv.get("folds")

            if model.get("error"):

                row["Error"] = model["error"]

            model_rows.append(row)

        comparison_df = pd.DataFrame(model_rows)
        if comparison_df.empty:
            st.info(
                "No machine learning models produced usable results. "
                "Try another target column or review the dataset."
            )

        if not comparison_df.empty:

            st.dataframe(

                comparison_df,

                width="stretch",

                hide_index=True

            )

        # ----------------------------------------------

        # Cross-validation explanation

        # ----------------------------------------------

        st.markdown("##### Cross-validation Results")

        cv_rows = []

        for model in sklearn_result.get("models", []):

            cv = model.get("cross_validation")

            if cv:

                cv_rows.append({

                    "Model": model.get("model"),

                    "Mean Score": cv.get("mean"),

                    "Standard Deviation": cv.get("std"),

                    "Folds": cv.get("folds")

                })

        if cv_rows:

            cv_df = pd.DataFrame(cv_rows)

            st.dataframe(

                cv_df,

                width="stretch",

                hide_index=True

            )

            if task_type == "regression":

                st.caption(

                    "Regression cross-validation uses RMSE. "

                    "Lower mean RMSE indicates better performance, "

                    "while lower standard deviation indicates more "

                    "consistent performance across validation folds."

                )

            else:

                st.caption(

                    "Classification cross-validation uses weighted F1. "

                    "Higher mean F1 indicates better performance, "

                    "while lower standard deviation indicates more "

                    "consistent performance across validation folds."

                )

        # ----------------------------------------------

        # Model performance chart

        # ----------------------------------------------

        st.markdown("##### Model Performance Comparison")

        if not comparison_df.empty:

            if task_type == "regression":

                metric_name = "rmse"

                chart_title = "RMSE Comparison (Lower is Better)"

            else:

                metric_name = "f1_weighted"

                chart_title = "Weighted F1 Comparison (Higher is Better)"

            if metric_name in comparison_df.columns:

                chart_df = comparison_df[

                    ["Model", metric_name]

                ].dropna()

                if not chart_df.empty:

                    figure = go.Figure(

                        data=[

                            go.Bar(

                                x=chart_df["Model"],

                                y=chart_df[metric_name],

                                text=chart_df[metric_name].round(4),

                                textposition="auto"

                            )

                        ]

                    )

                    figure.update_layout(

                        title=chart_title,

                        xaxis_title="Model",

                        yaxis_title=metric_name.upper(),

                        template="plotly_white"

                    )

                    st.plotly_chart(

                        figure,

                        width="stretch",

                        key=f"{key_prefix}_model_performance"

                    )

        # ----------------------------------------------

        # Feature quality diagnostics

        # ----------------------------------------------

        st.markdown("##### Feature Quality Diagnostics")

        feature_warnings = sklearn_result.get(

            "feature_warnings", []

        )

        if feature_warnings:

            for warning in feature_warnings:

                st.warning(str(warning))

        else:

            st.success(

                "No feature-quality warnings were reported."

            )

        removed_constants = sklearn_result.get(

            "removed_constant_columns", []

        )

        removed_target_copies = sklearn_result.get(

            "removed_exact_target_copies", []

        )

        unsupported_columns = sklearn_result.get(

            "unsupported_columns", []

        )

        with st.expander("View preprocessing details"):

            st.write(

                "**Removed constant columns:**",

                removed_constants or "None"

            )

            st.write(

                "**Removed exact copies of target:**",

                removed_target_copies or "None"

            )

            st.write(

                "**Unsupported columns:**",

                unsupported_columns or "None"

            )

        # ----------------------------------------------

        # Find best model result

        # ----------------------------------------------

        best_result = next(

            (

                model

                for model in sklearn_result.get("models", [])

                if model.get("model") == best_model

                and model.get("predictions") is not None

                and model.get("actual_values") is not None

            ),

            None

        )

        
        # ----------------------------------------------
        # Actual vs predicted values
        # ----------------------------------------------

        st.markdown("#### Actual vs Predicted Values")

        comparison = None

        if best_result:

            actual = best_result.get("actual_values") or []
            predicted = best_result.get("predictions") or []

            # Ensure both arrays contain matching observations.
            if len(actual) > 0 and len(predicted) > 0:

                if len(actual) != len(predicted):
                    st.warning(
                        "Actual and predicted values have different "
                        "lengths. The comparison cannot be displayed."
                    )

                else:

                    if task_type == "classification":

                        comparison = pd.DataFrame({
                            "Actual Class": actual,
                            "Predicted Class": predicted
                        })

                    else:

                        comparison = pd.DataFrame({
                            "Actual": actual,
                            "Predicted": predicted
                        })

                    # Display comparison only when valid.
                    if comparison is not None and not comparison.empty:

                        st.dataframe(
                            comparison,
                            width="stretch",
                            hide_index=True
                        )

                    # Regression actual vs predicted plot.
                    if task_type == "regression":

                        numeric_actual = pd.to_numeric(
                            pd.Series(actual),
                            errors="coerce"
                        )

                        numeric_predicted = pd.to_numeric(
                            pd.Series(predicted),
                            errors="coerce"
                        )

                        valid_pairs = pd.DataFrame({
                            "Actual": numeric_actual,
                            "Predicted": numeric_predicted
                        }).dropna()

                        if not valid_pairs.empty:

                            figure = go.Figure()

                            figure.add_trace(
                                go.Scatter(
                                    x=valid_pairs["Actual"],
                                    y=valid_pairs["Predicted"],
                                    mode="markers",
                                    name="Predictions"
                                )
                            )

                            minimum = min(
                                valid_pairs["Actual"].min(),
                                valid_pairs["Predicted"].min()
                            )

                            maximum = max(
                                valid_pairs["Actual"].max(),
                                valid_pairs["Predicted"].max()
                            )

                            figure.add_trace(
                                go.Scatter(
                                    x=[minimum, maximum],
                                    y=[minimum, maximum],
                                    mode="lines",
                                    name="Perfect prediction"
                                )
                            )

                            figure.update_layout(
                                title="Actual vs Predicted",
                                xaxis_title="Actual Values",
                                yaxis_title="Predicted Values",
                                template="plotly_white"
                            )

                            st.plotly_chart(
                                figure,
                                width="stretch",
                                key=f"{key_prefix}_actual_vs_predicted"
                            )

            else:

                st.info(
                    "Actual vs predicted values are unavailable "
                    "for the selected model."
                )

        else:

            st.info(
                "No valid best-model predictions are available "
                "to display this comparison."
            )

        # ----------------------------------------------

        # Classification confusion matrix

        # ----------------------------------------------

        if task_type == "classification" and best_result:

            metrics = best_result.get("metrics", {})

            matrix = metrics.get("confusion_matrix")

            if matrix is not None:

                st.markdown("#### Confusion Matrix")

                labels = best_result.get(

                    "class_labels",

                    list(range(len(matrix)))

                )

                figure = go.Figure(

                    data=go.Heatmap(

                        z=matrix,

                        x=labels,

                        y=labels,

                        text=matrix,

                        texttemplate="%{text}",

                        colorscale="Blues"

                    )

                )

                figure.update_layout(

                    xaxis_title="Predicted Class",

                    yaxis_title="Actual Class",

                    template="plotly_white"

                )

                st.plotly_chart(

                    figure,

                    width="stretch",

                    key=f"{key_prefix}_confusion_matrix"

                )

        # ==================================================
    # Holt-Winters forecasting results
    # ==================================================

    forecast_result = results.get("Holt-Winters")

    if forecast_result:

        st.markdown("#### Holt-Winters Forecast")

        if forecast_result.get("error"):

            st.error(forecast_result["error"])

        else:

            evaluation = forecast_result.get("evaluation") or {}

            if evaluation:

                col1, col2 = st.columns(2)

                col1.metric(
                    "Evaluation MAE",
                    f'{evaluation.get("mae", 0):,.2f}'
                )

                col2.metric(
                    "Evaluation RMSE",
                    f'{evaluation.get("rmse", 0):,.2f}'
                )

            config = forecast_result.get("model_configuration", {})

            st.caption(
                "Model: Holt-Winters Exponential Smoothing | "
                f"Trend: {config.get('trend') or 'None'} | "
                f"Seasonality: {config.get('seasonal') or 'None'}"
            )

            forecast_data = forecast_result.get("forecast", [])

            if forecast_data:

                forecast_df = pd.DataFrame(forecast_data)

                st.dataframe(
                    forecast_df,
                    width="stretch",
                    hide_index=True
                )

                if (
                    "date" in forecast_df.columns
                    and "forecast" in forecast_df.columns
                ):

                    figure = go.Figure()

                    figure.add_trace(
                        go.Scatter(
                            x=forecast_df["date"],
                            y=forecast_df["forecast"],
                            mode="lines+markers",
                            name="Forecast"
                        )
                    )

                    if (
                        "lower_bound" in forecast_df.columns
                        and "upper_bound" in forecast_df.columns
                    ):

                        figure.add_trace(
                            go.Scatter(
                                x=forecast_df["date"],
                                y=forecast_df["upper_bound"],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip"
                            )
                        )

                        figure.add_trace(
                            go.Scatter(
                                x=forecast_df["date"],
                                y=forecast_df["lower_bound"],
                                mode="lines",
                                fill="tonexty",
                                name="Approximate prediction interval",
                                line=dict(width=0)
                            )
                        )

                    figure.update_layout(
                        title="Future Forecast",
                        xaxis_title="Date",
                        yaxis_title=target or "Forecast",
                        template="plotly_white"
                    )

                    st.plotly_chart(
                        figure,
                        width="stretch",
                        key=f"{key_prefix}_holt_winters_forecast"
                    )

            else:
                st.info("No forecast records were returned.")


def render_advanced_results(advanced_result):
    """Render evidence-backed advanced analytical insights."""

    if not advanced_result:
        return

    if not advanced_result.get("success"):
        st.warning(advanced_result.get("error", "Advanced analysis failed."))
        return

    def format_label(value):
        return str(value).replace("_", " ").replace("-", " ").title()

    def render_structured_data(data):
        """Render nested evidence as tables and readable key/value rows."""
        if data is None:
            st.caption("No data available.")
            return

        if isinstance(data, dict):
            if not data:
                st.caption("No supporting information available.")
                return

            scalar_items = {
                key: value for key, value in data.items()
                if not isinstance(value, (dict, list, tuple))
            }
            nested_items = {
                key: value for key, value in data.items()
                if isinstance(value, (dict, list, tuple))
            }

            if scalar_items:
                rows = []
                for key, value in scalar_items.items():
                    if isinstance(value, float):
                        display_value = f"{value:,.4f}"
                    elif isinstance(value, (int, bool)):
                        display_value = str(value)
                    else:
                        display_value = str(value)
                    rows.append({"Metric / Detail": format_label(key), "Value": display_value})
                st.table(rows)

            for key, value in nested_items.items():
                with st.expander(format_label(key)):
                    render_structured_data(value)
            return

        if isinstance(data, (list, tuple)):
            if not data:
                st.caption("No records available.")
            elif all(isinstance(item, dict) for item in data):
                import pandas as pd
                st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
            elif all(not isinstance(item, (dict, list, tuple)) for item in data):
                st.table([
                    {"No.": index, "Value": str(value)}
                    for index, value in enumerate(data, start=1)
                ])
            else:
                for index, item in enumerate(data, start=1):
                    with st.expander(f"Item {index}"):
                        render_structured_data(item)
            return

        st.write(data)

    # Analysis warnings
    for section, error in advanced_result.get("errors", {}).items():
        st.warning(f"{format_label(section)} was skipped: {error}")

    insights_data = advanced_result.get("insights", {}) or {}
    insights = insights_data.get("insights", []) or []

    st.markdown("#### Advanced analytics report")
    col1, col2 = st.columns(2)
    col1.metric("Insights generated", insights_data.get("total_insights", len(insights)))
    col2.metric("High-priority findings", insights_data.get("high_priority_count", 0))

    if not insights:
        st.info("No notable findings were generated from the available results.")
    else:
        for index, item in enumerate(insights, start=1):
            priority = str(item.get("priority", "medium")).strip().lower()
            priority_styles = {
                "high": ("#FEE2E2", "#991B1B", "#DC2626", "HIGH PRIORITY"),
                "medium": ("#FEF3C7", "#92400E", "#D97706", "MEDIUM PRIORITY"),
                "low": ("#DCFCE7", "#166534", "#16A34A", "LOW PRIORITY"),
            }
            bg, fg, border, label = priority_styles.get(
                priority,
                ("#E2E8F0", "#334155", "#64748B", f"{priority.upper()} PRIORITY"),
            )
            title = item.get("title", f"Finding {index}")

            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(
                    f'<span style="display:inline-block; padding:4px 10px; '
                    f'border:1px solid {border}; border-radius:999px; '
                    f'background:{bg}; color:{fg}; font-size:0.78rem; '
                    f'font-weight:800; letter-spacing:.35px;">'
                    f'{format_label(item.get("category", "analysis"))} '
                    f'&nbsp;·&nbsp; {label}</span>',
                    unsafe_allow_html=True,
                )
                description = item.get("description")
                if description:
                    st.write(description)
                if item.get("recommendation"):
                    st.markdown(f"**Suggested next step:** {item['recommendation']}")
                evidence = item.get("evidence")
                if evidence:
                    with st.expander("View supporting evidence"):
                        render_structured_data(evidence)

    raw_results = advanced_result.get("results", {})
    if raw_results:
        with st.expander("View detailed analytical results"):
            if isinstance(raw_results, dict):
                for section, result in raw_results.items():
                    st.markdown(f"##### {format_label(section)}")
                    render_structured_data(result)
                    st.divider()
            else:
                render_structured_data(raw_results)



logger = logging.getLogger(__name__)


def get_friendly_error(error_code=None):
    """Return a safe message based on the error category."""

    messages = {
        "API_AUTHENTICATION_ERROR": (
            "The AI service could not authenticate your request. "
            "Please check the application configuration."
        ),
        "API_RATE_LIMIT": (
            "The AI service is receiving too many requests. "
            "Please wait a moment and try again."
        ),
        "REQUEST_TIMEOUT": (
            "The request took too long to complete. "
            "Please try again with a simpler question."
        ),
        "SERVICE_UNAVAILABLE": (
            "The AI service is temporarily unavailable. "
            "Please try again shortly."
        ),
        "COLUMN_OR_DATA_ERROR": (
            "The requested column or data could not be found. "
            "Please check your dataset and try again."
        ),
        "MEMORY_ERROR": (
            "This operation requires more memory than is currently "
            "available. Try a smaller dataset."
        ),
        "UNSUPPORTED_REQUEST": (
            "I can help with dataset profiling, data analysis, "
            "visualizations, advanced analytics and predictive analytics."
        ),
        "AGENT_EXECUTION_ERROR": (
            "The requested operation could not be completed. "
            "Please review your question and try again."
        ),
    }

    return messages.get(
        error_code,
        "Something went wrong while processing your request. "
        "Please try again."
    )


def render_safe_chart(chart_json, key):
    """Render a generated Plotly chart without crashing the chat."""
    import plotly.io as pio
    if not chart_json:
        return

    try:
        figure = pio.from_json(chart_json)

        st.plotly_chart(
            figure,
            width="stretch",
            key=key
        )

    except Exception as error:
        logger.warning(
            "Chart rendering failed: %s",
            type(error).__name__
        )

        st.warning(
            "The analysis was completed, but its chart "
            "could not be displayed. Please try generating it again."
        )

def render_chat_styles():
    """Apply the Aurora theme to the chat interface."""

    st.markdown(
        """
        <style>
        /* Chat section heading */
        .aurora-chat-header {
            padding: 18px 20px;
            margin-bottom: 18px;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            background: linear-gradient(
                110deg,
                #F0FDFF 0%,
                #F8FAFF 55%,
                #F5F3FF 100%
            );
        }

        .aurora-chat-title {
            font-size: 21px;
            font-weight: 750;
            color: #172033;
            margin: 0 0 5px 0;
        }

        .aurora-chat-subtitle {
            font-size: 13px;
            color: #64748B;
            margin: 0;
        }

        /* Shared message layout */
        div[data-testid="stChatMessage"] {
            padding: 15px 17px;
            margin-bottom: 14px;
            border-radius: 15px;
            border: 1px solid #E8EDF5;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
        }

        /* Assistant message */
        div[data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-assistant"]
        ) {
            background: #FFFFFF;
            border-left: 3px solid #06B6D4;
        }

        /* User message */
        div[data-testid="stChatMessage"]:has(
            [data-testid="chatAvatarIcon-user"]
        ) {
            background: linear-gradient(
                110deg,
                #EFF8FF,
                #F5F3FF
            );
            border-color: #DCE8FA;
        }

        /* Message typography */
        div[data-testid="stChatMessage"] p {
            color: #263449;
            line-height: 1.7;
            font-size: 14px;
        }

        div[data-testid="stChatMessage"] h1,
        div[data-testid="stChatMessage"] h2,
        div[data-testid="stChatMessage"] h3 {
            color: #172033;
        }

        /* Chat input */
        div[data-testid="stChatInput"] {
            border: 1px solid #CBD5E1;
            border-radius: 16px;
            background: #FFFFFF;
            box-shadow: 0 3px 12px rgba(59, 130, 246, 0.06);
            transition: border-color 0.2s ease,
                        box-shadow 0.2s ease;
        }

        div[data-testid="stChatInput"]:focus-within {
            border-color: #06B6D4;
            box-shadow:
                0 0 0 3px rgba(6, 182, 212, 0.10),
                0 4px 16px rgba(59, 130, 246, 0.10);
        }

        div[data-testid="stChatInput"] textarea {
            color: #172033;
            font-size: 14px;
        }

        /* Analytical metadata */
        div[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
            color: #64748B;
            font-size: 11px;
        }

        /* Tables inside chat */
        div[data-testid="stChatMessage"] div[data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            overflow: hidden;
        }

        /* Mobile spacing */
        @media (max-width: 768px) {
            div[data-testid="stChatMessage"] {
                padding: 12px;
                border-radius: 12px;
            }

            .aurora-chat-header {
                padding: 15px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_chat_panel():

    initialize_chat_state()
    render_chat_styles()

    chat_header_col, chat_action_col = st.columns(
    [4, 1],
    vertical_alignment="center",
)

    with chat_header_col:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="
                    font-size:26px;
                    font-weight:800;
                    background:linear-gradient(
                        90deg,#06B6D4,#3B82F6,#8B5CF6
                    );
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                ">
                    ✦ Ask Aurora AI
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Your intelligent AI data analyst. Ask questions, "
            "discover insights and explore your dataset."
        )

    with chat_action_col:
        if st.button(
            "＋ New Chat",
            key="dashboard_new_chat_button",
            width="stretch",
        ):
            st.session_state["messages"] = []
            st.session_state["recent_queries"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()

    # -------------------------------------------------
    # DISPLAY EXISTING CONVERSATION
    # -------------------------------------------------

    for message_index, message in enumerate(
        st.session_state["messages"]
    ):

        with st.chat_message(message["role"]):

            st.markdown(message.get("content", ""))

            # Safe Plotly chart rendering.
            if message.get("chart_json"):

                render_safe_chart(
                    message["chart_json"],
                    key=f"chat_chart_{message_index}"
                )

            if message.get("operation"):

                st.caption(
                    f"Analytical operation: "
                    f"{message['operation']}"
                )

            if message.get("agent"):

                st.caption(
                    f"Handled by: {message['agent']} agent"
                )

            if message.get("advanced_result"):

                render_advanced_results(
                    message["advanced_result"]
                )

            if message.get("prediction_result"):

                render_prediction_results(
                    message["prediction_result"],
                    key_prefix=f"chat_prediction_{message_index}"
                )

    # -------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------

    user_query = st.chat_input(
        "Ask something about your dataset..."
    )

    if not user_query:
        return

    # -------------------------------------------------
    # DATASET VALIDATION
    # -------------------------------------------------

    df = st.session_state.get("dataset")

    if df is None or df.empty:

        st.warning(
            "Please upload a valid CSV or Excel dataset "
            "before asking questions."
        )

        return

    if st.session_state.get("processing_query", False):

        st.info(
            "Your previous request is still being processed."
        )

        return

    # -------------------------------------------------
    # SAVE USER MESSAGE
    # -------------------------------------------------

    from datetime import datetime

    st.session_state["messages"].append({

        "role": "user",
        "content": user_query

    })

    st.session_state["recent_queries"].append({

        "query": user_query,

        "timestamp": datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        )

    })

    # Keep recent history limited.
    st.session_state["recent_queries"] = (
        st.session_state["recent_queries"][-100:]
    )

    # -------------------------------------------------
    # LANGGRAPH EXECUTION
    # -------------------------------------------------

    st.session_state["processing_query"] = True

    try:

        graph = get_compiled_graph()

        with st.spinner(
            "AI Data Analyst is processing your request. "
            "The system is selecting an agent and analysing your data..."
        ):

            result = graph.invoke(

                {
                    "user_question": user_query,

                    # Clear previous turn outputs.
                    "route": None,

                    "analysis_result": None,

                    "advanced_result": None,

                    "advanced_config": None,

                    "prediction_result": None,

                    "prediction_config": None,

                    "answer": "",

                    "error": "",

                    "error_code": "",

                    "error_message": "",

                    "chart_json": None,

                    "chart_plan": None,
                },

                config={

                    "configurable": {

                        "thread_id": (
                            st.session_state["thread_id"]
                        )

                    }

                },

                context={

                    "dataframe": df

                }

            )

        # -------------------------------------------------
        # PROCESS GRAPH RESPONSE
        # -------------------------------------------------

        answer = result.get(
            "answer",
            "The system could not generate a response."
        )

        error_code = result.get("error_code")

        error_message = result.get("error_message")

        # Use a friendly error message when an agent fails.
        if error_code and error_code != "UNSUPPORTED_REQUEST":

            answer = (
                error_message
                or get_friendly_error(error_code)
            )

        elif error_code == "UNSUPPORTED_REQUEST":

            answer = (
                answer
                or get_friendly_error(error_code)
            )

        agent = result.get("route")

        assistant_message = {

            "role": "assistant",

            "content": answer,

            "agent": agent,

            "chart_json": result.get("chart_json"),

            "chart_plan": result.get("chart_plan"),

            "operation": result.get("operation"),

            "prediction_result": result.get(
                "prediction_result"
            ),

            "advanced_result": result.get(
                "advanced_result"
            )

        }

        # IMPORTANT:
        # Preserve this append operation.
        # Otherwise, the assistant response disappears.

        st.session_state["messages"].append(
            assistant_message
        )

        # Show a visible warning for failed operations.
        if error_code and error_code != "UNSUPPORTED_REQUEST":

            st.warning(
                "The request could not be completed. "
                "You can review the message above and try again."
            )

    except Exception as error:

        # Log exception type, not the uploaded data or credentials.
        logger.exception(
            "Chat workflow execution failed: %s",
            type(error).__name__
        )

        friendly_message = (
            "I couldn't complete your request because the "
            "analysis workflow encountered an unexpected issue. "
            "Please try again. If the problem continues, "
            "check the application logs."
        )

        st.session_state["messages"].append({

            "role": "assistant",

            "content": friendly_message

        })

        st.error(friendly_message)

    finally:

        st.session_state["processing_query"] = False

    # -------------------------------------------------
    # REFRESH CHAT DISPLAY
    # -------------------------------------------------

    st.rerun()

def render_page_chat(
    page_key,
    title,
    placeholder,
    context_instruction,
    result_context=None,
):
    """
    Render an isolated conversation for a specific application page.
    Each page has its own messages and LangGraph thread.
    """

    messages_key = f"{page_key}_messages"
    thread_key = f"{page_key}_thread_id"

    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    if thread_key not in st.session_state:
        st.session_state[thread_key] = str(uuid.uuid4())

    messages = st.session_state[messages_key]

    st.divider()
    header_col, action_col = st.columns([4, 1])

    with header_col:
        st.subheader(title)
    with action_col:
        if st.button(
            "New Chat",
            key=f"{page_key}_new_chat",
            width="stretch",
        ):
            st.session_state[messages_key] = []
            st.session_state[thread_key] = str(uuid.uuid4())
            st.rerun()

    st.caption(context_instruction)

    # Display this page's conversation only.
    for index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message.get("content", ""))

            if message.get("chart_json"):
                render_safe_chart(
                    message["chart_json"],
                    key=f"{page_key}_chat_chart_{index}",
                )

            if message.get("prediction_result"):
                render_prediction_results(
                    message["prediction_result"],
                    key_prefix=f"{page_key}_prediction_{index}",
                )

            if message.get("advanced_result"):
                render_advanced_results(
                    message["advanced_result"]
                )

    user_query = st.chat_input(
        placeholder,
        key=f"{page_key}_chat_input",
    )

    if not user_query:
        return

    df = st.session_state.get("dataset")

    if df is None or df.empty:
        st.warning("Please upload a valid dataset first.")
        return

    # Keep the page context explicit so the supervisor can
    # select the appropriate specialist agent.
    contextual_question = (
        f"Application page: {page_key}. "
        f"Context: {context_instruction}. "
        f"User question: {user_query}"
    )

    if result_context:
        contextual_question += (
            f"\nLatest page result summary: {result_context}"
        )

    messages.append({
        "role": "user",
        "content": user_query,
    })

    try:
        graph = get_compiled_graph()

        with st.spinner("The AI agent is analysing your question..."):
            result = graph.invoke(
                {
                    "user_question": contextual_question,
                    "route": None,
                    "analysis_result": None,
                    "advanced_result": None,
                    "advanced_config": None,
                    "prediction_result": None,
                    "prediction_config": None,
                    "answer": "",
                    "error": "",
                    "error_code": "",
                    "error_message": "",
                    "chart_json": None,
                    "chart_plan": None,
                },
                config={
                    "configurable": {
                        "thread_id": st.session_state[thread_key]
                    }
                },
                context={"dataframe": df},
            )

        answer = (
            result.get("error_message")
            or result.get("answer")
            or "The system could not generate a response."
        )

        error_code = result.get("error_code")

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "agent": result.get("route"),
            "chart_json": result.get("chart_json"),
            "operation": result.get("operation"),
            "prediction_result": result.get("prediction_result"),
            "advanced_result": result.get("advanced_result"),
        }

        messages.append(assistant_message)

        if error_code and error_code != "UNSUPPORTED_REQUEST":
            st.warning(
                "The request could not be completed. "
                "Please review the response and try again."
            )

    except Exception as error:
        logger.exception(
            "Page-specific chat failed for %s: %s",
            page_key,
            type(error).__name__,
        )

        messages.append({
            "role": "assistant",
            "content": (
                "I couldn't complete your request because an "
                "unexpected issue occurred. Please try again."
            ),
        })

    st.rerun()