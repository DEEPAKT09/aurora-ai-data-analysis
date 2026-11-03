import logging

import uuid

import streamlit as st

import plotly.io as pio

import pandas as pd

import plotly.graph_objects as go

from graph.workflow import build_graph

@st.cache_resource

def get_compiled_graph():

    """

    Build the graph once and reuse it across Streamlit reruns.

    """

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

    # TabPFN results

    # ==================================================

    tabpfn_result = results.get("TabPFN")

    if tabpfn_result:

        st.markdown("#### TabPFN Results")

        if tabpfn_result.get("error"):

            st.warning(

                f"TabPFN could not run: "

                f"{tabpfn_result['error']}"

            )

        else:

            metrics = tabpfn_result.get("metrics", {})

            if metrics:
                metric_cols = st.columns(len(metrics))
                for index, (metric_name, metric_value) in enumerate(metrics.items()):
                    if isinstance(metric_value, (int, float)):
                        metric_cols[index].metric(
                            metric_name.replace("_", " ").upper(),
                            f"{metric_value:.4f}"
                        )

            # TabPFN holdout predictions and residual diagnostics.
            actual = tabpfn_result.get("actual_values", [])
            predicted = tabpfn_result.get("predictions", [])
            if actual and predicted and task_type == "regression":
                plot_df = pd.DataFrame({
                    "Actual": actual,
                    "Predicted": predicted
                })
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_df["Actual"], y=plot_df["Predicted"],
                    mode="markers", name="TabPFN predictions"
                ))
                low = min(float(plot_df["Actual"].min()), float(plot_df["Predicted"].min()))
                high = max(float(plot_df["Actual"].max()), float(plot_df["Predicted"].max()))
                fig.add_trace(go.Scatter(
                    x=[low, high], y=[low, high],
                    mode="lines", name="Ideal prediction"
                ))
                fig.update_layout(
                    title="TabPFN: Actual vs Predicted",
                    xaxis_title="Actual", yaxis_title="Predicted"
                )
                st.plotly_chart(
                    fig, width="stretch",
                    key=f"{key_prefix}_tabpfn_actual_predicted"
                )

                residuals = [
                    float(a) - float(p) for a, p in zip(actual, predicted)
                ]
                residual_fig = go.Figure()
                residual_fig.add_trace(go.Scatter(
                    x=predicted, y=residuals, mode="markers",
                    name="Residuals"
                ))
                residual_fig.add_hline(y=0, line_dash="dash")
                residual_fig.update_layout(
                    title="TabPFN: Residual Plot",
                    xaxis_title="Predicted", yaxis_title="Residual"
                )
                st.plotly_chart(
                    residual_fig, width="stretch",
                    key=f"{key_prefix}_tabpfn_residuals"
                )

    # Unified holdout comparison: same random seed, target-cleaned rows,
    # test size and stratification are used by both model runners.
    if sklearn_result and tabpfn_result and not tabpfn_result.get("error"):
        st.markdown("#### Unified Model Comparison")
        unified_rows = []
        for model_result in sklearn_result.get("models", []):
            if model_result.get("error"):
                continue
            row = {"Model": model_result.get("model"), "Engine": "scikit-learn"}
            row.update({
                key: value for key, value in model_result.get("metrics", {}).items()
                if key != "confusion_matrix"
            })
            cv = model_result.get("cross_validation", {})
            row["CV Mean"] = cv.get("mean")
            row["CV Std"] = cv.get("std")
            unified_rows.append(row)

        tabpfn_row = {"Model": "TabPFN", "Engine": "TabPFN"}
        tabpfn_row.update(tabpfn_result.get("metrics", {}))
        tab_cv = tabpfn_result.get("cross_validation", {})
        tabpfn_row["CV Mean"] = tab_cv.get("mean")
        tabpfn_row["CV Std"] = tab_cv.get("std")
        unified_rows.append(tabpfn_row)

        st.dataframe(
            pd.DataFrame(unified_rows),
            width="stretch",
            hide_index=True
        )

        # Select the winner using the same holdout metric across engines.
        candidates = []
        for row in unified_rows:
            if task_type == "regression" and isinstance(row.get("rmse"), (int, float)):
                candidates.append((row["rmse"], row["Model"]))
            elif task_type == "classification" and isinstance(row.get("f1_weighted"), (int, float)):
                candidates.append((-row["f1_weighted"], row["Model"]))
        if candidates:
            winner = min(candidates)[1]
            st.success(
                f"Best overall model on the shared holdout set: {winner}"
            )

    # ==================================================

    # Chronos-2 forecasting results

    # ==================================================

    chronos_result = results.get("Chronos-2")

    if chronos_result:

        st.markdown("#### Chronos-2 Forecast")

        if chronos_result.get("error"):

            st.error(chronos_result["error"])

        else:

            evaluation = chronos_result.get(

                "evaluation", {}

            )

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

            forecast_data = chronos_result.get(

                "forecast", []

            )

            if forecast_data:

                forecast_df = pd.DataFrame(forecast_data)
                if forecast_df.empty:
                    st.info(
                        "Chronos-2 did not return usable forecast records."
                    )
                
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

                                name="Prediction interval",

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

                        key=f"{key_prefix}_chronos_forecast"

                    )

def render_advanced_results(advanced_result):

    """Render evidence-backed advanced analytical insights."""

    if not advanced_result:

        return

    if not advanced_result.get("success"):

        st.warning(advanced_result.get("error", "Advanced analysis failed."))

        return

    for section, error in advanced_result.get("errors", {}).items():

        st.warning(f"{section.replace('_', ' ').title()} was skipped: {error}")

    insights_data = advanced_result.get("insights", {})

    insights = insights_data.get("insights", [])

    st.markdown("#### Advanced analytics report")

    col1, col2 = st.columns(2)

    col1.metric("Insights generated", insights_data.get("total_insights", len(insights)))

    col2.metric("High-priority findings", insights_data.get("high_priority_count", 0))

    if not insights:

        st.info("No notable findings were generated from the available results.")

    else:

        for index, item in enumerate(insights, start=1):

            priority = item.get("priority", "medium").upper()

            title = item.get("title", f"Finding {index}")

            with st.container(border=True):

                st.markdown(f"**{title}**")

                st.caption(

                    f"{item.get('category', 'analysis').replace('_', ' ').title()} "

                    f"· {priority} priority"

                )

                st.write(item.get("description", ""))

                if item.get("recommendation"):

                    st.markdown(

                        f"**Suggested next step:** {item['recommendation']}"

                    )

                evidence = item.get("evidence")

                if evidence:

                    with st.expander("View supporting evidence"):

                        st.json(evidence)

    raw_results = advanced_result.get("results", {})

    if raw_results:

        with st.expander("View detailed analytical results"):

            for section, result in raw_results.items():

                st.markdown(f"**{section.replace('_', ' ').title()}**")

                if isinstance(result, dict):

                    st.json(result)

                else:

                    st.write(result)


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