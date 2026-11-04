import logging
import streamlit as st
import pandas as pd
import numpy as np
import uuid
from ui.sidebar import render_sidebar
from ui.chat_panel import (
    render_chat_panel,
    render_page_chat,
    get_compiled_graph,
    render_prediction_results,
    initialize_chat_state,
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """
    Initialize application session variables.
    """
    defaults = {
        "messages": [],
        "recent_queries": [],
        "dataset": None,
        "dataset_profile": None,
        "dataset_name": None,
       "dataset_signature": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_dataset_conversation():
    """Reset all page conversations when the dataset changes."""

    conversation_keys = {
        "messages": [],
        "recent_queries": [],
        "thread_id": str(uuid.uuid4()),

        "analysis_messages": [],
        "analysis_thread_id": str(uuid.uuid4()),

        "visualization_messages": [],
        "visualization_thread_id": str(uuid.uuid4()),

        "prediction_messages": [],
        "prediction_thread_id": str(uuid.uuid4()),

        "advanced_messages": [],
        "advanced_thread_id": str(uuid.uuid4()),
    }

    for key, value in conversation_keys.items():
        st.session_state[key] = value

    st.session_state.pop("latest_prediction_result", None)
    st.session_state.pop("latest_advanced_result", None)
    st.session_state.pop("latest_advanced_operation", None)

def process_uploaded_dataset(uploaded_file):
    """
    Load and profile a dataset only when the uploaded file changes.
    Handle invalid uploads without exposing internal exception details.
    """
    from tools.data_loader import load_dataset
    from tools.profiling_tools import profile_dataset
    if uploaded_file is None:
        had_dataset = (
            st.session_state.get("dataset_signature") is not None
        )
        st.session_state["dataset"] = None
        st.session_state["dataset_profile"] = None
        st.session_state["dataset_name"] = None
        st.session_state["dataset_signature"] = None
        if had_dataset:
            reset_dataset_conversation()

        return None, None

    import hashlib
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    # Validate the file before processing it.
    if not file_bytes:
        st.error("The uploaded file is empty. Please choose another file.")
        return None, None
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if extension not in {"csv", "xlsx"}:
        st.error(
            "Unsupported file format. Please upload a CSV or XLSX file."
        )
        return None, None

    dataset_signature = hashlib.sha256(file_bytes).hexdigest()
    # Reuse the processed DataFrame if the uploaded file has not changed.
    if (
        st.session_state.get("dataset_signature") == dataset_signature
        and st.session_state.get("dataset") is not None
       and st.session_state.get("dataset_profile") is not None
    ):
        return (
            st.session_state["dataset"],
            st.session_state["dataset_profile"]
        )
    try:
        with st.spinner("Loading and profiling dataset..."):
            df = load_dataset(file_name, file_bytes)
            profile = profile_dataset(df)

        # A different file must start with a clean conversation.
        reset_dataset_conversation()

        st.session_state["dataset"] = df
        st.session_state["dataset_profile"] = profile
        st.session_state["dataset_name"] = file_name
        st.session_state["dataset_signature"] = dataset_signature

        return df, profile
    except ValueError as error:
        # These are controlled, user-friendly validation errors
        # raised by the data loader.
        logger.warning(
            "Dataset validation failed: %s",
            str(error)
        )
        st.session_state["dataset"] = None
        st.session_state["dataset_profile"] = None
        st.session_state["dataset_name"] = None
        st.session_state["dataset_signature"] = None
        reset_dataset_conversation()
        st.error(str(error))
        return None, None
    except Exception as error:
        # Keep internal exception details in application logs only.
        logger.exception(
            "Unexpected dataset processing failure: %s",
            type(error).__name__
        )
        st.session_state["dataset"] = None
        st.session_state["dataset_profile"] = None
        st.session_state["dataset_name"] = None
        st.session_state["dataset_signature"] = None
        reset_dataset_conversation()
        st.error(
            "We couldn't process this dataset because an unexpected "
            "error occurred. Please check the file and try again. "
            "If the problem continues, check the application logs."
        )

        return None, None


def render_dashboard():
    """Render the Aurora dashboard and route the selected page."""

    initialize_session_state()

    selected_page, uploaded_file = render_sidebar()
    df, profile = process_uploaded_dataset(uploaded_file)

    if selected_page == "Dashboard":

        # ------------------------------------------
        # AURORA HEADER
        # ------------------------------------------

        st.markdown(
            """
            <div style="margin-bottom: 18px;">
                <div style="
                    font-size: 2rem;
                    font-weight: 800;
                    letter-spacing: -0.8px;
                    background: linear-gradient(
                        90deg,
                        #06B6D4,
                        #3B82F6,
                        #8B5CF6
                    );
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">
                    AURORA AI DATA ANALYSIS
                </div>
                <div style="
                    color: #64748B;
                    font-size: 14px;
                    margin-top: 3px;
                ">
                    Your intelligent multi-agent data analytics platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------
        # FULL-WIDTH DATASET OVERVIEW
        # ------------------------------------------

        render_dashboard_overview(
            uploaded_file,
            df,
            profile,
        )

        st.markdown("<div style='height: 12px'></div>",
                    unsafe_allow_html=True)

        # ------------------------------------------
        # CHAT + HISTORY
        # ------------------------------------------

        chat_column, history_column = st.columns(
            [3.2, 1],
            gap="large",
        )

        with chat_column:
            with st.container(border=True):
                render_chat_panel()

        with history_column:
            with st.container(border=True):
                from ui.history_panel import render_history_panel
                render_history_panel()

    elif selected_page == "Data Analysis":

        st.title("Data Analysis")
        st.caption(
            "Explore data quality, descriptive statistics, "
            "relationships and column-level insights."
        )
        st.divider()

        render_data_analysis(df, profile)

        if df is not None:
            render_page_chat(
                page_key="analysis",
                title="Ask About Your Data Analysis",
                placeholder=(
                    "Ask about statistics, missing values or columns..."
                ),
                context_instruction=(
                    "Answer questions about descriptive statistics, "
                    "data quality, column information, correlations "
                    "and the analysis displayed on the Data Analysis page."
                ),
            )

    elif selected_page == "Visualizations":

        st.title("Visualizations")
        st.caption(
            "Build interactive charts and explore patterns "
            "in your dataset."
        )
        st.divider()
        from ui.visualization_panel import render_visualization_panel
        render_visualization_panel(df, profile)

        if df is not None:
            render_page_chat(
                page_key="visualization",
                title="Ask About Your Visualizations",
                placeholder=(
                    "Ask about chart types, patterns or visual insights..."
                ),
                context_instruction=(
                    "Answer questions about data visualizations, "
                    "chart selection, axes, aggregation, distributions "
                    "and patterns in the uploaded dataset."
                ),
            )

    elif selected_page == "Predictive Analytics":

        st.title("Predictive Analytics")
        st.caption(
            "Train and compare machine learning models "
            "or generate time-series forecasts."
        )
        st.divider()

        render_predictive_analytics(df)

        if df is not None:
            prediction_result = st.session_state.get(
                "latest_prediction_result"
            )

            prediction_context = None

            if prediction_result:
                prediction_context = str(
                    prediction_result.get("answer", "")
                )

            render_page_chat(
                page_key="prediction",
                title="Ask About Your Prediction Results",
                placeholder=(
                    "Ask about model performance, metrics or forecasts..."
                ),
                context_instruction=(
                    "Answer questions about predictive analytics, "
                    "regression, classification, model comparisons, "
                    "evaluation metrics and Holt-Winters forecasts."
                ),
                result_context=prediction_context,
            )

    elif selected_page == "Advanced Analytics":

        st.title("Advanced Analytics")
        st.caption(
            "Perform statistical testing, correlation analysis, "
            "outlier detection and time-series analysis."
        )
        st.divider()
        from ui.advanced_analytics_panel import render_advanced_analytics
        render_advanced_analytics(df)

        if df is not None:
            advanced_result = st.session_state.get(
                "latest_advanced_result"
            )

            advanced_context = None

            if advanced_result:
                advanced_context = str(
                    advanced_result.get("answer", "")
                )

            render_page_chat(
                page_key="advanced",
                title="Ask About Your Advanced Analysis",
                placeholder=(
                    "Ask about statistical tests, anomalies or insights..."
                ),
                context_instruction=(
                    "Answer questions about advanced analytics, "
                    "outlier detection, correlation, hypothesis testing, "
                    "time-series trends, seasonality and anomalies."
                ),
                result_context=advanced_context,
            )



def render_dashboard_overview(uploaded_file, df, profile):
    """Display a compact, full-width Aurora dataset overview."""
    from tools.profiling_tools import generate_data_quality_summary

    st.subheader("Dataset Overview")

    if df is None or profile is None:
        st.info(
            "Upload a CSV or Excel dataset from the sidebar "
            "to start your analysis."
        )
        return

    dataset_label = (
        uploaded_file.name
        if uploaded_file is not None
        else st.session_state.get("dataset_name", "Dataset")
    )

    # ------------------------------------------
    # DATASET STATUS
    # ------------------------------------------

    status_col, file_col = st.columns([3, 1])

    with status_col:
        st.success(f"Dataset loaded: {dataset_label}")

    with file_col:
        st.caption(
            f"{profile['total_records']:,} records · "
            f"{profile['total_columns']:,} columns"
        )

    # ------------------------------------------
    # PRIMARY KPI CARDS
    # ------------------------------------------

    kpi_cols = st.columns(4)

    kpi_cols[0].metric(
        "Total Records",
        f"{profile['total_records']:,}",
    )

    kpi_cols[1].metric(
        "Total Columns",
        f"{profile['total_columns']:,}",
    )

    kpi_cols[2].metric(
        "Missing Values",
        f"{profile['missing_values']:,}",
    )

    kpi_cols[3].metric(
        "Duplicate Rows",
        f"{profile['duplicate_rows']:,}",
    )

    # ------------------------------------------
    # DATASET DETAILS
    # ------------------------------------------

    info_col, quality_col, composition_col = st.columns(
        [1.1, 1.2, 1.2],
        gap="medium",
    )

    with info_col:
        with st.container(border=True):
            st.markdown("**Dataset Information**")

            st.write(f"**File:** {dataset_label}")

            st.write(
                f"**Dimensions:** "
                f"{profile['total_records']:,} × "
                f"{profile['total_columns']:,}"
            )

            st.write(
                f"**Memory:** "
                f"{profile['memory_usage_mb']} MB"
            )

    with quality_col:
        with st.container(border=True):
            st.markdown("**Data Quality**")

            st.write(
                f"**Missing percentage:** "
                f"{profile['missing_percentage']}%"
            )

            st.write(
                f"**Constant columns:** "
                f"{len(profile.get('constant_columns', []))}"
            )

            st.write(
                f"**Potential identifiers:** "
                f"{len(profile.get('potential_identifier_columns', []))}"
            )

            for observation in generate_data_quality_summary(profile):
                st.caption(f"• {observation}")

    with composition_col:
        with st.container(border=True):
            st.markdown("**Column Composition**")

            column_types = profile["column_types"]

            composition_items = [
                ("Numerical", "numerical"),
                ("Categorical", "categorical"),
                ("Date", "date"),
                ("Text", "text"),
            ]

            for label, key in composition_items:
                st.write(
                    f"{label}: "
                    f"**{len(column_types.get(key, []))}**"
                )

    # ------------------------------------------
    # OPTIONAL NUMERICAL KPI EXPLORER
    # ------------------------------------------

    numeric_columns = profile["column_types"]["numerical"]

    if numeric_columns:
        with st.expander(
            "Explore Numerical KPIs",
            expanded=False,
        ):
            kpi_column = st.selectbox(
                "Choose a numerical column",
                options=numeric_columns,
                key="dashboard_numeric_kpi_column",
            )

            values = pd.to_numeric(
                df[kpi_column],
                errors="coerce",
            ).dropna()

            if not values.empty:
                metric_cols = st.columns(4)

                metric_cols[0].metric(
                    "Mean",
                    f"{values.mean():,.3f}",
                )

                metric_cols[1].metric(
                    "Median",
                    f"{values.median():,.3f}",
                )

                metric_cols[2].metric(
                    "Minimum",
                    f"{values.min():,.3f}",
                )

                metric_cols[3].metric(
                    "Maximum",
                    f"{values.max():,.3f}",
                )
            else:
                st.info(
                    "This column has no valid numerical values."
                )

    # ------------------------------------------
    # OPTIONAL DATA PREVIEW
    # ------------------------------------------

    with st.expander(
        "Preview Dataset",
        expanded=False,
    ):
        preview_rows = st.slider(
            "Number of rows to display",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="dashboard_preview_rows",
        )

        st.dataframe(
            df.head(preview_rows),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            f"Showing {min(preview_rows, len(df)):,} "
            f"of {len(df):,} records."
        )

def render_data_analysis(df, profile):
    """Display detailed dataset quality, statistics and column exploration."""

    from tools.statistics_tools import (
        generate_descriptive_statistics,
        generate_categorical_summary,
        generate_correlation_matrix,
        generate_column_summary,
    )

    st.header("Data Analysis")

    if df is None or profile is None:
        st.warning("Please upload a valid dataset from the sidebar.")
        return

    st.success("Dataset successfully loaded and profiled.")

    st.subheader("Column Information")

    st.dataframe(
        pd.DataFrame(profile["column_details"]),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Data Quality Analysis")

    quality_left, quality_right = st.columns(2)

    with quality_left:

        st.markdown("**Missing values by column**")

        missing_frame = pd.DataFrame({
            "Column": list(profile.get("missing_by_column", {}).keys()),
            "Missing Values": list(profile.get("missing_by_column", {}).values()),
        })

        missing_frame = missing_frame[
            missing_frame["Missing Values"] > 0
        ]

        if not missing_frame.empty:

            st.dataframe(
                missing_frame.sort_values(
                    "Missing Values",
                    ascending=False
                ),
                width="stretch",
                hide_index=True,
            )

            st.bar_chart(
                missing_frame.set_index("Column")
            )

        else:
            st.success("No missing values found in any column.")

    with quality_right:

        st.markdown("**Columns requiring review**")

        review_rows = []

        for label, columns in (
            ("Constant", profile.get("constant_columns", [])),
            ("High cardinality", profile.get("high_cardinality_columns", [])),
            ("Potential identifier", profile.get("potential_identifier_columns", [])),
        ):

            review_rows.extend(
                {"Issue": label, "Column": column}
                for column in columns
            )

        if review_rows:

            st.dataframe(
                pd.DataFrame(review_rows),
                width="stretch",
                hide_index=True
            )

        else:
            st.success(
                "No constant, high-cardinality or potential identifier columns flagged."
            )

    st.divider()
    st.subheader("Numerical Descriptive Statistics")

    numerical_stats = generate_descriptive_statistics(df)

    if not numerical_stats.empty:

        st.dataframe(
            numerical_stats,
            width="stretch"
        )

    else:
        st.info("No numerical columns were detected.")

    st.divider()
    st.subheader("Categorical Summary")

    categorical_stats = generate_categorical_summary(df)

    if not categorical_stats.empty:

        st.dataframe(
            categorical_stats,
            width="stretch",
            hide_index=True
        )

    else:
        st.info("No categorical columns were detected.")

    st.divider()
    st.subheader("Correlation Analysis")

    correlation_matrix = generate_correlation_matrix(df)

    if not correlation_matrix.empty:

        st.dataframe(
            correlation_matrix,
            width="stretch"
        )

    else:
        st.info(
            "At least two numerical columns are required to calculate correlations."
        )

    # ==================================================
    # INTERACTIVE COLUMN EXPLORER
    # ==================================================

    st.divider()
    st.subheader("Interactive Column Explorer")

    selected_column = st.selectbox(
        "Select a column to inspect",
        options=df.columns.tolist(),
        key="profile_selected_column",
    )

    summary = generate_column_summary(
        df,
        selected_column
    )

    # Basic column information
    summary_cols = st.columns(4)

    summary_cols[0].metric(
        "Non-null",
        f'{summary["non_null"]:,}'
    )

    summary_cols[1].metric(
        "Missing",
        f'{summary["missing"]:,}'
    )

    summary_cols[2].metric(
        "Unique",
        f'{summary["unique"]:,}'
    )

    summary_cols[3].metric(
        "Data type",
        summary["dtype"]
    )

    selected_series = df[selected_column]

    # ==================================================
    # NUMERICAL COLUMN
    # ==================================================

    if pd.api.types.is_numeric_dtype(selected_series):

        numeric_values = pd.to_numeric(
            selected_series,
            errors="coerce"
        ).dropna()

        if not numeric_values.empty:

            st.markdown("**Distribution**")

            st.bar_chart(
                numeric_values.value_counts().sort_index().head(100)
            )

            # Descriptive statistics
            statistics = summary.get("statistics", {})

            if statistics:

                st.markdown("**Statistical Summary**")

                statistic_labels = [
                    ("Mean", "mean"),
                    ("Median", "median"),
                    ("Standard Deviation", "std"),
                    ("Minimum", "min"),
                    ("Maximum", "max"),
                ]

                available_statistics = [
                    (label, key)
                    for label, key in statistic_labels
                    if statistics.get(key) is not None
                ]

                # Display statistics in a clean metric layout.
                for start in range(0, len(available_statistics), 3):

                    metric_group = available_statistics[
                        start:start + 3
                    ]

                    metric_columns = st.columns(
                        len(metric_group)
                    )

                    for column, (label, key) in zip(
                        metric_columns,
                        metric_group
                    ):

                        value = statistics[key]

                        if isinstance(value, (int, float, np.number)):
                            formatted_value = f"{value:,.2f}"
                        else:
                            formatted_value = str(value)

                        column.metric(
                            label,
                            formatted_value
                        )

        else:
            st.info(
                "No valid numerical values are available to plot."
            )

    # ==================================================
    # CATEGORICAL COLUMN
    # ==================================================

    else:

        top_values = (
            selected_series.dropna()
            .astype(str)
            .value_counts()
            .head(15)
        )

        if not top_values.empty:

            st.markdown("**Most frequent values (top 15)**")

            st.bar_chart(top_values)

        else:

            st.info(
                "This column contains no non-missing values."
            )



def render_predictive_analytics(df):
    """Render the redesigned Aurora predictive analytics workspace."""

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="margin-bottom:14px;">
            <div style="
                font-size:24px;
                font-weight:800;
                background:linear-gradient(
                    90deg,#06B6D4,#3B82F6,#8B5CF6
                );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                Predictive Analytics Studio
            </div>
            <div style="
                color:#64748B;
                font-size:13px;
                margin-top:4px;
            ">
                Build, compare and evaluate machine learning models
                and time-series forecasts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------------

    if df is None or df.empty:
        st.info(
            "Upload a CSV or Excel dataset from the sidebar "
            "to start predictive analytics."
        )
        return

    initialize_chat_state()

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    all_columns = df.columns.tolist()

    if not numeric_columns:
        st.warning(
            "This dataset does not contain numerical columns. "
            "At least one numerical column is required for "
            "regression or forecasting."
        )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("**Dataset overview**")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Records", f"{len(df):,}")
        col2.metric("Total Columns", len(all_columns))
        col3.metric("Numerical Columns", len(numeric_columns))
        col4.metric(
            "Categorical Columns",
            len(df.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns),
        )

        st.caption(
            f"Active dataset: "
            f"{st.session_state.get('dataset_name', 'Uploaded dataset')}"
        )

    # --------------------------------------------------------
    # TASK SELECTION
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            color:#172033;
            font-size:18px;
            font-weight:750;
            margin:18px 0 8px;
        ">
            1. Choose prediction task
        </div>
        <div style="
            color:#64748B;
            font-size:12px;
            margin-bottom:10px;
        ">
            Select the type of prediction you want to perform.
        </div>
        """,
        unsafe_allow_html=True,
    )

    task_options = [
        "Regression",
        "Classification",
        "Time-series Forecasting",
    ]

    task_type = st.radio(
        "Prediction task",
        options=task_options,
        horizontal=True,
        key="prediction_task_type",
        label_visibility="collapsed",
    )

    task_descriptions = {
        "Regression": (
            "Predict a continuous numerical value, such as sales, "
            "revenue, salary or price."
        ),
        "Classification": (
            "Predict a category or class, such as approved/rejected "
            "or customer churn."
        ),
        "Time-series Forecasting": (
            "Predict future numerical values using historical "
            "time-based observations with Holt-Winters Exponential Smoothing."
        ),
    }

    with st.container(border=True):
        st.markdown(f"**{task_type}**")
        st.caption(task_descriptions[task_type])

    # --------------------------------------------------------
    # TARGET CONFIGURATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            color:#172033;
            font-size:18px;
            font-weight:750;
            margin:18px 0 8px;
        ">
            2. Configure target
        </div>
        <div style="
            color:#64748B;
            font-size:12px;
            margin-bottom:10px;
        ">
            Choose the column that the model should predict.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if task_type in ["Regression", "Time-series Forecasting"]:
        target_options = numeric_columns
    else:
        target_options = all_columns

    if not target_options:
        st.warning(
            "No suitable target columns are available for "
            "the selected prediction task."
        )
        return

    target_column = st.selectbox(
        "Target column",
        options=target_options,
        key=f"prediction_target_{task_type}",
        help="The target column contains the values to be predicted.",
    )

    date_column = None
    prediction_length = 6
    frequency = None

    selected_models = []

    # --------------------------------------------------------
    # FORECASTING CONFIGURATION
    # --------------------------------------------------------

    if task_type == "Time-series Forecasting":

        st.markdown(
            """
            <div style="
                color:#172033;
                font-size:18px;
                font-weight:750;
                margin:18px 0 8px;
            ">
                3. Configure forecast
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):

            st.markdown("**Holt-Winters forecast settings**")

            col1, col2 = st.columns(2)

            with col1:
                date_column = st.selectbox(
                    "Date / time column",
                    options=all_columns,
                    key="prediction_date_column_v2",
                    help="Select the column containing historical timestamps.",
                )

            with col2:
                prediction_length = st.number_input(
                    "Future periods",
                    min_value=1,
                    max_value=100,
                    value=6,
                    step=1,
                    key="prediction_length",
                )

            frequency_option = st.selectbox(
                "Time frequency",
                options=[
                    "Auto",
                    "Daily (D)",
                    "Weekly (W)",
                    "Monthly Start (MS)",
                    "Monthly End (ME)",
                    "Quarterly (Q)",
                    "Yearly (Y)",
                ],
                key="prediction_frequency",
            )

            frequency_mapping = {
                "Auto": None,
                "Daily (D)": "D",
                "Weekly (W)": "W",
                "Monthly Start (MS)": "MS",
                "Monthly End (ME)": "ME",
                "Quarterly (Q)": "Q",
                "Yearly (Y)": "Y",
            }

            frequency = frequency_mapping[frequency_option]

            st.caption(
                "Holt-Winters uses historical observations to model "
    "level, trend and (when sufficient history is available) "
    "seasonality. Automatic frequency detection is supported."
            )

        if target_column == date_column:
            st.error(
                "The target column and date column must be different."
            )
            return

    # --------------------------------------------------------
    # TABULAR MODEL CONFIGURATION
    # --------------------------------------------------------

    if task_type in ["Regression", "Classification"]:

        st.markdown(
            """
            <div style="
                color:#172033;
                font-size:18px;
                font-weight:750;
                margin:18px 0 8px;
            ">
                3. Configure machine learning models
            </div>
            """,
            unsafe_allow_html=True,
        )

        if task_type == "Regression":
            available_models = [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "KNN",
            ]
        else:
            available_models = [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "KNN",
            ]

        with st.container(border=True):

            st.markdown("**Scikit-learn model selection**")

            selected_models = st.multiselect(
                "Choose models to train and compare",
                options=available_models,
                default=available_models,
                key=f"prediction_models_{task_type}",
                help=(
                    "All selected models use the same train/test "
                    "evaluation workflow."
                ),
            )

            st.divider()

            st.markdown("**Additional prediction engine**")


    # --------------------------------------------------------
    # EXECUTION CONFIGURATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            color:#172033;
            font-size:18px;
            font-weight:750;
            margin:20px 0 8px;
        ">
            4. Execute predictive analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        st.markdown("**Ready to run?**")

        st.caption(
            "Review your configuration before starting model "
            "training or forecasting."
        )

        summary_col1, summary_col2 = st.columns(2)

        summary_col1.write(f"**Task:** {task_type}")
        summary_col1.write(f"**Target:** {target_column}")

        if task_type == "Time-series Forecasting":
            summary_col2.write("**Engine:** Holt-Winters")
            summary_col2.write(
                f"**Forecast horizon:** {prediction_length} periods"
            )
        else:
            summary_col2.write(
                f"**Scikit-learn models:** {len(selected_models)}"
            )

        run_prediction = st.button(
            "Run Predictive Analysis",
            type="primary",
            width="stretch",
            key="run_prediction_v2",
            disabled=(
                task_type in ["Regression", "Classification"]
                and not selected_models
            ),
        )

    # --------------------------------------------------------
    # EXECUTE LANGGRAPH PREDICTION WORKFLOW
    # --------------------------------------------------------

    if run_prediction:

        if (
            task_type == "Time-series Forecasting"
            and target_column == date_column
        ):
            st.error(
                "Please select different target and date columns."
            )
            return

        if task_type == "Regression":
            question = (
                f"Perform regression using the dataset. "
                f"The target column is '{target_column}'. "
                f"Compare only these scikit-learn models: "
                f"{', '.join(selected_models)}. "
            )

        elif task_type == "Classification":
            question = (
                f"Perform classification using the dataset. "
                f"The target column is '{target_column}'. "
                f"Compare only these scikit-learn models: "
                f"{', '.join(selected_models)}. "
            )

        else:
            question = (
                f"Perform time-series forecasting using Holt-Winters. "
                f"The target column is '{target_column}'. "
                f"The date column is '{date_column}'. "
                f"Generate {prediction_length} future periods. "
            )

            if frequency:
                question += f"Use frequency '{frequency}'. "
            else:
                question += "Automatically infer the time frequency. "

        configured_task = {
            "Regression": "regression",
            "Classification": "classification",
            "Time-series Forecasting": "forecasting",
        }[task_type]

        prediction_config = {
                "task_type": configured_task,
                "target_column": target_column,
                "date_column": date_column,
                "selected_models": selected_models,
                "prediction_length": int(prediction_length),
                "frequency": frequency,
            }

        try:
            graph = get_compiled_graph()

            with st.spinner(
                "Aurora is training models and evaluating predictions..."
            ):
                result = graph.invoke(
                    {
                        "user_question": question,
                        "route": None,
                        "analysis_result": None,
                        "advanced_result": None,
                        "advanced_config": None,
                        "prediction_result": None,
                        "prediction_config": prediction_config,
                        "answer": "",
                        "error": "",
                        "error_code": "",
                        "error_message": "",
                        "chart_json": None,
                        "chart_plan": None,
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.setdefault(
                                "prediction_thread_id",
                                str(uuid.uuid4()),
                            )
                        }
                    },
                    context={"dataframe": df},
                )

            prediction_result = result.get("prediction_result")

            if not prediction_result:
                st.error(
                    result.get(
                        "answer",
                        "The Prediction Agent did not return results.",
                    )
                )
                return

            st.session_state["latest_prediction_result"] = (
                prediction_result
            )

            if "prediction_messages" not in st.session_state:
                st.session_state["prediction_messages"] = []

            st.session_state["prediction_messages"].append({
                "role": "assistant",
                "content": result.get(
                    "answer",
                    "Predictive analysis completed.",
                ),
            })

            from datetime import datetime

            if "recent_queries" not in st.session_state:
                st.session_state["recent_queries"] = []

            st.session_state["recent_queries"].append({
                "query": question,
                "timestamp": datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                ),
            })

            if prediction_result.get("success"):
                st.success(
                    "Predictive analysis completed successfully."
                )
            else:
                st.error(
                    prediction_result.get(
                        "answer",
                        "Prediction failed.",
                    )
                )

        except Exception as error:
            logger.exception(
                "Prediction workflow failed: %s",
                type(error).__name__,
            )

            st.error(
                "The prediction workflow encountered an unexpected "
                "issue. Please review your configuration and try again. "
                "If the problem continues, check the application logs."
            )

    # --------------------------------------------------------
    # DISPLAY SAVED PREDICTION RESULTS
    # --------------------------------------------------------

    prediction_result = st.session_state.get(
        "latest_prediction_result"
    )

    if prediction_result:
        render_prediction_results(
            prediction_result,
            key_prefix="prediction_latest",
        )

