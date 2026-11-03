import uuid

import streamlit as st

from ui.chat_panel import get_compiled_graph, initialize_chat_state, render_advanced_results


FREQUENCY_OPTIONS = {
    "Automatic": None,
    "Daily (D)": "D",
    "Weekly (W)": "W",
    "Monthly start (MS)": "MS",
    "Monthly end (ME)": "ME",
    "Quarterly (Q)": "Q",
    "Yearly (YS)": "YS",
}


def _numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def _date_columns(df):
    detected = [
        column for column in df.columns
        if str(df[column].dtype).startswith("datetime")
    ]
    # Include object/string columns because date values are often imported as text.
    for column in df.columns:
        if column not in detected and any(
            token in str(column).lower()
            for token in ("date", "time", "timestamp")
        ):
            detected.append(column)
    return detected or df.columns.tolist()


def render_advanced_analytics(df):
    """Dedicated UI for the existing LangGraph advanced analytics agent."""
    st.header("Advanced Analytics")
    st.caption(
        "Run statistical tests, detect unusual observations, study time-series "
        "patterns and generate evidence-backed insights."
    )

    if df is None or df.empty:
        st.warning("Upload a valid dataset from the sidebar to use advanced analytics.")
        return

    initialize_chat_state()
    numeric_columns = _numeric_columns(df)
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    st.success(
        f"Dataset ready: {st.session_state.get('dataset_name', 'Uploaded dataset')} "
        f"· {len(df):,} rows · {len(df.columns)} columns"
    )

    operation_label = st.selectbox(
        "Choose an analysis",
        [
            "Outlier Detection",
            "Correlation Analysis",
            "Hypothesis Testing",
            "Trend and Seasonality",
            "Time-series Anomaly Detection",
            "Comprehensive Analysis",
        ],
        key="advanced_operation",
    )

    operation_map = {
        "Outlier Detection": "outliers",
        "Correlation Analysis": "correlation",
        "Hypothesis Testing": "hypothesis",
        "Trend and Seasonality": "trend",
        "Time-series Anomaly Detection": "anomaly",
        "Comprehensive Analysis": "comprehensive",
    }
    operation = operation_map[operation_label]
    config = {"operation": operation}

    if operation in ("outliers", "correlation", "comprehensive"):
        if not numeric_columns:
            st.warning("This analysis requires at least one numerical column.")
            return
        selected_columns = st.multiselect(
            "Numerical columns to analyse",
            options=numeric_columns,
            default=numeric_columns,
            key=f"advanced_{operation}_columns",
        )
        config["columns"] = selected_columns

    if operation == "hypothesis":
        test_label = st.selectbox(
            "Statistical test",
            [
                "Independent t-test (two groups)",
                "Paired t-test (two measurements)",
                "One-way ANOVA (three or more groups)",
                "Chi-square test (two categorical columns)",
                "Normality test (one numerical column)",
            ],
            key="advanced_hypothesis_test",
        )
        test_map = {
            "Independent t-test (two groups)": "independent_t",
            "Paired t-test (two measurements)": "paired_t",
            "One-way ANOVA (three or more groups)": "anova",
            "Chi-square test (two categorical columns)": "chi_square",
            "Normality test (one numerical column)": "normality",
        }
        config["test_type"] = test_map[test_label]

        if config["test_type"] in ("independent_t", "anova"):
            if not numeric_columns or not categorical_columns:
                st.warning("This test requires numerical and categorical columns.")
                return
            config["numeric_column"] = st.selectbox(
                "Numerical measure", numeric_columns, key="advanced_test_numeric"
            )
            config["group_column"] = st.selectbox(
                "Grouping column", categorical_columns, key="advanced_test_group"
            )
            if config["test_type"] == "independent_t":
                group_values = df[config["group_column"]].dropna().unique().tolist()
                if len(group_values) < 2:
                    st.warning("The selected grouping column must contain at least two groups.")
                    return
                group_options = [str(value) for value in group_values]
                group_a = st.selectbox(
                    "First group", group_options, key="advanced_group_a"
                )
                group_b_options = [value for value in group_options if value != group_a]
                group_b = st.selectbox(
                    "Second group", group_b_options, key="advanced_group_b"
                )
                config["group_a"] = group_a
                config["group_b"] = group_b

        elif config["test_type"] == "paired_t":
            if len(numeric_columns) < 2:
                st.warning("A paired t-test requires two numerical columns.")
                return
            config["column_a"] = st.selectbox(
                "First measurement", numeric_columns, key="advanced_paired_a"
            )
            second_options = [
                column for column in numeric_columns
                if column != config["column_a"]
            ]
            config["column_b"] = st.selectbox(
                "Second measurement", second_options, key="advanced_paired_b"
            )

        elif config["test_type"] == "chi_square":
            if len(categorical_columns) < 2:
                st.warning("A chi-square test requires two categorical columns.")
                return
            config["column_a"] = st.selectbox(
                "First categorical column", categorical_columns,
                key="advanced_chi_a",
            )
            second_options = [
                column for column in categorical_columns
                if column != config["column_a"]
            ]
            config["column_b"] = st.selectbox(
                "Second categorical column", second_options,
                key="advanced_chi_b",
            )

        else:
            if not numeric_columns:
                st.warning("Normality testing requires a numerical column.")
                return
            config["numeric_column"] = st.selectbox(
                "Numerical column", numeric_columns,
                key="advanced_normality_column",
            )

    if operation in ("trend", "anomaly", "comprehensive"):
        date_options = _date_columns(df)
        default_date = date_options[0]
        config["date_column"] = st.selectbox(
            "Date/time column", date_options, index=0,
            key=f"advanced_{operation}_date",
        )
        if not numeric_columns:
            st.warning("Time-series analysis requires a numerical target column.")
            return
        config["target_column"] = st.selectbox(
            "Numerical target", numeric_columns,
            key=f"advanced_{operation}_target",
        )
        frequency_label = st.selectbox(
            "Time frequency", list(FREQUENCY_OPTIONS.keys()),
            key=f"advanced_{operation}_frequency",
        )
        config["frequency"] = FREQUENCY_OPTIONS[frequency_label]
        config["aggregation"] = st.selectbox(
            "Aggregation when multiple records share a period",
            ["mean", "sum", "median", "min", "max"],
            key=f"advanced_{operation}_aggregation",
        )
        if operation in ("trend", "comprehensive"):
            config["seasonal_period"] = st.number_input(
                "Seasonal period (0 = skip seasonality)",
                min_value=0, max_value=365, value=12, step=1,
                key=f"advanced_{operation}_seasonal_period",
            )
            if config["seasonal_period"] == 0:
                config["seasonal_period"] = None
        if operation in ("anomaly", "comprehensive"):
            config["anomaly_method"] = st.selectbox(
                "Anomaly detection method",
                ["rolling_mad", "rolling_zscore", "seasonal_residual"],
                key=f"advanced_{operation}_anomaly_method",
            )
            config["window"] = st.number_input(
                "Rolling window", min_value=3, max_value=100,
                value=6, step=1, key=f"advanced_{operation}_window",
            )
            config["threshold"] = st.number_input(
                "Anomaly threshold", min_value=1.0, max_value=10.0,
                value=3.5, step=0.5, key=f"advanced_{operation}_threshold",
            )

    st.divider()
    run_analysis = st.button(
        "Run Advanced Analysis", type="primary", width="stretch",
        key="run_advanced_analytics",
    )

    if run_analysis:
        if operation in ("outliers", "correlation") and not config.get("columns"):
            st.warning("Select at least one numerical column.")
            return
        question = (
            f"Run {operation_label.lower()} on the uploaded dataset using "
            "the exact configuration supplied in the advanced analytics dashboard. "
            "Explain the results and report limitations."
        )
        try:
            graph = get_compiled_graph()
            with st.spinner("Advanced Analytics Agent is running the analysis..."):
                thread_id = st.session_state.setdefault(
                    "advanced_analytics_thread_id", str(uuid.uuid4())
                )
                result = graph.invoke(
                    {
                        "user_question": question,
                        "advanced_config": config,
                    },
                    config={"configurable": {"thread_id": thread_id}},
                    context={"dataframe": df},
                )
            advanced_result = result.get("advanced_result")
            if advanced_result and advanced_result.get("success"):
                st.session_state["latest_advanced_result"] = advanced_result
                st.session_state["latest_advanced_operation"] = operation_label
            else:
                st.error(
                    result.get("error")
                    or (advanced_result or {}).get("error")
                    or "The advanced analysis did not return a successful result."
                )
        except Exception as error:
            st.error(f"Advanced analysis failed: {error}")

    latest_result = st.session_state.get("latest_advanced_result")
    if latest_result:
        st.divider()
        st.subheader(
            f"Latest Results · {st.session_state.get('latest_advanced_operation', 'Advanced Analytics')}"
        )
        render_advanced_results(latest_result)
