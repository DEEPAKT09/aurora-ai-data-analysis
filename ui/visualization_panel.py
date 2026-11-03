
import logging

import pandas as pd
import streamlit as st

from tools.visualization_tools import (
    prepare_visualization_data,
    create_histogram,
    create_bar_chart,
    create_scatter_plot,
    create_line_chart,
    create_box_plot,
    create_correlation_heatmap,
)

logger = logging.getLogger(__name__)


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

CHART_DESCRIPTIONS = {
    "Histogram": (
        "Understand the distribution, spread and frequency "
        "of numerical values."
    ),
    "Bar Chart": (
        "Compare categories using record counts or aggregated "
        "numerical values."
    ),
    "Scatter Plot": (
        "Explore relationships and patterns between two "
        "numerical variables."
    ),
    "Line Chart": (
        "Observe changes and trends across dates or categories."
    ),
    "Box Plot": (
        "Compare distributions, quartiles and potential outliers."
    ),
    "Correlation Heatmap": (
        "Identify the strength and direction of relationships "
        "between numerical variables."
    ),
}


CHART_OPTIONS = [
    "Histogram",
    "Bar Chart",
    "Scatter Plot",
    "Line Chart",
    "Box Plot",
    "Correlation Heatmap",
]


# ============================================================
# UI HELPERS
# ============================================================

def _section_heading(number, title, description):
    """Render a consistent visualization section heading."""

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:10px;
            margin-top:8px;
            margin-bottom:4px;
        ">
            <div style="
                min-width:29px;
                height:29px;
                border-radius:9px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:linear-gradient(
                    135deg,#06B6D4,#3B82F6
                );
                color:white;
                font-size:13px;
                font-weight:800;
            ">
                {number}
            </div>
            <div style="
                color:#172033;
                font-size:17px;
                font-weight:750;
            ">
                {title}
            </div>
        </div>
        <div style="
            color:#64748B;
            font-size:12px;
            margin-left:39px;
            margin-bottom:12px;
        ">
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_chart_summary(chart_type, row_count, column_count):
    """Show compact chart information."""

    col1, col2, col3 = st.columns(3)

    col1.metric("Visualization", chart_type)
    col2.metric("Available Records", f"{row_count:,}")
    col3.metric("Available Columns", f"{column_count:,}")


def _show_empty_state(message):
    """Display a consistent empty state."""

    st.info(message)


# ============================================================
# VISUALIZATION PANEL
# ============================================================

def render_visualization_panel(
    df: pd.DataFrame,
    profile: dict,
):
    """Render the redesigned interactive Aurora visualization page."""

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="margin-bottom:5px;">
            <div style="
                font-size:23px;
                font-weight:800;
                background:linear-gradient(
                    90deg,#06B6D4,#3B82F6,#8B5CF6
                );
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
            ">
                Visualization Studio
            </div>
            <div style="
                color:#64748B;
                font-size:13px;
                margin-top:3px;
            ">
                Transform your dataset into interactive visual insights.
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
            "to start creating visualizations."
        )
        return

    if not isinstance(profile, dict):
        st.warning(
            "Dataset profiling information is unavailable. "
            "Please upload the dataset again."
        )
        return

    column_types = profile.get("column_types", {})

    if not isinstance(column_types, dict):
        st.warning(
            "Dataset column information is incomplete. "
            "Please upload the dataset again."
        )
        return

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    try:
        date_columns = column_types.get("date", [])

        plot_df = prepare_visualization_data(
            df,
            date_columns,
        )

        if plot_df is None or plot_df.empty:
            _show_empty_state(
                "There is no usable data available for visualization."
            )
            return

    except Exception as error:
        logger.exception(
            "Visualization data preparation failed: %s",
            type(error).__name__,
        )

        st.error(
            "The dataset could not be prepared for visualization. "
            "Please check the uploaded data and try again."
        )
        return

    all_columns = plot_df.columns.tolist()

    numerical_columns = (
        plot_df.select_dtypes(include="number")
        .columns.tolist()
    )

    categorical_columns = (
        plot_df.select_dtypes(
            include=["object", "category", "string", "bool"]
        )
        .columns.tolist()
    )

    date_columns = [
        column
        for column in date_columns
        if column in all_columns
    ]

    if not all_columns:
        _show_empty_state(
            "No usable columns were found for visualization."
        )
        return

    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    with st.container(border=True):
        st.markdown("**Dataset available for visualization**")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Records", f"{len(plot_df):,}")
        col2.metric("Total Columns", len(all_columns))
        col3.metric("Numerical", len(numerical_columns))
        col4.metric("Categorical", len(categorical_columns))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # CHART SELECTION
    # --------------------------------------------------------

    _section_heading(
        "1",
        "Choose Visualization",
        "Select the chart that best represents the pattern you want to explore.",
    )

    with st.container(border=True):

        chart_type = st.selectbox(
            "Visualization type",
            options=CHART_OPTIONS,
            key="visualization_chart_type",
        )

        st.caption(
            CHART_DESCRIPTIONS.get(chart_type, "")
        )

    # --------------------------------------------------------
    # CHART CONFIGURATION
    # --------------------------------------------------------

    _section_heading(
        "2",
        "Configure Your Chart",
        "Select the columns and aggregation settings for your visualization.",
    )

    fig = None
    configuration_valid = True

    with st.container(border=True):

        # ====================================================
        # HISTOGRAM
        # ====================================================

        if chart_type == "Histogram":

            if not numerical_columns:
                _show_empty_state(
                    "A histogram requires at least one numerical column."
                )
                return

            selected_column = st.selectbox(
                "Numerical column",
                numerical_columns,
                key="histogram_column",
            )

            valid_values = pd.to_numeric(
                plot_df[selected_column],
                errors="coerce",
            ).dropna()

            if valid_values.empty:
                _show_empty_state(
                    "The selected column has no valid numerical values."
                )
                return

            fig = create_histogram(
                plot_df,
                selected_column,
            )

        # ====================================================
        # BAR CHART
        # ====================================================

        elif chart_type == "Bar Chart":

            left, right = st.columns(2)

            with left:
                x_column = st.selectbox(
                    "Category / X-axis",
                    all_columns,
                    key="bar_x_column",
                )

            with right:
                y_options = ["Record Count"] + numerical_columns

                y_selection = st.selectbox(
                    "Value / Y-axis",
                    y_options,
                    key="bar_y_column",
                )

            y_column = (
                None
                if y_selection == "Record Count"
                else y_selection
            )

            aggregation = "count"

            if y_column:
                aggregation = st.selectbox(
                    "Aggregation method",
                    ["sum", "mean", "count", "max", "min"],
                    key="bar_aggregation",
                )

            if plot_df[x_column].dropna().empty:
                _show_empty_state(
                    "The selected category column contains no usable values."
                )
                return

            if y_column and pd.to_numeric(
                plot_df[y_column],
                errors="coerce",
            ).dropna().empty:
                _show_empty_state(
                    "The selected numerical column has no valid values."
                )
                return

            fig = create_bar_chart(
                plot_df,
                x_column,
                y_column,
                aggregation,
            )

        # ====================================================
        # SCATTER PLOT
        # ====================================================

        elif chart_type == "Scatter Plot":

            if len(numerical_columns) < 2:
                _show_empty_state(
                    "A scatter plot requires at least two numerical columns."
                )
                return

            left, right = st.columns(2)

            with left:
                x_column = st.selectbox(
                    "X-axis",
                    numerical_columns,
                    key="scatter_x_column",
                )

            available_y = [
                column
                for column in numerical_columns
                if column != x_column
            ]

            with right:
                y_column = st.selectbox(
                    "Y-axis",
                    available_y,
                    key="scatter_y_column",
                )

            color_options = ["None"] + categorical_columns

            color_selection = st.selectbox(
                "Color by category (optional)",
                color_options,
                key="scatter_color_column",
            )

            color_column = (
                None
                if color_selection == "None"
                else color_selection
            )

            valid_data = plot_df[
                [x_column, y_column]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()

            if valid_data.empty:
                _show_empty_state(
                    "The selected columns do not contain enough "
                    "valid numerical pairs."
                )
                return

            fig = create_scatter_plot(
                plot_df,
                x_column,
                y_column,
                color_column,
            )

        # ====================================================
        # LINE CHART
        # ====================================================

        elif chart_type == "Line Chart":

            if not numerical_columns:
                _show_empty_state(
                    "A line chart requires at least one numerical column."
                )
                return

            x_options = list(dict.fromkeys(
                date_columns + categorical_columns
            ))

            if not x_options:
                _show_empty_state(
                    "A line chart requires a date or categorical X-axis."
                )
                return

            left, right = st.columns(2)

            with left:
                x_column = st.selectbox(
                    "X-axis",
                    x_options,
                    key="line_x_column",
                )

            with right:
                y_column = st.selectbox(
                    "Y-axis",
                    numerical_columns,
                    key="line_y_column",
                )

            aggregation = st.selectbox(
                "Aggregation method",
                ["sum", "mean", "count", "max", "min"],
                key="line_aggregation",
            )

            valid_data = plot_df[
                [x_column, y_column]
            ].dropna()

            if valid_data.empty:
                _show_empty_state(
                    "The selected columns do not contain enough "
                    "valid records for a line chart."
                )
                return

            fig = create_line_chart(
                plot_df,
                x_column,
                y_column,
                aggregation,
            )

        # ====================================================
        # BOX PLOT
        # ====================================================

        elif chart_type == "Box Plot":

            if not numerical_columns:
                _show_empty_state(
                    "A box plot requires at least one numerical column."
                )
                return

            y_column = st.selectbox(
                "Numerical column",
                numerical_columns,
                key="box_y_column",
            )

            x_options = ["None"] + categorical_columns

            x_selection = st.selectbox(
                "Group by category (optional)",
                x_options,
                key="box_x_column",
            )

            x_column = (
                None
                if x_selection == "None"
                else x_selection
            )

            if pd.to_numeric(
                plot_df[y_column],
                errors="coerce",
            ).dropna().empty:
                _show_empty_state(
                    "The selected column has no valid numerical values."
                )
                return

            fig = create_box_plot(
                plot_df,
                y_column,
                x_column,
            )

        # ====================================================
        # CORRELATION HEATMAP
        # ====================================================

        elif chart_type == "Correlation Heatmap":

            if len(numerical_columns) < 2:
                _show_empty_state(
                    "A correlation heatmap requires at least two "
                    "numerical columns."
                )
                return

            usable_numeric = plot_df[
                numerical_columns
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            usable_numeric = usable_numeric.loc[
                :,
                usable_numeric.notna().any(),
            ]

            if usable_numeric.shape[1] < 2:
                _show_empty_state(
                    "At least two numerical columns with valid "
                    "values are required."
                )
                return

            fig = create_correlation_heatmap(
                plot_df,
            )

    # --------------------------------------------------------
    # CHART PREVIEW
    # --------------------------------------------------------

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    _section_heading(
        "3",
        "Visualization Preview",
        "Explore your generated chart and inspect the underlying data patterns.",
    )

    if fig is not None:

        with st.container(border=True):

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:space-between;
                    flex-wrap:wrap;
                    gap:8px;
                    margin-bottom:8px;
                ">
                    <div style="
                        color:#172033;
                        font-size:16px;
                        font-weight:750;
                    ">
                        {chart_type}
                    </div>
                    <div style="
                        color:#64748B;
                        font-size:11px;
                    ">
                        AURORA · INTERACTIVE CHART
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="interactive_visualization",
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": True,
                },
            )

            st.caption(
                "Use the chart toolbar to zoom, pan, inspect data points "
                "and download the chart as an image."
            )

    else:
        _show_empty_state(
            "No chart could be generated from the selected configuration."
        )

    # --------------------------------------------------------
    # DATA EXPLORATION
    # --------------------------------------------------------

    with st.expander(
        "Explore available columns",
        expanded=False,
    ):

        column_summary = pd.DataFrame({
            "Column": plot_df.columns.astype(str),
            "Data Type": plot_df.dtypes.astype(str).values,
            "Non-null Records": plot_df.notna().sum().values,
            "Unique Values": plot_df.nunique(dropna=True).values,
        })

        st.dataframe(
            column_summary,
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        "Preview visualization data",
        expanded=False,
    ):

        preview_rows = st.slider(
            "Number of records",
            min_value=5,
            max_value=min(100, len(plot_df)),
            value=min(10, len(plot_df)),
            step=5,
            key="visualization_preview_rows",
        )

        st.dataframe(
            plot_df.head(preview_rows),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            f"Showing {min(preview_rows, len(plot_df)):,} "
            f"of {len(plot_df):,} records."
        )
