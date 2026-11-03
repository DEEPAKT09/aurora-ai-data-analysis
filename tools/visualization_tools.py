
import pandas as pd
import plotly.express as px


def prepare_visualization_data(
    df: pd.DataFrame,
    date_columns: list | None = None
) -> pd.DataFrame:
    """
    Prepare a copy of the dataset for visualization.

    Converts recognized date columns into datetime format
    without modifying the original DataFrame.
    """

    plot_df = df.copy()

    for column in date_columns or []:

        if column in plot_df.columns:

            plot_df[column] = pd.to_datetime(
                plot_df[column],
                errors="coerce",
                format="mixed"
            )

    return plot_df


def create_histogram(
    df: pd.DataFrame,
    column: str
):
    """
    Create a histogram for a numerical column.
    """

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        marginal="box",
        template="plotly_white",
        color_discrete_sequence=["#2563EB"]
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        bargap=0.1
    )

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None = None,
    aggregation: str = "sum"
):
    """
    Create a categorical bar chart.

    If a Y column is selected, aggregate it by X.
    Otherwise, display category counts.
    """

    if y_column:

        if aggregation == "mean":
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column].mean()

        elif aggregation == "count":
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column].count()

        elif aggregation == "max":
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column].max()

        elif aggregation == "min":
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column].min()

        else:
            grouped = df.groupby(
                x_column,
                dropna=False
            )[y_column].sum()

        chart_df = grouped.reset_index(
            name=f"{aggregation.title()} of {y_column}"
        )

        y_axis = chart_df.columns[-1]

    else:

        chart_df = (
            df[x_column]
            .fillna("(Missing)")
            .value_counts()
            .rename_axis(x_column)
            .reset_index(name="Count")
        )

        y_axis = "Count"

    # Prevent unreadable charts with too many categories
    chart_df = chart_df.sort_values(
        y_axis,
        ascending=False
    ).head(30)

    fig = px.bar(
        chart_df,
        x=x_column,
        y=y_axis,
        title=f"{y_axis} by {x_column}",
        template="plotly_white",
        color_discrete_sequence=["#0891B2"]
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_axis,
        xaxis_tickangle=-30
    )

    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None
):
    """
    Create a scatter plot between two numerical columns.
    """

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"{y_column} vs {x_column}",
        template="plotly_white",
        opacity=0.75
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    return fig


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    aggregation: str = "sum"
):
    """
    Create a line chart to visualize trends.
    """

    if aggregation == "mean":
        grouped = df.groupby(
            x_column,
            dropna=False
        )[y_column].mean()

    elif aggregation == "count":
        grouped = df.groupby(
            x_column,
            dropna=False
        )[y_column].count()

    elif aggregation == "max":
        grouped = df.groupby(
            x_column,
            dropna=False
        )[y_column].max()

    elif aggregation == "min":
        grouped = df.groupby(
            x_column,
            dropna=False
        )[y_column].min()

    else:
        grouped = df.groupby(
            x_column,
            dropna=False
        )[y_column].sum()

    chart_df = grouped.reset_index(
        name=y_column
    )

    fig = px.line(
        chart_df,
        x=x_column,
        y=y_column,
        title=f"{y_column} Trend",
        markers=True,
        template="plotly_white",
        color_discrete_sequence=["#2563EB"]
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )

    return fig


def create_box_plot(
    df: pd.DataFrame,
    y_column: str,
    x_column: str | None = None
):
    """
    Create a box plot to examine distributions and outliers.
    """

    fig = px.box(
        df,
        x=x_column,
        y=y_column,
        title=f"Box Plot of {y_column}",
        points="outliers",
        template="plotly_white",
        color_discrete_sequence=["#7C3AED"]
    )

    return fig


def create_correlation_heatmap(
    df: pd.DataFrame
):
    """
    Create a correlation heatmap for numerical columns.
    """

    numerical_df = df.select_dtypes(
        include="number"
    )

    if numerical_df.shape[1] < 2:
        return None

    correlation = numerical_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numerical Correlation Heatmap",
        aspect="auto",
        template="plotly_white"
    )

    return fig
