from typing import Literal

from pydantic import BaseModel, Field

from agents.query_planner import get_gemini_model
from tools.visualization_tools import (
    create_histogram,
    create_bar_chart,
    create_scatter_plot,
    create_line_chart,
    create_box_plot,
    create_correlation_heatmap
)


class VisualizationPlan(BaseModel):

    chart_type: Literal[
        "histogram",
        "bar",
        "scatter",
        "line",
        "box",
        "correlation_heatmap"
    ] = Field(
        description="The type of chart to create."
    )

    x_column: str | None = Field(
        default=None,
        description="Column for the X axis."
    )

    y_column: str | None = Field(
        default=None,
        description="Numerical column for the Y axis."
    )

    color_column: str | None = Field(
        default=None,
        description="Optional column used to distinguish data groups by color."
    )

    aggregation: Literal[
        "sum",
        "mean",
        "count",
        "min",
        "max"
    ] = Field(
        default="sum",
        description="Aggregation for grouped bar and line charts."
    )

    explanation: str = Field(
        description="Brief explanation of the selected chart."
    )


def create_visualization_plan(
    user_question: str,
    df
) -> VisualizationPlan:

    model = get_gemini_model()

    structured_model = model.with_structured_output(
        VisualizationPlan,
        method="json_schema"
    )

    column_information = [
        {
            "name": str(column),
            "dtype": str(df[column].dtype),
            "unique_values": int(df[column].nunique())
        }
        for column in df.columns
    ]

    prompt = f"""
You are the Visualization Agent of an AI Data Analyst platform.

Your task is to convert the user's natural-language request
into a structured chart plan.

AVAILABLE CHARTS:

1. histogram:
Show the distribution of one numerical column.
Use x_column.

2. bar:
Compare categories or show aggregated numerical values.
Use x_column for categories and y_column for numerical values.
If the user asks for frequency or count, y_column can be null.
Use aggregation to select sum, mean, count, min or max.

3. scatter:
Show the relationship between two numerical columns.
Use x_column and y_column.
color_column is optional.

4. line:
Show trends over time or ordered values.
Use x_column and y_column.
Use aggregation when values need grouping.

5. box:
Show the distribution and spread of a numerical column.
Use y_column. x_column is optional for category-wise comparison.

6. correlation_heatmap:
Show correlations among numerical columns.
No column selection is required.

DATASET COLUMNS:
{column_information}

USER REQUEST:
{user_question}

RULES:
- Use only exact column names from the dataset.
- Never invent column names.
- Do not generate Python code or SQL.
- Do not use identifiers as measurements without a clear reason.
- Use histogram for distribution requests.
- Use bar for category comparisons.
- Use scatter for relationships between two numerical columns.
- Use line for trends over time.
- Use box for spread and outlier comparisons.
- Use correlation_heatmap for correlation matrix requests.
- Return a structured plan only.
"""

    plan = structured_model.invoke(prompt)

    valid_columns = set(str(column) for column in df.columns)

    for column in [
        plan.x_column,
        plan.y_column,
        plan.color_column
    ]:
        if column is not None and column not in valid_columns:
            raise ValueError(
                f"Invalid column selected by Gemini: {column}"
            )

    return plan


def generate_visualization(
    user_question: str,
    df
) -> dict:

    if df is None or df.empty:
        raise ValueError(
            "Please upload a valid dataset first."
        )

    plan = create_visualization_plan(
        user_question,
        df
    )

    chart_type = plan.chart_type

    # Validate and create the selected chart.
    if chart_type == "histogram":

        if not plan.x_column:
            raise ValueError(
                "A numerical column is required for a histogram."
            )

        if not __import__("pandas").api.types.is_numeric_dtype(
            df[plan.x_column]
        ):
            raise ValueError(
                "Histogram requires a numerical column."
            )

        figure = create_histogram(
            df,
            plan.x_column
        )

    elif chart_type == "bar":

        if not plan.x_column:
            raise ValueError(
                "A category column is required for a bar chart."
            )

        figure = create_bar_chart(
            df,
            plan.x_column,
            plan.y_column,
            plan.aggregation
        )

    elif chart_type == "scatter":

        if not plan.x_column or not plan.y_column:
            raise ValueError(
                "Scatter plots require two numerical columns."
            )

        import pandas as pd

        if not pd.api.types.is_numeric_dtype(df[plan.x_column]):
            raise ValueError(
                f"{plan.x_column} must be numerical."
            )

        if not pd.api.types.is_numeric_dtype(df[plan.y_column]):
            raise ValueError(
                f"{plan.y_column} must be numerical."
            )

        figure = create_scatter_plot(
            df,
            plan.x_column,
            plan.y_column,
            plan.color_column
        )

    elif chart_type == "line":

        if not plan.x_column or not plan.y_column:
            raise ValueError(
                "Line charts require X and Y columns."
            )

        figure = create_line_chart(
            df,
            plan.x_column,
            plan.y_column,
            plan.aggregation
        )

    elif chart_type == "box":

        if not plan.y_column:
            raise ValueError(
                "A numerical Y column is required for a box plot."
            )

        figure = create_box_plot(
            df,
            plan.y_column,
            plan.x_column
        )

    elif chart_type == "correlation_heatmap":

        figure = create_correlation_heatmap(df)

        if figure is None:
            raise ValueError(
                "At least two numerical columns are required "
                "for a correlation heatmap."
            )

    else:
        raise ValueError(
            "Unsupported visualization type."
        )

    if figure is None:
        raise ValueError(
            "The visualization tool did not return a chart."
        )

    return {
        "chart_type": chart_type,
        "x_column": plan.x_column,
        "y_column": plan.y_column,
        "color_column": plan.color_column,
        "aggregation": plan.aggregation,
        "explanation": plan.explanation,
        "chart_json": figure.to_json()
    }