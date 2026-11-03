
import pandas as pd


def execute_analysis(df: pd.DataFrame, plan) -> dict:
    """
    Execute a validated analytical plan using Pandas.

    The function never executes model-generated code.
    """

    operation = plan.operation

    if operation == "dataset_overview":

        return {
            "operation": operation,
            "records": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }

    elif operation == "missing_values":

        missing = df.isna().sum()

        result = {
            column: int(value)
            for column, value in missing.items()
        }

        return {
            "operation": operation,
            "total_missing_values": int(missing.sum()),
            "missing_by_column": result
        }

    elif operation == "column_summary":

        if not plan.column:
            raise ValueError("A column must be specified.")

        series = df[plan.column]

        if pd.api.types.is_numeric_dtype(series):

            return {
                "operation": operation,
                "column": plan.column,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "standard_deviation": float(series.std())
            }

        return {
            "operation": operation,
            "column": plan.column,
            "unique_values": int(series.nunique()),
            "most_frequent_value": (
                str(series.mode().iloc[0])
                if not series.mode().empty
                else None
            ),
            "missing_values": int(series.isna().sum()),
            "value_counts": {
                str(key): int(value)
                for key, value in (
                    series.value_counts().head(10).items()
                )
            }
        }

    elif operation == "aggregate":

        if not plan.column:
            raise ValueError(
                "A numerical column must be specified."
            )

        if not pd.api.types.is_numeric_dtype(df[plan.column]):
            raise ValueError(
                f"{plan.column} is not a numerical column."
            )

        aggregation_functions = {
            "sum": "sum",
            "mean": "mean",
            "count": "count",
            "min": "min",
            "max": "max"
        }

        aggregation = aggregation_functions[
            plan.aggregation
        ]

        if plan.group_by:

            grouped = df.groupby(
                plan.group_by,
                dropna=False
            )[plan.column].agg(aggregation)

            result = {
                str(key): (
                    float(value)
                    if pd.notna(value)
                    else None
                )
                for key, value in grouped.items()
            }

            # Return top groups first
            reverse = aggregation in ["sum", "mean", "count", "max"]

            sorted_result = dict(
                sorted(
                    result.items(),
                    key=lambda item: (
                        item[1] is not None,
                        item[1] if item[1] is not None else 0
                    ),
                    reverse=reverse
                )
            )

            return {
                "operation": operation,
                "column": plan.column,
                "group_by": plan.group_by,
                "aggregation": aggregation,
                "results": sorted_result
            }

        value = getattr(df[plan.column], aggregation)()

        return {
            "operation": operation,
            "column": plan.column,
            "aggregation": aggregation,
            "result": (
                float(value)
                if pd.notna(value)
                else None
            )
        }

    elif operation == "top_categories":

        if not plan.column:
            raise ValueError(
                "A categorical column must be specified."
            )

        counts = (
            df[plan.column]
            .value_counts(dropna=False)
            .head(10)
        )

        return {
            "operation": operation,
            "column": plan.column,
            "results": {
                str(key): int(value)
                for key, value in counts.items()
            }
        }

    elif operation == "correlation":

        if not plan.column or not plan.second_column:
            raise ValueError(
                "Two numerical columns must be specified."
            )

        if not pd.api.types.is_numeric_dtype(df[plan.column]):
            raise ValueError(
                f"{plan.column} must be numerical."
            )

        if not pd.api.types.is_numeric_dtype(df[plan.second_column]):
            raise ValueError(
                f"{plan.second_column} must be numerical."
            )

        correlation = df[
            [plan.column, plan.second_column]
        ].corr().iloc[0, 1]

        return {
            "operation": operation,
            "column_1": plan.column,
            "column_2": plan.second_column,
            "correlation": (
                float(correlation)
                if pd.notna(correlation)
                else None
            )
        }

    else:

        return {
            "operation": "unsupported",
            "message": (
                "The requested analysis is not supported "
                "by the current analytical engine."
            )
        }
