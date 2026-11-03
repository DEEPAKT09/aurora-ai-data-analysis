import pandas as pd


def identify_column_types(df: pd.DataFrame) -> dict:
    """Identify numerical, categorical, text and date columns."""
    numerical_columns, categorical_columns = [], []
    text_columns, date_columns = [], []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numerical_columns.append(column)
        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            non_null = series.dropna()
            if len(non_null) == 0:
                categorical_columns.append(column)
                continue

            sample = non_null.head(100)
            try:
                parsed = pd.to_datetime(
                    sample, errors="coerce", format="mixed"
                )
                date_ratio = parsed.notna().mean()
            except (TypeError, ValueError):
                date_ratio = 0

            if date_ratio >= 0.8:
                date_columns.append(column)
            elif non_null.nunique() <= 50:
                categorical_columns.append(column)
            else:
                text_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "numerical": numerical_columns,
        "categorical": categorical_columns,
        "text": text_columns,
        "date": date_columns,
    }


def _column_category(column: str, column_types: dict) -> str:
    for category, columns in column_types.items():
        if column in columns:
            return category
    return "other"


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate dataset-level and column-level profiling information."""
    total_records, total_columns = df.shape
    total_cells = total_records * total_columns
    missing_by_column = df.isna().sum()
    missing_values = int(missing_by_column.sum())
    duplicate_rows = int(df.duplicated().sum())
    column_types = identify_column_types(df)

    column_details = []
    constant_columns = []
    high_cardinality_columns = []
    potential_identifier_columns = []

    for column in df.columns:
        series = df[column]
        non_null = int(series.count())
        unique_values = int(series.nunique(dropna=True))
        missing_count = int(missing_by_column[column])
        unique_ratio = unique_values / max(non_null, 1)
        category = _column_category(column, column_types)

        if unique_values <= 1 and non_null > 0:
            constant_columns.append(column)

        if non_null >= 10 and unique_ratio >= 0.8:
            high_cardinality_columns.append(column)

        name_hint = any(
            token in str(column).strip().lower()
            for token in ("id", "uuid", "key", "code")
        )
        if non_null >= 10 and unique_ratio >= 0.9 and (
            name_hint or unique_ratio >= 0.98
        ):
            potential_identifier_columns.append(column)

        column_details.append({
            "Column": str(column),
            "Data Type": str(series.dtype),
            "Column Category": category,
            "Non-Null Values": non_null,
            "Missing Values": missing_count,
            "Missing Percentage": round(
                missing_count / max(total_records, 1) * 100, 2
            ),
            "Unique Values": unique_values,
            "Unique Percentage": round(unique_ratio * 100, 2),
            "Duplicate Values": int(series.duplicated().sum()),
            "Constant": unique_values <= 1 and non_null > 0,
            "Potential Identifier": column in potential_identifier_columns,
        })

    return {
        "total_records": int(total_records),
        "total_columns": int(total_columns),
        "missing_values": missing_values,
        "missing_percentage": round(
            missing_values / total_cells * 100 if total_cells else 0, 2
        ),
        "duplicate_rows": duplicate_rows,
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        ),
        "column_types": column_types,
        "column_details": column_details,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "potential_identifier_columns": potential_identifier_columns,
        "missing_by_column": {
            str(column): int(value)
            for column, value in missing_by_column.items()
        },
    }


def generate_data_quality_summary(profile: dict) -> list:
    """Generate rule-based, evidence-backed data quality observations."""
    observations = []

    if profile["missing_values"] == 0:
        observations.append("No missing values were detected.")
    else:
        observations.append(
            f'{profile["missing_values"]} missing values were detected '
            f'({profile["missing_percentage"]}% of all cells).'
        )

    if profile["duplicate_rows"] == 0:
        observations.append("No duplicate rows were detected.")
    else:
        observations.append(
            f'{profile["duplicate_rows"]} duplicate rows were detected.'
        )

    observations.append(
        f'{len(profile["column_types"]["numerical"])} numerical columns '
        "were identified."
    )
    observations.append(
        f'{len(profile["column_types"]["categorical"])} categorical '
        "columns were identified."
    )
    observations.append(
        f'{len(profile["column_types"]["date"])} date columns were identified.'
    )

    if profile.get("constant_columns"):
        observations.append(
            "Constant columns: " + ", ".join(profile["constant_columns"])
        )
    if profile.get("potential_identifier_columns"):
        observations.append(
            "Potential identifier columns: "
            + ", ".join(profile["potential_identifier_columns"])
            + ". Review these before predictive modelling."
        )
    if profile.get("high_cardinality_columns"):
        observations.append(
            "High-cardinality columns: "
            + ", ".join(profile["high_cardinality_columns"])
        )

    return observations
