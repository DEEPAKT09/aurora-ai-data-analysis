import pandas as pd


def generate_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numerical columns."""
    numerical_df = df.select_dtypes(include="number")
    if numerical_df.empty:
        return pd.DataFrame()

    statistics = numerical_df.describe().T
    statistics["median"] = numerical_df.median()
    statistics["missing_values"] = numerical_df.isna().sum()
    statistics["skewness"] = numerical_df.skew()
    statistics["zero_values"] = numerical_df.eq(0).sum()
    return statistics.round(3)


def generate_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summaries for categorical, boolean and string columns."""
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns

    results = []
    for column in categorical_columns:
        series = df[column]
        mode_values = series.mode(dropna=True)
        most_frequent = mode_values.iloc[0] if not mode_values.empty else None
        frequency = int(series.value_counts(dropna=True).iloc[0]) if series.notna().any() else 0

        results.append({
            "Column": column,
            "Unique Values": int(series.nunique()),
            "Most Frequent Value": most_frequent,
            "Most Frequent Count": frequency,
            "Most Frequent Percentage": round(
                frequency / max(int(series.notna().sum()), 1) * 100, 2
            ),
            "Missing Values": int(series.isna().sum()),
        })

    return pd.DataFrame(results)


def generate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation between numerical columns."""
    numerical_df = df.select_dtypes(include="number")
    if numerical_df.shape[1] < 2:
        return pd.DataFrame()
    return numerical_df.corr().round(3)


def generate_column_summary(df: pd.DataFrame, column: str) -> dict:
    """Return a detailed summary for one selected column."""
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    series = df[column]
    non_null = series.dropna()
    summary = {
        "column": column,
        "dtype": str(series.dtype),
        "records": int(len(series)),
        "non_null": int(series.notna().sum()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        summary["statistics"] = {
            "mean": float(numeric.mean()) if numeric.notna().any() else None,
            "median": float(numeric.median()) if numeric.notna().any() else None,
            "std": float(numeric.std()) if numeric.notna().any() else None,
            "min": float(numeric.min()) if numeric.notna().any() else None,
            "max": float(numeric.max()) if numeric.notna().any() else None,
        }
    else:
        counts = non_null.astype(str).value_counts().head(10)
        summary["top_values"] = counts.to_dict()

    return summary
