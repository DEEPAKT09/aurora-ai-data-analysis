from io import BytesIO

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_dataset(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """
    Load and validate a CSV or XLSX dataset.

    Args:
        file_name: Original uploaded filename.
        file_bytes: Uploaded file contents.

    Returns:
        A validated Pandas DataFrame.

    Raises:
        ValueError: If the file is invalid, unsupported, empty,
                    or cannot be processed.
    """

    # 1. Validate the uploaded file.
    if not file_name or "." not in file_name:
        raise ValueError(
            "Invalid filename. Please upload a CSV or XLSX file."
        )

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    extension = file_name.rsplit(".", 1)[-1].lower()

    if extension not in {"csv", "xlsx"}:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or XLSX file."
        )

    # 2. Read the file.
    file_buffer = BytesIO(file_bytes)

    try:
        if extension == "csv":
            try:
                df = pd.read_csv(
                    file_buffer,
                    encoding="utf-8-sig"
                )
            except UnicodeDecodeError:
                file_buffer.seek(0)
                df = pd.read_csv(
                    file_buffer,
                    encoding="latin-1"
                )

        else:
            df = pd.read_excel(
                file_buffer,
                engine="openpyxl"
            )

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "The uploaded file does not contain any readable data."
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            "The CSV file could not be parsed. Please check its structure "
            "and try uploading it again."
        ) from error

    except Exception as error:
        raise ValueError(
            "Unable to read the uploaded file. It may be corrupted, "
            "password-protected, or incorrectly formatted. "
            "Please check the file and try again."
        ) from error

    # 3. Validate the resulting DataFrame.
    if df.empty:
        raise ValueError(
            "The uploaded dataset contains no records."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "The uploaded dataset does not contain any columns."
        )

    # 4. Normalize column names.
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # 5. Reject empty column names.
    if any(not column for column in df.columns):
        raise ValueError(
            "The dataset contains empty column names. "
            "Please rename those columns and upload again."
        )

    # 6. Reject duplicate column names after trimming whitespace.
    if df.columns.duplicated().any():
        raise ValueError(
            "The dataset contains duplicate column names. "
            "Please make each column name unique and upload again."
        )

    # 7. Reject datasets that contain no usable records.
    if df.dropna(how="all").empty:
        raise ValueError(
            "The dataset contains no usable records. "
            "Please upload a file with at least one non-empty row."
        )

    return df