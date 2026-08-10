from __future__ import annotations

from typing import Any

import pandas as pd


# Common date-related words that help identify likely date columns.
DATE_NAME_HINTS = {
    "date",
    "datetime",
    "timestamp",
    "time",
    "day",
    "order_date",
    "orderdate",
    "sale_date",
    "saledate",
    "sales_date",
    "transaction_date",
    "transactiondate",
    "purchase_date",
    "purchasedate",
}


def _name_looks_like_date(column_name: str) -> bool:
    """
    Check whether a column name looks like a date/time column.
    """
    name = str(column_name).strip().lower()

    normalized = (
        name.replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

    if normalized in DATE_NAME_HINTS:
        return True

    date_keywords = [
        "date",
        "datetime",
        "timestamp",
        "order_time",
        "sale_time",
        "purchase_time",
        "transaction_time",
    ]

    return any(keyword in normalized for keyword in date_keywords)


def _parse_date_series(
    series: pd.Series,
) -> tuple[pd.Series | None, str | None, float]:
    """
    Try to parse a Series as dates.

    Returns:
        parsed_series
        detected_format
        success_ratio
    """

    clean = series.dropna()

    if clean.empty:
        return None, None, 0.0

    # Already datetime → no parsing necessary.
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, "datetime64", 1.0

    values = clean.astype(str).str.strip()

    # ---------------------------------------------------------
    # Try several common formats explicitly.
    #
    # Explicit formats are important because values such as
    # 01-05-2015 are ambiguous.
    # ---------------------------------------------------------

    formats = [
        ("%Y-%m-%d", "YYYY-MM-DD"),
        ("%Y/%m/%d", "YYYY/MM/DD"),
        ("%m-%d-%Y", "MM-DD-YYYY"),
        ("%m/%d/%Y", "MM/DD/YYYY"),
        ("%d-%m-%Y", "DD-MM-YYYY"),
        ("%d/%m/%Y", "DD/MM/YYYY"),
        ("%Y-%d-%m", "YYYY-DD-MM"),
        ("%Y/%d/%m", "YYYY/DD/MM"),
    ]

    candidates = []

    for fmt, label in formats:
        try:
            parsed = pd.to_datetime(
                values,
                format=fmt,
                errors="coerce",
            )

            ratio = parsed.notna().mean()

            if ratio > 0:
                candidates.append(
                    (
                        ratio,
                        label,
                        parsed,
                    )
                )

        except Exception:
            continue

    if not candidates:
        return None, None, 0.0

    # Highest successful parsing ratio.
    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_ratio, best_format, best_parsed = candidates[0]

    # ---------------------------------------------------------
    # Ambiguous formats
    #
    # Example:
    # 01-05-2015
    #
    # Both MM-DD-YYYY and DD-MM-YYYY may parse successfully.
    #
    # We use the following rule:
    #
    # If another candidate has a clearly better success ratio,
    # use it.
    #
    # Otherwise pandas' general interpretation is used as a
    # fallback rather than silently making a strong assumption.
    # ---------------------------------------------------------

    if len(candidates) > 1:

        second_ratio = candidates[1][0]

        if abs(best_ratio - second_ratio) < 0.001:

            try:
                fallback = pd.to_datetime(
                    values,
                    errors="coerce",
                )

                fallback_ratio = fallback.notna().mean()

                if fallback_ratio >= best_ratio:
                    best_parsed = fallback
                    best_format = "auto-detected"

            except Exception:
                pass

    # Only accept the column if almost all non-null values
    # can be interpreted as dates.
    if best_ratio < 0.80:
        return None, None, best_ratio

    # Reconstruct a Series with the original index.
    parsed_full = pd.to_datetime(
        series,
        format=(
            None
            if best_format == "auto-detected"
            else next(
                fmt
                for fmt, label in formats
                if label == best_format
            )
        ),
        errors="coerce",
    )

    return (
        parsed_full,
        best_format,
        float(best_ratio),
    )


def detect_date_columns(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Detect columns that appear to contain dates.

    Returns metadata such as:

    {
        "Date": {
            "format": "MM-DD-YYYY",
            "confidence": 1.0,
        }
    }
    """

    if df is None:
        return {}

    detected = {}

    for column in df.columns:

        series = df[column]

        # Already datetime.
        if pd.api.types.is_datetime64_any_dtype(series):

            detected[str(column)] = {
                "format": "datetime64",
                "confidence": 1.0,
            }

            continue

        # Numeric columns are not considered dates.
        if pd.api.types.is_numeric_dtype(series):
            continue

        parsed, detected_format, ratio = _parse_date_series(
            series
        )

        if parsed is None:
            continue

        # Give a small confidence boost to columns whose names
        # strongly indicate a date.
        confidence = ratio

        if _name_looks_like_date(str(column)):
            confidence = min(1.0, confidence + 0.05)

        detected[str(column)] = {
            "format": detected_format,
            "confidence": round(
                float(confidence),
                3,
            ),
        }

    return detected


def convert_date_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Detect and convert date columns to pandas datetime.

    Returns:
        converted_dataframe
        detected_date_metadata
    """

    if df is None:
        return df, {}

    result = df.copy()

    detected = detect_date_columns(result)

    for column, metadata in detected.items():

        try:

            if metadata["format"] == "datetime64":

                result[column] = pd.to_datetime(
                    result[column],
                    errors="coerce",
                )

            elif metadata["format"] == "auto-detected":

                result[column] = pd.to_datetime(
                    result[column],
                    errors="coerce",
                )

            else:

                format_map = {
                    "YYYY-MM-DD": "%Y-%m-%d",
                    "YYYY/MM/DD": "%Y/%m/%d",
                    "MM-DD-YYYY": "%m-%d-%Y",
                    "MM/DD/YYYY": "%m/%d/%Y",
                    "DD-MM-YYYY": "%d-%m-%Y",
                    "DD/MM/YYYY": "%d/%m/%Y",
                    "YYYY-DD-MM": "%Y-%d-%m",
                    "YYYY/DD/MM": "%Y/%d/%m",
                }

                fmt = format_map.get(
                    metadata["format"]
                )

                result[column] = pd.to_datetime(
                    result[column],
                    format=fmt,
                    errors="coerce",
                )

        except Exception:
            # Never prevent the dataset from loading because
            # date detection failed.
            continue

    return result, detected


def get_date_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return the names of detected date columns.
    """

    return list(
        detect_date_columns(df).keys()
    )
