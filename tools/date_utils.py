from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================
# DATE COLUMN DETECTION
# This is the single, consolidated date-handling implementation
# used by app.py (on upload) and dataset_info.py (for reporting).
# ============================================================

DATE_NAME_HINTS = {
    "date", "datetime", "timestamp", "time", "day",
    "order_date", "orderdate", "sale_date", "saledate",
    "sales_date", "transaction_date", "transactiondate",
    "purchase_date", "purchasedate",
}


def _name_looks_like_date(column_name: str) -> bool:
    name = str(column_name).strip().lower()
    normalized = name.replace("-", "_").replace("/", "_").replace(" ", "_")

    if normalized in DATE_NAME_HINTS:
        return True

    date_keywords = [
        "date", "datetime", "timestamp",
        "order_time", "sale_time", "purchase_time", "transaction_time",
    ]
    return any(keyword in normalized for keyword in date_keywords)


def _parse_date_series(series: pd.Series) -> tuple[pd.Series | None, str | None, float]:
    """
    Try to parse a Series as dates by scoring several explicit formats.
    This matters for ambiguous values like 03-04-2024, where both
    MM-DD-YYYY and DD-MM-YYYY could apply -- we never silently guess.

    Returns (parsed_series, detected_format, success_ratio).
    """
    clean = series.dropna()

    if clean.empty:
        return None, None, 0.0

    if pd.api.types.is_datetime64_any_dtype(series):
        return series, "datetime64", 1.0

    values = clean.astype(str).str.strip()

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
            parsed = pd.to_datetime(values, format=fmt, errors="coerce")
            ratio = parsed.notna().mean()
            if ratio > 0:
                candidates.append((ratio, label, parsed))
        except Exception:
            continue

    if not candidates:
        return None, None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_ratio, best_format, best_parsed = candidates[0]

    # Ambiguous case: two formats parse equally well (e.g. 03-04-2024 could
    # be MM-DD or DD-MM). Only fall back to pandas' generic parser if it
    # does at least as well as the best explicit format -- never silently
    # assume day-first or month-first when the evidence is a genuine tie.
    if len(candidates) > 1:
        second_ratio = candidates[1][0]
        if abs(best_ratio - second_ratio) < 0.001:
            try:
                fallback = pd.to_datetime(values, errors="coerce")
                fallback_ratio = fallback.notna().mean()
                if fallback_ratio >= best_ratio:
                    best_parsed = fallback
                    best_format = "auto-detected"
            except Exception:
                pass

    # Only accept the column as a date if almost all values parsed.
    if best_ratio < 0.80:
        return None, None, best_ratio

    parsed_full = pd.to_datetime(
        series,
        format=None if best_format == "auto-detected" else next(
            fmt for fmt, label in formats if label == best_format
        ),
        errors="coerce",
    )

    return parsed_full, best_format, float(best_ratio)


def detect_date_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Detect columns that appear to contain dates.

    Returns metadata such as:
        {"Date": {"format": "MM-DD-YYYY", "confidence": 1.0}}
    """
    if df is None:
        return {}

    detected = {}

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            detected[str(column)] = {"format": "datetime64", "confidence": 1.0}
            continue

        if pd.api.types.is_numeric_dtype(series):
            continue

        parsed, detected_format, ratio = _parse_date_series(series)

        if parsed is None:
            continue

        confidence = ratio
        if _name_looks_like_date(str(column)):
            confidence = min(1.0, confidence + 0.05)

        detected[str(column)] = {
            "format": detected_format,
            "confidence": round(float(confidence), 3),
        }

    return detected


def convert_date_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Detect and convert date columns to pandas datetime.

    This is called once, from app.py's loader, right after a file is
    uploaded -- it is the ONLY place date conversion happens in the app.
    """
    if df is None:
        return df, {}

    result = df.copy()
    detected = detect_date_columns(result)

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

    for column, metadata in detected.items():
        try:
            if metadata["format"] in ("datetime64", "auto-detected"):
                result[column] = pd.to_datetime(result[column], errors="coerce")
            else:
                fmt = format_map.get(metadata["format"])
                result[column] = pd.to_datetime(result[column], format=fmt, errors="coerce")
        except Exception:
            # Never prevent the dataset from loading because date
            # detection/conversion failed on one column.
            continue

    return result, detected


def get_date_columns(df: pd.DataFrame) -> list[str]:
    return list(detect_date_columns(df).keys())


# ============================================================
# PERIOD BUCKETING
# Shared by tools/time_analysis.py and tools/visualization.py so
# there is one place that defines what "by month" / "by quarter"
# etc. actually mean.
# ============================================================

ALLOWED_PERIODS = {"day", "week", "month", "quarter", "year"}


def add_period_column(series: pd.Series, period: str) -> pd.Series:
    """
    Bucket a datetime Series into period-start timestamps. Returned
    values sort chronologically since they are real pandas Timestamps.
    """
    if period not in ALLOWED_PERIODS:
        raise ValueError(f"Unsupported period '{period}'. Allowed: {sorted(ALLOWED_PERIODS)}")

    if period == "day":
        return series.dt.floor("D")
    if period == "week":
        return series.dt.to_period("W").dt.start_time
    if period == "month":
        return series.dt.to_period("M").dt.start_time
    if period == "quarter":
        return series.dt.to_period("Q").dt.start_time
    if period == "year":
        return series.dt.to_period("Y").dt.start_time

    raise ValueError(f"Unsupported period '{period}'.")


def format_period_label(timestamp: pd.Timestamp, period: str) -> str:
    """Human-readable label for a period bucket, e.g. '2024-03' or '2024-Q1'."""
    if pd.isna(timestamp):
        return "Unknown"

    if period == "day":
        return timestamp.strftime("%Y-%m-%d")
    if period == "week":
        return f"Week of {timestamp.strftime('%Y-%m-%d')}"
    if period == "month":
        return timestamp.strftime("%Y-%m")
    if period == "quarter":
        quarter = (timestamp.month - 1) // 3 + 1
        return f"{timestamp.year}-Q{quarter}"
    if period == "year":
        return timestamp.strftime("%Y")

    return timestamp.strftime("%Y-%m-%d")