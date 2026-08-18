import pandas as pd
import pytest

from tools.date_utils import (
    ALLOWED_PERIODS,
    _name_looks_like_date,
    add_period_column,
    convert_date_columns,
    detect_date_columns,
    format_period_label,
    get_date_columns,
)


def test_name_looks_like_date():
    """Test date column name heuristics."""
    assert _name_looks_like_date("date") is True
    assert _name_looks_like_date("Order_Date") is True
    assert _name_looks_like_date("timestamp") is True
    assert _name_looks_like_date("sales_date") is True
    assert _name_looks_like_date("Store") is False
    assert _name_looks_like_date("Weekly_Sales") is False


def test_detect_date_columns(df_date_strings: pd.DataFrame):
    """Test detection of date strings in various formats and non-dates."""
    detected = detect_date_columns(df_date_strings)
    assert "Date_ISO" in detected
    assert "Date_US" in detected
    assert "Date_Slash" in detected
    assert "Not_A_Date" not in detected
    assert "Numeric_ID" not in detected


def test_convert_date_columns(df_date_strings: pd.DataFrame):
    """Test converting string date columns into pandas datetime."""
    df_converted, detected = convert_date_columns(df_date_strings)
    assert pd.api.types.is_datetime64_any_dtype(df_converted["Date_ISO"])
    assert pd.api.types.is_datetime64_any_dtype(df_converted["Date_US"])
    assert pd.api.types.is_datetime64_any_dtype(df_converted["Date_Slash"])
    assert not pd.api.types.is_datetime64_any_dtype(df_converted["Not_A_Date"])


def test_convert_date_columns_none_df():
    """Test convert_date_columns with None returns None, {}."""
    df_conv, det = convert_date_columns(None)
    assert df_conv is None
    assert det == {}


def test_get_date_columns(sample_df: pd.DataFrame):
    """Test get_date_columns helper."""
    cols = get_date_columns(sample_df)
    assert "Date" in cols


@pytest.mark.parametrize("period", ["day", "week", "month", "quarter", "year"])
def test_add_period_column_valid_periods(sample_df: pd.DataFrame, period: str):
    """Test add_period_column across all allowed periods."""
    res = add_period_column(sample_df["Date"], period)
    assert len(res) == len(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(res)


def test_add_period_column_invalid_period(sample_df: pd.DataFrame):
    """Test add_period_column with unsupported period raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported period"):
        add_period_column(sample_df["Date"], "century")


def test_format_period_label():
    """Test format_period_label for various periods and NaT."""
    ts = pd.Timestamp("2024-03-15")
    assert format_period_label(ts, "day") == "2024-03-15"
    assert format_period_label(ts, "week") == "Week of 2024-03-15"
    assert format_period_label(ts, "month") == "2024-03"
    assert format_period_label(ts, "quarter") == "2024-Q1"
    assert format_period_label(ts, "year") == "2024"
    assert format_period_label(pd.NaT, "month") == "Unknown"
