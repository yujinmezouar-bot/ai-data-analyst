import pandas as pd
from tools.missing_values import missing_values, MISSING_VALUES_SCHEMA


def test_missing_values_with_missing(df_missing: pd.DataFrame):
    """Test missing_values correctly calculates counts and percentages."""
    res = missing_values(df_missing)
    assert "error" not in res
    assert res["has_missing_values"] is True
    assert res["total_rows"] == 5
    assert res["total_missing_values"] == 7  # A:1, B:1, All_Null:5 -> 7
    assert "A" in res["columns_with_missing"]
    assert res["columns_with_missing"]["A"]["missing_count"] == 1
    assert res["columns_with_missing"]["A"]["missing_percentage"] == 20.0
    assert res["columns_with_missing"]["All_Null"]["missing_count"] == 5
    assert res["columns_with_missing"]["All_Null"]["missing_percentage"] == 100.0
    assert "C" not in res["columns_with_missing"]


def test_missing_values_no_missing(sample_df: pd.DataFrame):
    """Test missing_values when there are no missing values."""
    res = missing_values(sample_df)
    assert "error" not in res
    assert res["has_missing_values"] is False
    assert res["total_missing_values"] == 0
    assert res["columns_with_missing"] == {}


def test_missing_values_none_df():
    """Test missing_values with df=None."""
    res = missing_values(None)
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_missing_values_empty_df(empty_df: pd.DataFrame):
    """Test missing_values on empty DataFrame."""
    res = missing_values(empty_df)
    assert "error" not in res
    assert res["total_rows"] == 0
    assert res["total_missing_values"] == 0
    assert res["has_missing_values"] is False


def test_missing_values_schema():
    """Test MISSING_VALUES_SCHEMA structure."""
    assert MISSING_VALUES_SCHEMA["type"] == "function"
    assert MISSING_VALUES_SCHEMA["function"]["name"] == "missing_values"
