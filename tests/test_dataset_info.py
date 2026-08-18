import numpy as np
import pandas as pd
import pytest

from tools.dataset_info import (
    DATASET_INFO_SCHEMA,
    MAX_CATEGORY_SAMPLES,
    build_dataset_profile,
    dataset_info,
    format_dataset_context,
)


def test_dataset_info_happy_path(sample_df: pd.DataFrame):
    """Test dataset_info on a standard valid DataFrame."""
    info = dataset_info(sample_df)
    assert "error" not in info
    assert info["num_rows"] == 6
    assert info["num_columns"] == 6
    assert "Weekly_Sales" in info["column_names"]
    assert "Date" in info["date_columns"]
    assert info["memory_usage_kb"] > 0
    assert "Weekly_Sales" in info["numeric_columns"]
    assert "Store" in info["categorical_columns"]
    assert "Date" in info["datetime_columns"]
    assert "IsHoliday" in info["boolean_columns"]
    assert "Store" in info["candidate_group_columns"]


def test_dataset_info_none_df():
    """Test dataset_info with df=None returns an error dict."""
    info = dataset_info(None)
    assert "error" in info
    assert "No dataset is loaded" in info["error"]


def test_dataset_info_empty_df(empty_df: pd.DataFrame):
    """Test dataset_info with an empty DataFrame."""
    info = dataset_info(empty_df)
    assert "error" not in info
    assert info["num_rows"] == 0
    assert info["num_columns"] == 0
    assert info["column_names"] == []
    assert info["date_columns"] == []
    assert info["numeric_columns"] == []
    assert info["categorical_columns"] == []


def test_dataset_info_schema():
    """Test DATASET_INFO_SCHEMA structure."""
    assert DATASET_INFO_SCHEMA["type"] == "function"
    assert DATASET_INFO_SCHEMA["function"]["name"] == "dataset_info"
    assert "parameters" in DATASET_INFO_SCHEMA["function"]


def test_profile_semantic_type_classification(sample_df: pd.DataFrame):
    """Test semantic type classification across numeric, categorical, datetime, boolean."""
    profile = build_dataset_profile(sample_df)
    assert profile["semantic_types"]["Weekly_Sales"] == "numeric"
    assert profile["semantic_types"]["Temperature"] == "numeric"
    assert profile["semantic_types"]["Store"] == "categorical"
    assert profile["semantic_types"]["Date"] == "datetime"
    assert profile["semantic_types"]["IsHoliday"] == "boolean"


def test_profile_missing_and_all_null(df_missing: pd.DataFrame):
    """Test detection of missing values and all-null columns."""
    profile = build_dataset_profile(df_missing)
    assert "All_Null" in profile["all_null_columns"]
    assert "A" in profile["missing_summary"]
    assert profile["missing_summary"]["A"]["missing_count"] == 1
    assert profile["missing_summary"]["A"]["missing_percentage"] == 20.0
    assert profile["missing_summary"]["All_Null"]["missing_count"] == 5
    assert profile["missing_summary"]["All_Null"]["missing_percentage"] == 100.0


def test_profile_constant_columns(df_constants: pd.DataFrame):
    """Test detection of constant / zero-variance columns."""
    profile = build_dataset_profile(df_constants)
    assert "Constant" in profile["constant_columns"]
    assert "Varying_1" not in profile["constant_columns"]


def test_profile_identifier_detection(df_identifiers: pd.DataFrame):
    """Test conservative identifier detection distinguishes detected IDs from normal categories."""
    profile = build_dataset_profile(df_identifiers)
    id_cols = [item["column"] for item in profile["potential_identifiers"]]
    assert "user_id" in id_cols
    assert "order_code" in id_cols

    # Normal categorical column 'City' should NOT be an identifier
    assert "City" not in id_cols
    # 'City' should be recommended as a candidate group column
    assert "City" in profile["candidate_group_columns"]
    # Identifiers should NOT be in candidate_group_columns
    assert "user_id" not in profile["candidate_group_columns"]
    assert "order_code" not in profile["candidate_group_columns"]


def test_profile_bounded_category_samples():
    """Test that sample_values for high-cardinality categorical columns remain bounded."""
    df = pd.DataFrame({"HighCard": [f"item_{i}" for i in range(100)]})
    profile = build_dataset_profile(df)
    samples = profile["column_profiles"]["HighCard"]["sample_values"]
    assert len(samples) <= MAX_CATEGORY_SAMPLES


def test_profile_date_strings_detection(df_date_strings: pd.DataFrame):
    """Test that date string columns are correctly recognized as datetime semantic type."""
    profile = build_dataset_profile(df_date_strings)
    assert "Date_ISO" in profile["datetime_columns"]
    assert "Date_US" in profile["datetime_columns"]
    assert "Date_Slash" in profile["datetime_columns"]
    assert "Not_A_Date" not in profile["datetime_columns"]
    assert "Not_A_Date" in profile["categorical_columns"]


def test_profile_no_numeric_dataset(df_no_numeric: pd.DataFrame):
    """Test profiling a dataset with no numeric columns."""
    profile = build_dataset_profile(df_no_numeric)
    assert profile["numeric_columns"] == []
    assert "Name" in profile["categorical_columns"]
    assert "Department" in profile["categorical_columns"]
    assert "JoinDate" in profile["datetime_columns"]


def test_profile_no_categorical_dataset(df_no_categorical: pd.DataFrame):
    """Test profiling a dataset with no categorical columns."""
    profile = build_dataset_profile(df_no_categorical)
    assert profile["categorical_columns"] == []
    assert "Sales" in profile["numeric_columns"]
    assert "Units" in profile["numeric_columns"]
    assert "Date" in profile["datetime_columns"]


def test_profile_no_date_dataset(df_no_date: pd.DataFrame):
    """Test profiling a dataset with no date columns."""
    profile = build_dataset_profile(df_no_date)
    assert profile["datetime_columns"] == []
    assert "Category" in profile["categorical_columns"]
    assert "Price" in profile["numeric_columns"]


def test_profile_multiple_date_columns():
    """Test dataset with multiple datetime columns."""
    df = pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "ship_date": pd.to_datetime(["2024-01-05", "2024-01-06"]),
        "amount": [50.0, 100.0],
    })
    profile = build_dataset_profile(df)
    assert "order_date" in profile["datetime_columns"]
    assert "ship_date" in profile["datetime_columns"]


def test_format_dataset_context_output(sample_df: pd.DataFrame):
    """Test format_dataset_context creates a concise, structured markdown/text block."""
    context_text = format_dataset_context(sample_df)
    assert "[Active Dataset Context]" in context_text
    assert "Shape: 6 rows, 6 columns" in context_text
    assert "Weekly_Sales" in context_text
    assert "Store" in context_text
    assert "Date" in context_text
    assert "IsHoliday" in context_text


def test_format_dataset_context_empty_and_none(empty_df: pd.DataFrame):
    """Test format_dataset_context returns empty string for None or empty DataFrame."""
    assert format_dataset_context(None) == ""
    assert format_dataset_context(empty_df) == ""
