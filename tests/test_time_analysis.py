import pandas as pd
import pytest

from tools.time_analysis import time_analysis, TIME_ANALYSIS_SCHEMA


def test_time_analysis_basic_monthly(sample_df: pd.DataFrame):
    """Test standard time_analysis monthly aggregation with best/worst period."""
    res = time_analysis(sample_df, date_column="Date", value_column="Weekly_Sales", period="month")
    assert "error" not in res
    assert res["date_column"] == "Date"
    assert res["value_column"] == "Weekly_Sales"
    assert res["period"] == "month"
    assert "2024-01" in res["result"]
    assert "2024-06" in res["result"]
    assert res["best_period"] == "2024-06"  # Sales=350
    assert res["worst_period"] == "2024-01"  # Sales=100


@pytest.mark.parametrize("period", ["day", "week", "month", "quarter", "year"])
def test_time_analysis_periods(sample_df: pd.DataFrame, period: str):
    """Test time_analysis across all allowed periods."""
    res = time_analysis(sample_df, date_column="Date", value_column="Weekly_Sales", period=period)
    assert "error" not in res
    assert len(res["result"]) > 0


def test_time_analysis_filter_year(df_multi_year: pd.DataFrame):
    """Test filtering time_analysis by a specific year."""
    res = time_analysis(df_multi_year, date_column="Date", value_column="Weekly_Sales", year=2024)
    assert "error" not in res
    assert all("2024" in label for label in res["result"].keys())


def test_time_analysis_date_range_filter(sample_df: pd.DataFrame):
    """Test filtering by start_date and end_date."""
    res = time_analysis(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        start_date="2024-02-01",
        end_date="2024-04-30",
    )
    assert "error" not in res
    assert set(res["result"].keys()) == {"2024-02", "2024-03", "2024-04"}


def test_time_analysis_invalid_date_format(sample_df: pd.DataFrame):
    """Test unparseable start_date or end_date."""
    res = time_analysis(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        start_date="not-a-date",
    )
    assert "error" in res
    assert "Could not parse start_date" in res["error"]


def test_time_analysis_secondary_group(sample_df: pd.DataFrame):
    """Test time analysis broken down by a secondary group_column."""
    res = time_analysis(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        group_column="Store",
        filter_values=["A", "B"],
    )
    assert "error" not in res
    assert res["group_column"] == "Store"
    assert isinstance(res["result"]["2024-01"], dict)


def test_time_analysis_no_matching_filter(sample_df: pd.DataFrame):
    """Test when date filters exclude all rows."""
    res = time_analysis(sample_df, date_column="Date", value_column="Weekly_Sales", year=1990)
    assert "error" in res
    assert "No rows remain" in res["error"]


def test_time_analysis_non_date_column(sample_df: pd.DataFrame):
    """Test passing a non-datetime column as date_column."""
    res = time_analysis(sample_df, date_column="Store", value_column="Weekly_Sales")
    assert "error" in res
    assert "not a recognized date column" in res["error"]


def test_time_analysis_non_numeric_value_column(sample_df: pd.DataFrame):
    """Test passing a non-numeric column as value_column for mean aggregation."""
    res = time_analysis(sample_df, date_column="Date", value_column="Store", agg_function="mean")
    assert "error" in res
    assert "not numeric" in res["error"]


def test_time_analysis_none_df():
    """Test time_analysis with df=None."""
    res = time_analysis(None, date_column="Date", value_column="Weekly_Sales")
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_time_analysis_schema():
    """Test TIME_ANALYSIS_SCHEMA required parameters."""
    schema = TIME_ANALYSIS_SCHEMA["function"]
    assert schema["name"] == "time_analysis"
    assert schema["parameters"]["required"] == ["date_column", "value_column"]


def test_time_analysis_increasing_trend(sample_df: pd.DataFrame):
    """Test strictly increasing trend identification."""
    res = time_analysis(sample_df, date_column="Date", value_column="Weekly_Sales", period="month")
    assert res["trend_direction"] == "strictly_increasing"
    assert res["overall_change"] == 250.0  # 350.0 - 100.0
    assert res["overall_percentage_change"] == 250.0  # (350 - 100) / 100 * 100


def test_time_analysis_decreasing_trend():
    """Test strictly decreasing trend identification."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "Sales": [300.0, 200.0, 100.0],
    })
    res = time_analysis(df, date_column="Date", value_column="Sales", period="month")
    assert res["trend_direction"] == "strictly_decreasing"
    assert res["overall_change"] == -200.0


def test_time_analysis_stable_trend():
    """Test stable trend identification."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "Sales": [150.0, 150.0, 150.0],
    })
    res = time_analysis(df, date_column="Date", value_column="Sales", period="month")
    assert res["trend_direction"] == "stable"
    assert res["overall_change"] == 0.0


def test_time_analysis_insufficient_data():
    """Test trend direction with only 1 period."""
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01"]),
        "Sales": [100.0],
    })
    res = time_analysis(df, date_column="Date", value_column="Sales", period="month")
    assert res["trend_direction"] == "insufficient_data"
    assert res["overall_change"] is None

