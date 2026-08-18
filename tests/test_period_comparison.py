import pandas as pd
from tools.period_comparison import _safe_pct_change, percentage_change, PERCENTAGE_CHANGE_SCHEMA


def test_percentage_change_regression_null_args(sample_df: pd.DataFrame):
    """Regression test: verify Groq null args for optional parameters don't break."""
    # 1. period=None
    res1 = percentage_change(sample_df, date_column="Date", value_column="Weekly_Sales", period=None)
    assert "error" not in res1
    assert "changes" in res1
    assert res1["period"] == "month"

    # 2. agg_function=None
    res2 = percentage_change(sample_df, date_column="Date", value_column="Weekly_Sales", agg_function=None)
    assert "error" not in res2
    assert "changes" in res2
    assert res2["agg_function"] == "sum"

    # 3. All optional arguments explicitly None
    res3 = percentage_change(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        period=None,
        agg_function=None,
        year_1=None,
        year_2=None,
        group_column=None,
        filter_values=None,
    )
    assert "error" not in res3
    assert "changes" in res3


def test_safe_pct_change_division_by_zero():
    """Test _safe_pct_change returns None when previous value is 0."""
    assert _safe_pct_change(0.0, 100.0) is None
    assert _safe_pct_change(100.0, 150.0) == 50.0
    assert _safe_pct_change(100.0, 50.0) == -50.0


def test_percentage_change_division_by_zero_dataset(df_zero_sales: pd.DataFrame):
    """Test percentage_change gracefully handles 0 base period value."""
    res = percentage_change(
        df_zero_sales,
        date_column="Date",
        value_column="Weekly_Sales",
        period="year",
    )
    assert "error" not in res
    assert res["latest_change"]["previous_value"] == 0.0
    assert res["latest_change"]["percentage_change"] is None


def test_percentage_change_year_vs_year(df_multi_year: pd.DataFrame):
    """Test Mode 1: explicit year_1 vs year_2 comparison."""
    res = percentage_change(
        df_multi_year,
        date_column="Date",
        value_column="Weekly_Sales",
        year_1=2023,
        year_2=2024,
        agg_function="sum",
    )
    assert "error" not in res
    assert res["previous_period"] == "2023"
    assert res["current_period"] == "2024"
    # 2023 sum: 100+200+300 = 600.0; 2024 sum: 150+250+350 = 750.0
    assert res["previous_value"] == 600.0
    assert res["current_value"] == 750.0
    assert res["absolute_change"] == 150.0
    assert res["percentage_change"] == 25.0


def test_percentage_change_partial_years_error(df_multi_year: pd.DataFrame):
    """Test providing only one of year_1 or year_2 returns an error."""
    res = percentage_change(
        df_multi_year,
        date_column="Date",
        value_column="Weekly_Sales",
        year_1=2023,
    )
    assert "error" in res
    assert "Both year_1 and year_2 must be provided" in res["error"]


def test_percentage_change_period_over_period(sample_df: pd.DataFrame):
    """Test Mode 2: period-over-period changes and largest increase/decrease."""
    res = percentage_change(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        period="month",
        agg_function="sum",
    )
    assert "error" not in res
    assert len(res["changes"]) == 5  # 6 months -> 5 changes
    assert "latest_change" in res
    assert "largest_increase" in res
    assert "largest_decrease" in res
    assert res["latest_change"]["current_period"] == "2024-06"


def test_percentage_change_invalid_columns(sample_df: pd.DataFrame):
    """Test non-existent date and value columns."""
    res_date = percentage_change(sample_df, date_column="BadDate", value_column="Weekly_Sales")
    assert "error" in res_date
    assert "BadDate" in res_date["error"]

    res_val = percentage_change(sample_df, date_column="Date", value_column="BadVal")
    assert "error" in res_val
    assert "BadVal" in res_val["error"]


def test_percentage_change_non_date_column(sample_df: pd.DataFrame):
    """Test non-datetime column as date_column."""
    res = percentage_change(sample_df, date_column="Store", value_column="Weekly_Sales")
    assert "error" in res
    assert "not a recognized date column" in res["error"]


def test_percentage_change_none_df():
    """Test percentage_change with df=None."""
    res = percentage_change(None, date_column="Date", value_column="Weekly_Sales")
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_percentage_change_schema():
    """Test PERCENTAGE_CHANGE_SCHEMA required properties and null annotations."""
    schema = PERCENTAGE_CHANGE_SCHEMA["function"]
    assert schema["name"] == "percentage_change"
    assert schema["parameters"]["required"] == ["date_column", "value_column"]
    assert "null" in schema["parameters"]["properties"]["period"]["type"]
    assert "null" in schema["parameters"]["properties"]["agg_function"]["type"]
    assert "null" in schema["parameters"]["properties"]["year_1"]["type"]
    assert "null" in schema["parameters"]["properties"]["year_2"]["type"]


def test_percentage_change_percentage_difference_and_summary(df_multi_year: pd.DataFrame):
    """Test percentage_difference and comparison_summary in year-over-year comparison."""
    res = percentage_change(
        df_multi_year,
        date_column="Date",
        value_column="Weekly_Sales",
        year_1=2023,
        year_2=2024,
    )
    assert "error" not in res
    assert res["comparison_summary"] == "increased"
    assert res["percentage_difference"] is not None
    assert res["percentage_change"] is not None


def test_percentage_change_overall_period_summary(sample_df: pd.DataFrame):
    """Test overall_change and total_periods_compared in period-over-period mode."""
    res = percentage_change(
        sample_df,
        date_column="Date",
        value_column="Weekly_Sales",
        period="month",
    )
    assert res["total_periods_compared"] == 5
    assert res["overall_change"] == 250.0  # 350 - 100
    assert res["overall_percentage_change"] == 250.0
