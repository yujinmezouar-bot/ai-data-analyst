import pandas as pd
import pytest
from tools.groupby import groupby_analysis, GROUPBY_ANALYSIS_SCHEMA, MAX_GROUPS_RETURNED


def test_groupby_analysis_basic(sample_df: pd.DataFrame):
    """Test groupby_analysis with default aggregation (mean)."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Weekly_Sales")
    assert "error" not in res
    assert res["group_column"] == "Store"
    assert res["value_column"] == "Weekly_Sales"
    assert res["agg_function"] == "mean"
    # Store A: (100 + 250) / 2 = 175.0
    # Store B: (150 + 300) / 2 = 225.0
    # Store C: (200 + 350) / 2 = 275.0
    assert res["result"]["A"] == 175.0
    assert res["result"]["B"] == 225.0
    assert res["result"]["C"] == 275.0


@pytest.mark.parametrize("agg,expected_a", [
    ("sum", 350.0),
    ("count", 2),
    ("min", 100.0),
    ("max", 250.0),
    ("median", 175.0),
])
def test_groupby_analysis_aggregations(sample_df: pd.DataFrame, agg: str, expected_a: float):
    """Test groupby_analysis with multiple aggregation functions."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Weekly_Sales", agg_function=agg)
    assert "error" not in res
    assert res["result"]["A"] == expected_a


def test_groupby_analysis_filter_values(sample_df: pd.DataFrame):
    """Test groupby_analysis with filter_values."""
    res = groupby_analysis(
        sample_df,
        group_column="Store",
        value_column="Weekly_Sales",
        filter_values=["A", "C"],
    )
    assert "error" not in res
    assert set(res["result"].keys()) == {"A", "C"}
    assert "B" not in res["result"]
    assert "filter_applied" in res


def test_groupby_analysis_filter_values_no_match(sample_df: pd.DataFrame):
    """Test filter_values when no rows match."""
    res = groupby_analysis(
        sample_df,
        group_column="Store",
        value_column="Weekly_Sales",
        filter_values=["NonExistentStore"],
    )
    assert "error" in res
    assert "No rows matched" in res["error"]


def test_groupby_analysis_top_n(sample_df: pd.DataFrame):
    """Test top_n with sort_order desc and asc."""
    res_desc = groupby_analysis(
        sample_df,
        group_column="Store",
        value_column="Weekly_Sales",
        top_n=2,
        sort_order="desc",
    )
    assert "error" not in res_desc
    assert len(res_desc["result"]) == 2
    assert list(res_desc["result"].keys()) == ["C", "B"]

    res_asc = groupby_analysis(
        sample_df,
        group_column="Store",
        value_column="Weekly_Sales",
        top_n=2,
        sort_order="asc",
    )
    assert "error" not in res_asc
    assert len(res_asc["result"]) == 2
    assert list(res_asc["result"].keys()) == ["A", "B"]


def test_groupby_analysis_datetime_column(sample_df: pd.DataFrame):
    """Test groupby with datetime group_column."""
    res = groupby_analysis(sample_df, group_column="Date", value_column="Weekly_Sales", agg_function="sum")
    assert "error" not in res
    assert res["group_column_type"] == "datetime"


def test_groupby_analysis_count_on_non_numeric(sample_df: pd.DataFrame):
    """Test count aggregation on non-numeric value column."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="IsHoliday", agg_function="count")
    assert "error" not in res
    assert res["result"]["A"] == 2


def test_groupby_analysis_numeric_agg_on_non_numeric_error(sample_df: pd.DataFrame):
    """Test mean aggregation on non-numeric value column returns error."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Store", agg_function="mean")
    assert "error" in res
    assert "not numeric" in res["error"]


def test_groupby_analysis_invalid_columns(sample_df: pd.DataFrame):
    """Test groupby with non-existent columns."""
    res_grp = groupby_analysis(sample_df, group_column="MissingCol", value_column="Weekly_Sales")
    assert "error" in res_grp
    assert "MissingCol" in res_grp["error"]

    res_val = groupby_analysis(sample_df, group_column="Store", value_column="MissingVal")
    assert "error" in res_val
    assert "MissingVal" in res_val["error"]


def test_groupby_analysis_invalid_agg_function(sample_df: pd.DataFrame):
    """Test groupby with unsupported agg_function."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Weekly_Sales", agg_function="variance")
    assert "error" in res
    assert "not supported" in res["error"]


def test_groupby_analysis_max_groups_capping():
    """Test groupby output truncation when groups exceed MAX_GROUPS_RETURNED."""
    n_groups = MAX_GROUPS_RETURNED + 20
    df = pd.DataFrame({
        "Category": [f"Cat_{i}" for i in range(n_groups)],
        "Value": list(range(n_groups)),
    })
    res = groupby_analysis(df, group_column="Category", value_column="Value")
    assert "error" not in res
    assert len(res["result"]) == MAX_GROUPS_RETURNED
    assert "note" in res
    assert f"Showing the top {MAX_GROUPS_RETURNED}" in res["note"]


def test_groupby_analysis_none_df():
    """Test groupby with df=None."""
    res = groupby_analysis(None, group_column="Store", value_column="Weekly_Sales")
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_groupby_analysis_schema():
    """Test GROUPBY_ANALYSIS_SCHEMA structure."""
    assert GROUPBY_ANALYSIS_SCHEMA["type"] == "function"
    assert GROUPBY_ANALYSIS_SCHEMA["function"]["name"] == "groupby_analysis"


def test_groupby_comparative_ranking_and_best_worst(sample_df: pd.DataFrame):
    """Test best_group, worst_group, and ranking calculations."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Weekly_Sales", agg_function="mean")
    assert res["best_group"] == "C"
    assert res["worst_group"] == "A"
    assert res["ranking"] == ["C", "B", "A"]


def test_groupby_pairwise_comparison_metrics(sample_df: pd.DataFrame):
    """Test absolute and percentage differences when comparing exactly 2 groups."""
    res = groupby_analysis(sample_df, group_column="Store", value_column="Weekly_Sales", agg_function="mean", filter_values=["A", "B"])
    assert "comparison" in res
    comp = res["comparison"]
    assert comp["group_1"] == "A"
    assert comp["group_2"] == "B"
    assert comp["absolute_difference"] == 50.0  # 225.0 vs 175.0
    assert comp["percentage_difference"] is not None
    assert comp["percentage_change"] is not None
