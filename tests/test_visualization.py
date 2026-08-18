import pandas as pd
import plotly.graph_objects as go
import pytest

from tools.visualization import create_visualization, CREATE_VISUALIZATION_SCHEMA


def test_create_visualization_bar_aggregated(sample_df: pd.DataFrame):
    """Test aggregated bar chart."""
    res = create_visualization(
        sample_df,
        chart_type="bar",
        x_column="Store",
        y_column="Weekly_Sales",
        agg_function="mean",
    )
    assert "error" not in res
    assert res["chart_type"] == "bar"
    assert res["aggregated"] is True
    assert isinstance(res["figure"], go.Figure)
    assert res["rows_used"] == 3


def test_create_visualization_line_time_aggregated(sample_df: pd.DataFrame):
    """Test line chart aggregated by time period."""
    res = create_visualization(
        sample_df,
        chart_type="line",
        x_column="Date",
        y_column="Weekly_Sales",
        agg_function="sum",
        period="month",
    )
    assert "error" not in res
    assert res["chart_type"] == "line"
    assert res["aggregated"] is True
    assert res["period"] == "month"
    assert isinstance(res["figure"], go.Figure)


def test_create_visualization_scatter(sample_df: pd.DataFrame):
    """Test raw scatter plot."""
    res = create_visualization(
        sample_df,
        chart_type="scatter",
        x_column="Temperature",
        y_column="Weekly_Sales",
    )
    assert "error" not in res
    assert res["chart_type"] == "scatter"
    assert res["aggregated"] is False
    assert isinstance(res["figure"], go.Figure)


def test_create_visualization_histogram(sample_df: pd.DataFrame):
    """Test single-column histogram without y_column."""
    res = create_visualization(
        sample_df,
        chart_type="histogram",
        x_column="Weekly_Sales",
    )
    assert "error" not in res
    assert res["chart_type"] == "histogram"
    assert isinstance(res["figure"], go.Figure)


def test_create_visualization_box(sample_df: pd.DataFrame):
    """Test box plot with x and y columns."""
    res = create_visualization(
        sample_df,
        chart_type="box",
        x_column="Store",
        y_column="Weekly_Sales",
    )
    assert "error" not in res
    assert res["chart_type"] == "box"
    assert isinstance(res["figure"], go.Figure)


def test_create_visualization_filter_values(sample_df: pd.DataFrame):
    """Test visualization with category filtering."""
    res = create_visualization(
        sample_df,
        chart_type="bar",
        x_column="Store",
        y_column="Weekly_Sales",
        agg_function="mean",
        filter_values=["A", "B"],
    )
    assert "error" not in res
    assert res["rows_used"] == 2


def test_create_visualization_filter_values_no_match(sample_df: pd.DataFrame):
    """Test visualization filter_values with no matching data."""
    res = create_visualization(
        sample_df,
        chart_type="bar",
        x_column="Store",
        y_column="Weekly_Sales",
        filter_values=["NonExistent"],
    )
    assert "error" in res
    assert "No rows matched" in res["error"]


def test_create_visualization_top_n(sample_df: pd.DataFrame):
    """Test top_n limiting on aggregated bar chart."""
    res = create_visualization(
        sample_df,
        chart_type="bar",
        x_column="Store",
        y_column="Weekly_Sales",
        agg_function="mean",
        top_n=2,
        sort_order="desc",
    )
    assert "error" not in res
    assert res["rows_used"] == 2


@pytest.mark.parametrize("chart_type", ["bar", "line", "scatter", "box"])
def test_create_visualization_missing_y_column(sample_df: pd.DataFrame, chart_type: str):
    """Test that chart types requiring y_column return an error when y_column is None."""
    res = create_visualization(sample_df, chart_type=chart_type, x_column="Store")
    assert "error" in res
    assert "requires a y_column" in res["error"]


def test_create_visualization_histogram_with_agg_error(sample_df: pd.DataFrame):
    """Test histogram with agg_function returns an error."""
    res = create_visualization(
        sample_df,
        chart_type="histogram",
        x_column="Weekly_Sales",
        agg_function="mean",
    )
    assert "error" in res
    assert "Aggregation is not applicable to histogram" in res["error"]


def test_create_visualization_invalid_chart_type(sample_df: pd.DataFrame):
    """Test invalid chart type."""
    res = create_visualization(sample_df, chart_type="pie", x_column="Store")
    assert "error" in res
    assert "not supported" in res["error"]


def test_create_visualization_invalid_columns(sample_df: pd.DataFrame):
    """Test non-existent x and y columns."""
    res_x = create_visualization(sample_df, chart_type="bar", x_column="BadX", y_column="Weekly_Sales")
    assert "error" in res_x
    assert "BadX" in res_x["error"]

    res_y = create_visualization(sample_df, chart_type="bar", x_column="Store", y_column="BadY")
    assert "error" in res_y
    assert "BadY" in res_y["error"]


def test_create_visualization_invalid_period(sample_df: pd.DataFrame):
    """Test invalid period for datetime aggregation."""
    res = create_visualization(
        sample_df,
        chart_type="line",
        x_column="Date",
        y_column="Weekly_Sales",
        agg_function="sum",
        period="decade",
    )
    assert "error" in res
    assert "Time period 'decade' is not supported" in res["error"]


def test_create_visualization_none_df():
    """Test visualization with df=None."""
    res = create_visualization(None, chart_type="bar", x_column="Store", y_column="Weekly_Sales")
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_create_visualization_schema():
    """Test CREATE_VISUALIZATION_SCHEMA required properties."""
    schema = CREATE_VISUALIZATION_SCHEMA["function"]
    assert schema["name"] == "create_visualization"
    assert schema["parameters"]["required"] == ["chart_type", "x_column"]
