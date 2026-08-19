import pandas as pd
import pytest

from tools.groupby import GROUPBY_ANALYSIS_SCHEMA, groupby_analysis
from tools.period_comparison import PERCENTAGE_CHANGE_SCHEMA, percentage_change
from tools.time_analysis import TIME_ANALYSIS_SCHEMA, time_analysis
from tools.visualization import CREATE_VISUALIZATION_SCHEMA, create_visualization


@pytest.fixture
def numeric_store_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-01",
            "2025-01-01", "2025-01-01", "2025-01-01",
        ]),
        "Store": [20, 4, 14, 20, 4, 14],
        "Weekly_Sales": [100.0, 80.0, 60.0, 120.0, 90.0, 75.0],
    })


@pytest.mark.parametrize("filter_values", [[20, 4, 14], ["20", "4", "14"]])
def test_numeric_and_string_filter_values_work_across_tools(
    numeric_store_df: pd.DataFrame,
    filter_values: list[int] | list[str],
):
    """Numeric categorical IDs and their string representations filter identically."""
    grouped = groupby_analysis(
        numeric_store_df, "Store", "Weekly_Sales", filter_values=filter_values,
    )
    timed = time_analysis(
        numeric_store_df, "Date", "Weekly_Sales", period="month",
        group_column="Store", filter_values=filter_values,
    )
    compared = percentage_change(
        numeric_store_df, "Date", "Weekly_Sales", period="year",
        group_column="Store", filter_values=filter_values,
    )
    charted = create_visualization(
        numeric_store_df, chart_type="bar", x_column="Store", y_column="Weekly_Sales",
        agg_function="sum", filter_values=filter_values,
    )

    assert "error" not in grouped
    assert set(grouped["result"]) == {"20", "4", "14"}
    assert "error" not in timed
    assert timed["filter_applied"]["values"] == filter_values
    assert "error" not in compared
    assert compared["filter_applied"]["values"] == filter_values
    assert "error" not in charted


def test_filter_value_schemas_accept_strings_and_numbers():
    """Groq receives a schema that matches the tools' supported categorical inputs."""
    schemas = [
        GROUPBY_ANALYSIS_SCHEMA,
        TIME_ANALYSIS_SCHEMA,
        PERCENTAGE_CHANGE_SCHEMA,
        CREATE_VISUALIZATION_SCHEMA,
    ]
    for schema in schemas:
        item_types = schema["function"]["parameters"]["properties"]["filter_values"]["items"]["type"]
        assert item_types == ["string", "number"]
