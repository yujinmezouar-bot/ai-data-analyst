import pandas as pd
import pytest

from tools.contribution_analysis import (
    KPI_CONTRIBUTION_SCHEMA,
    MAX_CONTRIBUTORS_RETURNED,
    kpi_contribution_analysis,
)


def contribution_df(rows):
    return pd.DataFrame(rows, columns=["Date", "Product", "Region", "Sales"]).assign(
        Date=lambda frame: pd.to_datetime(frame["Date"])
    )


def analyze(df, **kwargs):
    arguments = {
        "date_column": "Date", "metric_column": "Sales", "group_column": "Product",
        "period_a": "2024", "period_b": "2025",
    }
    arguments.update(kwargs)
    return kpi_contribution_analysis(df, **arguments)


def test_positive_growth_contributions_sum_to_100():
    df = contribution_df([
        ("2024-01-01", "A", "North", 100), ("2025-01-01", "A", "North", 180),
        ("2024-01-01", "B", "South", 100), ("2025-01-01", "B", "South", 120),
    ])
    result = analyze(df)

    assert result["overall"] == {
        "value_a": 200.0, "value_b": 300.0, "absolute_change": 100.0,
        "percentage_change": 50.0, "direction": "increase",
    }
    assert result["top_driver"] == "A"
    assert sum(item["contribution_to_total_change_percentage"] for item in result["contributors"]) == 100.0


def test_decline_has_signed_driver_and_offset_contributions():
    df = contribution_df([
        ("2024-01-01", "A", "North", 200), ("2025-01-01", "A", "North", 100),
        ("2024-01-01", "B", "South", 100), ("2025-01-01", "B", "South", 120),
    ])
    result = analyze(df)
    items = {item["group"]: item for item in result["contributors"]}

    assert result["overall"]["absolute_change"] == -80.0
    assert items["A"]["contribution_to_total_change_percentage"] == 125.0
    assert items["A"]["effect"] == "reinforces_decrease"
    assert items["B"]["contribution_to_total_change_percentage"] == -25.0
    assert items["B"]["effect"] == "offsets_decrease"
    assert result["largest_offset"] == "B"


def test_new_and_disappeared_groups_have_correct_lifecycle():
    df = contribution_df([
        ("2024-01-01", "Old", "North", 50),
        ("2025-01-01", "New", "North", 80),
    ])
    result = analyze(df)
    items = {item["group"]: item for item in result["contributors"]}

    assert items["New"]["group_status"] == "new"
    assert items["New"]["value_a"] == 0.0
    assert items["New"]["percentage_change"] is None
    assert items["Old"]["group_status"] == "disappeared"
    assert items["Old"]["percentage_change"] == -100.0


def test_zero_total_change_has_null_contributions_and_ranks_movements():
    df = contribution_df([
        ("2024-01-01", "A", "North", 100), ("2025-01-01", "A", "North", 50),
        ("2024-01-01", "B", "South", 50), ("2025-01-01", "B", "South", 100),
    ])
    result = analyze(df)

    assert result["overall"]["direction"] == "unchanged"
    assert all(item["contribution_to_total_change_percentage"] is None for item in result["contributors"])
    assert all(item["effect"] == "net_zero_movement" for item in result["contributors"])
    assert "netted to zero" in result["note"]


def test_independent_filter_then_product_grouping():
    df = contribution_df([
        ("2024-01-01", "A", "North", 100), ("2025-01-01", "A", "North", 50),
        ("2024-01-01", "B", "South", 1000), ("2025-01-01", "B", "South", 2000),
    ])
    result = analyze(df, filter_column="Region", filter_values=["North"])

    assert result["overall"]["absolute_change"] == -50.0
    assert [item["group"] for item in result["contributors"]] == ["A"]
    assert result["filter_applied"] == {"column": "Region", "values": ["North"]}


def test_missing_rows_are_excluded_and_reported():
    df = contribution_df([
        (None, "A", "North", 10),
        ("2024-01-01", None, "North", 20),
        ("2024-01-01", "A", "North", None),
        ("2024-01-01", "A", "North", 100),
        ("2025-01-01", "A", "North", 120),
    ])
    result = analyze(df)

    assert result["excluded_rows"] == {
        "missing_date": 1, "missing_group": 1, "missing_metric": 1, "total_excluded": 3,
    }


def test_bounded_output_keeps_complete_overall_totals_and_offsets():
    rows = []
    expected_a = expected_b = 0
    for index in range(30):
        value_a = 100 + index
        value_b = value_a + index - 5
        expected_a += value_a
        expected_b += value_b
        rows.extend([
            ("2024-01-01", f"P{index:02}", "All", value_a),
            ("2025-01-01", f"P{index:02}", "All", value_b),
        ])
    result = analyze(contribution_df(rows), top_n=10)

    assert result["overall"]["value_a"] == expected_a
    assert result["overall"]["value_b"] == expected_b
    assert result["groups_analyzed"] == 30
    assert result["groups_returned"] == 10
    assert result["truncated"] is True
    assert any(item["effect"].startswith("offsets_") for item in result["contributors"])


@pytest.mark.parametrize("kwargs, error", [
    ({"date_column": "Missing"}, "not found"),
    ({"metric_column": "Missing"}, "not found"),
    ({"group_column": "Missing"}, "not found"),
    ({"agg_function": "mean"}, "not supported"),
    ({"period_b": "2026"}, "do not exist"),
    ({"period_a": "2025", "period_b": "2024"}, "must precede"),
    ({"period_a": "2024", "period_b": "2024"}, "must be different"),
    ({"filter_column": "Region"}, "supplied together"),
    ({"filter_values": ["North"]}, "supplied together"),
    ({"filter_column": "Region", "filter_values": []}, "non-empty"),
    ({"top_n": 0}, "positive integer"),
    ({"top_n": MAX_CONTRIBUTORS_RETURNED + 1}, "capped"),
])
def test_validation_errors(kwargs, error):
    df = contribution_df([
        ("2024-01-01", "A", "North", 100),
        ("2025-01-01", "A", "North", 120),
    ])
    result = analyze(df, **kwargs)
    assert error in result["error"]


def test_sum_rejects_nonnumeric_metric_but_count_is_supported():
    df = contribution_df([
        ("2024-01-01", "A", "North", "one"),
        ("2025-01-01", "A", "North", "two"),
    ])
    assert "numeric" in analyze(df)["error"]
    assert analyze(df, agg_function="count")["overall"]["absolute_change"] == 0.0


def test_schema_contract():
    function = KPI_CONTRIBUTION_SCHEMA["function"]
    assert function["name"] == "kpi_contribution_analysis"
    assert function["parameters"]["required"] == [
        "date_column", "metric_column", "group_column", "period_a", "period_b",
    ]


def test_rejects_non_datetime_date_column_and_invalid_period():
    df = contribution_df([
        ("2024-01-01", "A", "North", 100),
        ("2025-01-01", "A", "North", 120),
    ])
    df["Date"] = df["Date"].astype(str)
    assert "recognized date" in analyze(df)["error"]
    df["Date"] = pd.to_datetime(df["Date"])
    assert "not supported" in analyze(df, period="decade")["error"]


def test_rejects_invalid_or_excessive_filter_values():
    df = contribution_df([
        ("2024-01-01", "A", "North", 100),
        ("2025-01-01", "A", "North", 120),
    ])
    assert "strings or numbers" in analyze(
        df, filter_column="Region", filter_values=[{"region": "North"}]
    )["error"]
    assert "capped" in analyze(
        df, filter_column="Region", filter_values=list(range(51))
    )["error"]
    assert "No rows remain" in analyze(
        df, filter_column="Region", filter_values=["Missing"]
    )["error"]
