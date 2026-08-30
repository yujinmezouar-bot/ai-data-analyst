import pandas as pd

from agent.agent import MAX_LLM_REQUEST_CHARS, TOOL_FUNCTIONS, TOOL_SCHEMAS, _estimate_request_chars
from autonomous.planner import AnalysisPlanner, TEMPORAL_GROUNDING_GUIDANCE
from tools.contribution_analysis import kpi_contribution_analysis
from tools.dataset_info import build_dataset_profile, format_datasets_context
from tools.date_utils import add_period_column, format_period_label


class _Provider:
    def chat(self, **kwargs):  # pragma: no cover - prompts are inspected directly
        raise AssertionError("LLM should not be called")


def _period_profile(dates):
    dataframe = pd.DataFrame({"Date": pd.to_datetime(dates)})
    return build_dataset_profile(dataframe)["date_column_details"]["Date"]["period_profile"]


def test_period_profile_has_range_populated_years_and_counts():
    profile = _period_profile(["2021-01-03", "2023-04-10", "2025-08-19"])

    assert profile["min_date"] == "2021-01-03"
    assert profile["max_date"] == "2025-08-19"
    assert profile["years"] == ["2021", "2023", "2025"]
    assert profile["period_counts"]["years"] == 3


def test_long_history_keeps_latest_eight_years_and_total_count():
    profile = _period_profile([f"{year}-01-01" for year in range(2000, 2026)])

    assert profile["years"] == [str(year) for year in range(2018, 2026)]
    assert profile["period_counts"]["years"] == 26


def test_recent_month_quarter_and_week_labels_are_bounded_chronological_and_canonical():
    dates = pd.date_range("2022-01-05", "2025-03-19", freq="MS") + pd.Timedelta(days=4)
    profile = _period_profile(dates)

    assert len(profile["recent_months"]) == 12
    assert profile["recent_months"] == sorted(profile["recent_months"])
    assert len(profile["recent_quarters"]) == 8
    assert profile["recent_quarters"] == sorted(profile["recent_quarters"])
    assert len(profile["recent_weeks"]) == 8

    source = pd.Series(pd.to_datetime(dates))
    expected_weeks = [
        format_period_label(pd.Timestamp(bucket), "week")
        for bucket in add_period_column(source, "week").drop_duplicates().sort_values().iloc[-8:]
    ]
    assert profile["recent_weeks"] == expected_weeks
    assert all(label.startswith("Week of ") for label in profile["recent_weeks"])


def test_empty_native_datetime_column_has_no_period_profile():
    dataframe = pd.DataFrame({"Date": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]")})
    profile = build_dataset_profile(dataframe)

    assert "Date" in profile["datetime_columns"]
    assert "period_profile" not in profile["date_column_details"]["Date"]


def test_only_three_datetime_columns_receive_detail_with_deterministic_priority():
    dataframe = pd.DataFrame({
        "created_at": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "value_when": pd.to_datetime(["2024-01-02", "2024-02-02"]),
        "sale_date": pd.to_datetime(["2024-01-03", "2024-02-03"]),
        "ship_date": pd.to_datetime(["2024-01-04", "2024-02-04"]),
    })
    profile = build_dataset_profile(dataframe)
    detailed = [
        column for column in profile["datetime_columns"]
        if "period_profile" in profile["date_column_details"][column]
    ]

    assert detailed == ["created_at", "sale_date", "ship_date"]
    assert profile["temporal_profile_omitted_count"] == 1


def test_multiple_datasets_render_independent_temporal_profiles():
    context = format_datasets_context({
        "older": pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2021-01-01"])}),
        "newer": pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2025-01-01"])}),
    })

    assert "[Dataset: older]" in context and "range 2020-01-01 -> 2021-01-01" in context
    assert "[Dataset: newer]" in context and "range 2024-01-01 -> 2025-01-01" in context


def test_planner_and_reviewer_share_bounded_temporal_context_and_guidance():
    datasets = {"sales": pd.DataFrame({
        "Date": pd.date_range("2015-01-01", periods=132, freq="MS"),
        "Sales": range(132),
    })}
    context = format_datasets_context(datasets)
    planner = AnalysisPlanner(_Provider(), tools_registry=TOOL_FUNCTIONS)
    planner_messages = planner._build_prompt("Analyze recent growth", {
        "datasets": ["sales"], "dataset_context": context, "tool_schemas": TOOL_SCHEMAS,
    })
    review_messages = planner.build_review_prompt(
        "Analyze recent growth", [], context, TOOL_SCHEMAS
    )

    assert "recent months" in planner_messages[0]["content"]
    assert "recent months" in review_messages[1]["content"]
    assert TEMPORAL_GROUNDING_GUIDANCE in planner_messages[0]["content"]
    assert TEMPORAL_GROUNDING_GUIDANCE in review_messages[0]["content"]
    assert _estimate_request_chars(planner_messages) <= MAX_LLM_REQUEST_CHARS


def test_contribution_accepts_canonical_year_labels_from_period_profile():
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01", "2025-01-01"]),
        "Product": ["A", "B", "A", "B"],
        "Sales": [100.0, 80.0, 70.0, 60.0],
    })
    periods = build_dataset_profile(dataframe)["date_column_details"]["Date"]["period_profile"]["years"]
    result = kpi_contribution_analysis(
        dataframe, date_column="Date", metric_column="Sales", group_column="Product",
        period="year", period_a=periods[-2], period_b=periods[-1],
    )

    assert "error" not in result
    assert result["period_a"] == "2024"
    assert result["period_b"] == "2025"
