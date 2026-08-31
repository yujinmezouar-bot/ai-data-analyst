from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go

from autonomous.results import Finding
from reports.report_builder import AnalysisReport, build_analysis_report, render_markdown


def _datasets():
    return {"sales": pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
        "Sales": [100.0, None],
        "Product": ["A", "B"],
    })}


def _finding(identifier, tool, result, step=None):
    return Finding(
        id=identifier,
        step_id=step or f"step_{identifier}",
        tool_name=tool,
        datasets=["sales"],
        result=result,
        metadata={"plan_id": "plan_1", "objective": "Investigate sales"},
        provenance={"plan_id": "plan_1", "category": "analysis_tool"},
    )


def _contribution_result():
    return {
        "date_column": "Date", "metric_column": "Sales", "group_column": "Product",
        "period": "year", "period_a": "2024", "period_b": "2025",
        "overall": {
            "value_a": 1000.0, "value_b": 800.0, "absolute_change": -200.0,
            "percentage_change": -20.0, "direction": "decrease",
        },
        "contributors": [
            {"group": "A", "absolute_change": -250.0,
             "contribution_to_total_change_percentage": 125.0, "effect": "reinforces_decrease"},
            {"group": "B", "absolute_change": 50.0,
             "contribution_to_total_change_percentage": -25.0, "effect": "offsets_decrease"},
        ],
        "filter_applied": {"column": "Region", "values": ["North"]},
        "excluded_rows": {"total_excluded": 2},
        "truncated": False,
    }


def test_autonomous_report_has_stable_structure_multiple_findings_and_provenance():
    result = {
        "answer": "Sales declined and Product A was the largest mathematical contributor.",
        "figure": None,
        "findings": [
            _finding("finding_1", "statistics", {"column": "Sales", "mean": 900.0}),
            _finding("finding_2", "kpi_contribution_analysis", _contribution_result()),
        ],
        "trace": [{"step": "autonomous_plan", "plan": {"objective": "Explain the decline"}}],
    }
    report = build_analysis_report("Why did sales decline?", result, _datasets())
    markdown = render_markdown(report)

    assert isinstance(report, AnalysisReport)
    assert [item["evidence_id"] for item in report.findings] == ["F1", "F2"]
    assert "F1 (finding_1) — statistics — sales" in markdown
    assert "F2 (finding_2) — kpi_contribution_analysis — sales" in markdown
    headings = [
        "## Analysis Objective", "## Executive Summary", "## Dataset Overview",
        "## Key Findings", "## KPI / Change Evidence", "## Drivers and Offsets",
        "## Limitations", "## Methodology", "## Provenance / Evidence Appendix",
    ]
    positions = [markdown.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_contribution_report_preserves_totals_signs_filters_and_noncausal_wording():
    result = {
        "answer": "The observed decline was decomposed.", "figure": None,
        "findings": [_finding("finding_1", "kpi_contribution_analysis", _contribution_result())],
        "trace": [],
    }
    markdown = render_markdown(build_analysis_report("Explain decline", result, _datasets()))

    for value in ("2024", "2025", "1000", "800", "-200", "-20%", "Region", "North"):
        assert value in markdown
    assert "mathematical contributor to the observed decrease" in markdown
    assert "offset part of the observed decrease" in markdown
    assert "do not establish causation" in markdown
    assert "caused" not in markdown.lower()


def test_correlation_report_is_explicitly_noncausal():
    correlation = {
        "strongest_positive_pair": {"column_1": "Price", "column_2": "Quantity", "correlation": 0.75}
    }
    result = {
        "answer": "Price and quantity are associated.", "figure": None,
        "findings": [_finding("finding_1", "correlation_analysis", correlation)], "trace": [],
    }
    markdown = render_markdown(build_analysis_report("Find correlation", result, _datasets()))

    assert "association" in markdown.lower()
    assert "do not establish causation" in markdown


def test_dataset_overview_uses_profile_shape_missingness_and_date_coverage():
    report = build_analysis_report("Describe data", {"answer": "Overview.", "trace": [], "figure": None}, _datasets())
    markdown = render_markdown(report)

    assert "2 rows × 3 columns" in markdown
    assert "Date coverage for `Date`: 2024-01-01 → 2025-01-01" in markdown
    assert "sales.Sales" in markdown and "1 missing" in markdown


def test_truncated_failed_and_adaptive_trace_artifacts_create_limitations():
    result = {
        "answer": "Partial answer.", "figure": None,
        "evidence": [{"tool_name": "groupby_analysis", "result": {"note": "Result truncated."}}],
        "trace": [
            {"step": "tool_call", "tool": "statistics", "success": False},
            {"step": "adaptive_review", "status": "invalid"},
            {"step": "adaptive_stop", "reason": "global_step_limit"},
        ],
    }
    limitations = build_analysis_report("Analyze", result, _datasets()).limitations

    assert any("truncated" in item for item in limitations)
    assert any("statistics failed" in item for item in limitations)
    assert any("Adaptive review" in item for item in limitations)
    assert any("global_step_limit" in item for item in limitations)


def test_reactive_evidence_is_reported_and_direct_answer_has_limited_evidence():
    reactive = {
        "answer": "Average sales were 225.", "figure": None,
        "evidence": [{"tool_name": "statistics", "result": {"column": "Sales", "mean": 225.0}}],
        "trace": [{"step": "tool_call", "tool": "statistics", "arguments": {"dataset_name": "sales"}, "success": True}],
    }
    report = build_analysis_report("Average sales?", reactive, _datasets())
    assert report.findings[0]["result"]["mean"] == 225.0
    assert not any("Structured analytical evidence was unavailable" in item for item in report.limitations)

    direct = build_analysis_report(
        "Hello", {"answer": "Hello.", "figure": None, "evidence": [], "trace": []}, _datasets()
    )
    assert any("Structured analytical evidence was unavailable" in item for item in direct.limitations)


def test_groupby_report_does_not_label_the_only_returned_top_group_as_worst():
    result = {
        "answer": "Type A has the highest total sales.",
        "figure": None,
        "evidence": [{"tool_name": "groupby_analysis", "result": {
            "group_column": "Store_Type",
            "value_column": "Weekly_Sales",
            "agg_function": "sum",
            "result": {"A": 2520470000.0},
            "best_group": "A",
            "worst_group": "A",
            "ranking": ["A"],
        }}],
        "trace": [],
    }

    markdown = render_markdown(build_analysis_report("Highest sales by type?", result, _datasets()))

    assert "Grouped analysis: best group=A." in markdown
    assert "worst group=A" not in markdown


def test_figure_metadata_is_bounded_and_plotly_object_is_not_rendered():
    figure = go.Figure(data=go.Bar(x=["A"], y=[1]))
    figure.update_layout(title="Sales by product")
    result = {
        "answer": "Chart created.", "figure": figure,
        "evidence": [{"tool_name": "create_visualization", "result": {
            "chart_type": "bar", "description": "Total sales by product."
        }}],
        "trace": [],
    }
    report = build_analysis_report("Chart sales", result, _datasets())
    markdown = render_markdown(report)

    assert "## Visualizations" in markdown and "Sales by product" in markdown
    assert "not embedded" in markdown
    assert "Figure({" not in markdown and "plotly.graph_objs" not in markdown

    without_figure = render_markdown(build_analysis_report(
        "No chart", {"answer": "Done.", "figure": None, "trace": []}, _datasets()
    ))
    assert "## Visualizations" not in without_figure


def test_repeated_report_builds_are_identical_and_do_not_invoke_execution():
    result = {
        "answer": "Average is 10.", "figure": None,
        "evidence": [{"tool_name": "statistics", "result": {"column": "Sales", "mean": 10.0}}],
        "trace": [],
    }
    with patch("agent.agent.Agent.run", side_effect=AssertionError("must not execute")):
        first = render_markdown(build_analysis_report("Average?", result, _datasets()))
        second = render_markdown(build_analysis_report("Average?", result, _datasets()))

    assert first == second


def test_autonomous_to_reactive_fallback_artifacts_build_reactive_report():
    fallback = {
        "answer": "Reactive fallback answer.", "figure": None,
        "evidence": [{"tool_name": "percentage_change", "result": {
            "absolute_change": -5.0, "percentage_change": -10.0,
        }}],
        "trace": [{"step": "routing", "decision": "autonomous"}, {"step": "final_answer", "tool_used": True}],
    }
    report = build_analysis_report("What changed?", fallback, _datasets())

    assert not report.findings[0]["finding_id"]
    assert "percentage_change" in report.methodology
