import json
from types import SimpleNamespace

import pandas as pd

from agent.agent import (
    FINAL_EXPLANATION_SYSTEM_PROMPT,
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    Agent,
    _summarise_tool_results,
)
from autonomous.executor import Executor
from autonomous.plan import AnalysisPlan, PlanStep


def message(content):
    return SimpleNamespace(content=content, tool_calls=None)


def contribution_df():
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01", "2025-01-01"]),
        "Product": ["A", "B", "A", "B"],
        "Sales": [200.0, 100.0, 100.0, 120.0],
    })


def contribution_kwargs():
    return {
        "dataset_name": "sales",
        "date_column": "Date",
        "metric_column": "Sales",
        "group_column": "Product",
        "period_a": "2024",
        "period_b": "2025",
    }


def test_contribution_tool_works_through_reactive_agent_routing():
    agent = Agent()
    agent.active_datasets = {"sales": contribution_df()}
    agent.derived_datasets = {}

    result = agent._execute_tool("kpi_contribution_analysis", contribution_kwargs(), contribution_df())

    assert result["overall"]["absolute_change"] == -80.0
    assert result["top_driver"] == "A"


def test_contribution_plan_passes_preflight_and_executes():
    plan = AnalysisPlan(
        id="contribution",
        objective="Find mathematical drivers",
        datasets=["sales"],
        steps=[PlanStep(
            id="drivers", tool_name="kpi_contribution_analysis",
            kwargs=contribution_kwargs(), read_only=True,
        )],
    )
    findings = Executor(TOOL_FUNCTIONS, tool_schemas=TOOL_SCHEMAS).execute(
        plan, {"sales": contribution_df()}
    )

    assert findings.all()[0].result["top_driver"] == "A"


def test_initial_autonomous_plan_executes_contribution_tool(monkeypatch):
    plan = {
        "id": "initial_contribution", "objective": "Find drivers", "datasets": ["sales"],
        "steps": [{
            "id": "drivers", "tool_name": "kpi_contribution_analysis",
            "kwargs": contribution_kwargs(), "read_only": True,
        }],
    }
    responses = iter([
        message(json.dumps(plan)),
        message(json.dumps({"status": "complete", "reason": "Drivers are quantified."})),
        message("Product A accounted for the largest mathematical contribution to the decline."),
    ])
    agent = Agent()
    agent.llm = SimpleNamespace(chat=lambda *args, **kwargs: next(responses))

    result = agent.run("Which products drove the decline?", datasets={"sales": contribution_df()}, autonomous=True)

    assert result["answer"].startswith("Product A accounted")
    assert result["findings"][0].tool_name == "kpi_contribution_analysis"


def test_adaptive_reviewer_can_request_contribution_follow_up():
    initial = {
        "id": "initial", "objective": "Inspect sales", "datasets": ["sales"],
        "steps": [{
            "id": "info", "tool_name": "dataset_info",
            "kwargs": {"dataset_name": "sales"}, "read_only": True,
        }],
    }
    adaptive_step = {
        "id": "adaptive_drivers", "tool_name": "kpi_contribution_analysis",
        "kwargs": contribution_kwargs(), "read_only": True,
    }
    responses = iter([
        message(json.dumps(initial)),
        message(json.dumps({
            "status": "follow_up", "reason": "The initial profile does not quantify drivers.",
            "steps": [adaptive_step],
        })),
        message("Product A was the largest mathematical driver; Product B offset part of the decline."),
    ])
    agent = Agent()
    agent.llm = SimpleNamespace(chat=lambda *args, **kwargs: next(responses))

    result = agent.run("Which products drove the decline?", datasets={"sales": contribution_df()}, autonomous=True)

    assert [finding.tool_name for finding in result["findings"]] == [
        "dataset_info", "kpi_contribution_analysis",
    ]
    assert result["findings"][-1].provenance["adaptive_round"] == 1
    assert any(item.get("step") == "adaptive_execution" and item["status"] == "completed" for item in result["trace"])


def test_contribution_synthesis_grounding_distinguishes_driver_and_offset():
    result = {
        "overall": {"value_a": 300.0, "value_b": 220.0, "absolute_change": -80.0,
                    "percentage_change": -26.67, "direction": "decrease"},
        "contributors": [
            {"group": "A", "absolute_change": -100.0,
             "contribution_to_total_change_percentage": 125.0, "effect": "reinforces_decrease"},
            {"group": "B", "absolute_change": 20.0,
             "contribution_to_total_change_percentage": -25.0, "effect": "offsets_decrease"},
        ],
    }
    summary = _summarise_tool_results([
        {"tool_name": "kpi_contribution_analysis", "result": result}
    ], has_figure=False)

    assert "mathematical driver" in summary
    assert "Largest returned offset: B" in summary
    assert "not causation" in FINAL_EXPLANATION_SYSTEM_PROMPT
