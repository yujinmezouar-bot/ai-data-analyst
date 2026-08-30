import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent.agent import MAX_LLM_REQUEST_CHARS, TOOL_FUNCTIONS, TOOL_SCHEMAS, _estimate_request_chars
from autonomous.planner import AnalysisPlanner, PlannerError
from autonomous.plan import AnalysisPlan
from tools.dataset_info import format_datasets_context
from tools.relationship_discovery import build_schema_graph_summary


class FakeProvider:
    def __init__(self, content: str):
        self._content = content
        self.chat_called = False

    def chat(self, messages=None, tools=None, tool_choice=None):
        self.chat_called = True
        return SimpleNamespace(content=self._content)


def make_simple_plan_dict():
    return {
        "id": "plan_x",
        "objective": "Summarize sales",
        "datasets": ["sales"],
        "steps": [
            {"id": "step1", "tool_name": "statistics", "kwargs": {}, "read_only": True, "outputs_expected": ["summary"]}
        ],
        "constraints": {},
        "expected_outputs": ["summary"],
    }


def test_planner_parses_valid_json():
    plan_dict = make_simple_plan_dict()
    provider = FakeProvider(json.dumps(plan_dict))
    planner = AnalysisPlanner(provider, tools_registry={"statistics": {}}, max_steps=5, validate_tools=True)

    plan = planner.plan("Analyze sales", context={"datasets": ["sales"]})
    assert isinstance(plan, AnalysisPlan)
    assert plan.id == "plan_x"
    assert provider.chat_called


def test_planner_prompt_defines_expected_outputs_arrays_and_complete_json_example():
    planner = AnalysisPlanner(FakeProvider("{}"), tools_registry={"statistics": {}})
    prompt = planner._build_prompt("Analyze sales", {"datasets": ["sales"]})[0]["content"]

    assert "Top-level expected_outputs and every step-level outputs_expected MUST be JSON arrays" in prompt
    assert "of strings; use [] when none" in prompt
    marker = "Complete JSON shape example (replace d/t with listed names): "
    example = prompt.split(marker, 1)[1].splitlines()[0]
    parsed = json.loads(example)
    assert set(parsed) == {"id", "objective", "datasets", "steps", "constraints", "expected_outputs"}
    assert isinstance(parsed["expected_outputs"], list)
    assert isinstance(parsed["steps"][0]["outputs_expected"], list)


def test_planner_preserves_valid_expected_outputs_list():
    payload = make_simple_plan_dict()
    payload["expected_outputs"] = ["ranked products", "supporting evidence"]
    plan = AnalysisPlanner(FakeProvider(json.dumps(payload))).plan(
        "Analyze sales", context={"datasets": ["sales"]}
    )

    assert plan.expected_outputs == ["ranked products", "supporting evidence"]


def test_planner_normalizes_non_empty_scalar_expected_outputs():
    payload = make_simple_plan_dict()
    payload["expected_outputs"] = "ranked product declines"
    plan = AnalysisPlanner(FakeProvider(json.dumps(payload))).plan(
        "Analyze sales", context={"datasets": ["sales"]}
    )

    assert plan.expected_outputs == ["ranked product declines"]


@pytest.mark.parametrize("invalid", [{"type": "summary"}, 7])
def test_planner_rejects_non_string_scalar_expected_outputs(invalid):
    payload = make_simple_plan_dict()
    payload["expected_outputs"] = invalid
    with pytest.raises(PlannerError, match="expected_outputs.*must be a list"):
        AnalysisPlanner(FakeProvider(json.dumps(payload))).plan(
            "Analyze sales", context={"datasets": ["sales"]}
        )


@pytest.mark.parametrize("value", [None, ""])
def test_planner_preserves_falsy_expected_outputs_behavior(value):
    payload = make_simple_plan_dict()
    payload["expected_outputs"] = value
    plan = AnalysisPlanner(FakeProvider(json.dumps(payload))).plan(
        "Analyze sales", context={"datasets": ["sales"]}
    )

    assert plan.expected_outputs == []


def test_planner_preserves_missing_expected_outputs_behavior():
    payload = make_simple_plan_dict()
    payload.pop("expected_outputs")
    plan = AnalysisPlanner(FakeProvider(json.dumps(payload))).plan(
        "Analyze sales", context={"datasets": ["sales"]}
    )

    assert plan.expected_outputs == []


def test_planner_rejects_malformed_json():
    provider = FakeProvider("not a json")
    planner = AnalysisPlanner(provider)
    with pytest.raises(PlannerError):
        planner.plan("x", context={})


def test_planner_rejects_unknown_tool_when_validating():
    plan = make_simple_plan_dict()
    # change tool to something unknown
    plan["steps"][0]["tool_name"] = "unknown_tool"
    provider = FakeProvider(json.dumps(plan))
    planner = AnalysisPlanner(provider, tools_registry={"statistics": {}}, validate_tools=True)
    with pytest.raises(PlannerError):
        planner.plan("x", context={"datasets": ["sales"]})


def test_planner_respects_max_steps():
    plan = make_simple_plan_dict()
    plan["steps"] = [{"id": f"s{i}", "tool_name": "statistics"} for i in range(10)]
    provider = FakeProvider(json.dumps(plan))
    planner = AnalysisPlanner(provider, max_steps=5)
    with pytest.raises(PlannerError):
        planner.plan("x", context={})


def test_planner_prompt_contains_bounded_dataset_and_tool_grounding():
    datasets = {
        "sales": pd.DataFrame({
            "sale_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "region": ["North", "South"],
            "sales": [100.0, 120.0],
        })
    }
    planner = AnalysisPlanner(FakeProvider("{}"), tools_registry=TOOL_FUNCTIONS)
    messages = planner._build_prompt("Analyze sales", {
        "datasets": ["sales"],
        "dataset_context": format_datasets_context(datasets),
        "tool_schemas": TOOL_SCHEMAS,
    })
    prompt = messages[0]["content"]

    assert "[Dataset: sales]" in prompt
    assert "sales" in prompt and "Numeric columns" in prompt
    assert "sales: float64" in prompt
    assert "sale_date" in prompt and "Datetime columns" in prompt
    assert "region" in prompt and "North" in prompt
    assert "calculation BY, PER, FOR EACH, or ACROSS categories" in prompt
    assert '"required":["group_column","value_column"]' in prompt
    assert '"type":"string"' in prompt
    assert '"enum":["count","max","mean"' in prompt


def test_planner_prompt_is_bounded_for_wide_high_cardinality_data():
    dataframe = pd.DataFrame({
        f"category_{index}": [f"value_{index}_{row}" for row in range(20)]
        for index in range(150)
    })
    datasets = {"wide": dataframe, "other": dataframe.iloc[:, :2].copy()}
    planner = AnalysisPlanner(FakeProvider("{}"), tools_registry=TOOL_FUNCTIONS)
    messages = planner._build_prompt("Compare the datasets", {
        "datasets": list(datasets),
        "dataset_context": format_datasets_context(datasets),
        "relationship_context": build_schema_graph_summary(datasets),
        "tool_schemas": TOOL_SCHEMAS,
    })

    assert _estimate_request_chars(messages) <= MAX_LLM_REQUEST_CHARS


@pytest.mark.parametrize("payload", [
    {"status": "unknown", "reason": "x"},
    {"status": "complete", "reason": ""},
    {"status": "complete", "reason": "done", "extra": True},
    {"status": "complete", "reason": "done", "steps": [{"id": "s", "tool_name": "statistics"}]},
    {"status": "follow_up", "reason": "more evidence"},
    {"status": "follow_up", "reason": "more evidence", "steps": []},
    {"status": "follow_up", "reason": "more evidence", "steps": None},
    {"status": "follow_up", "reason": "more evidence", "steps": {"id": "s", "tool_name": "statistics"}},
    {"status": "follow_up", "reason": "more evidence", "steps": [{"id": "missing_tool"}]},
    {"status": "follow_up", "reason": "more evidence", "steps": [{"id": "s", "tool_name": "statistics", "read_only": "false"}]},
])
def test_adaptive_review_contract_rejects_invalid_payloads(payload):
    with pytest.raises(PlannerError):
        AnalysisPlanner.parse_review(json.dumps(payload))


def test_adaptive_review_contract_accepts_complete_and_follow_up():
    complete = AnalysisPlanner.parse_review(json.dumps({"status": "complete", "reason": "Enough evidence"}))
    follow_up = AnalysisPlanner.parse_review(json.dumps({
        "status": "follow_up",
        "reason": "Need a grouped comparison",
        "steps": [{
            "id": "adaptive_1", "tool_name": "groupby_analysis",
            "kwargs": {"group_column": "region", "value_column": "sales"},
            "read_only": True,
        }],
    }))

    assert complete["steps"] == []
    assert follow_up["steps"][0]["tool_name"] == "groupby_analysis"


def test_adaptive_review_prompt_has_valid_complete_and_follow_up_examples():
    planner = AnalysisPlanner(FakeProvider("{}"), tools_registry=TOOL_FUNCTIONS)
    prompt = planner.build_review_prompt("Review sales", [], "[Dataset: sales]", TOOL_SCHEMAS)[0]["content"]

    complete_text = re.search(r"Complete example: (\{[^\n]+\})\n", prompt).group(1)
    follow_up_text = prompt.split("Follow-up example (replace dataset/column names as needed): ", 1)[1]
    follow_up_text = follow_up_text.split(" Temporal grounding rule:", 1)[0]
    complete = json.loads(complete_text)
    follow_up = json.loads(follow_up_text)

    assert complete["status"] == "complete" and "steps" not in complete
    assert follow_up["status"] == "follow_up" and len(follow_up["steps"]) == 1
    assert "reason is always required" in prompt
    assert "complete means current findings are sufficient" in prompt
    assert "follow_up requires a non-empty steps array with 1-2 entries" in prompt


@pytest.mark.parametrize("steps", [pytest.param("missing", id="omitted"), [], None])
def test_complete_review_normalizes_only_empty_or_null_steps(steps):
    payload = {"status": "complete", "reason": "The evidence is sufficient."}
    if steps != "missing":
        payload["steps"] = steps

    parsed = AnalysisPlanner.parse_review(json.dumps(payload))
    assert parsed == {
        "status": "complete", "reason": "The evidence is sufficient.", "steps": [],
    }
