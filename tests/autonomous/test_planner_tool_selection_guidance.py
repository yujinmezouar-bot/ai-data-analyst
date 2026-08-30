import json
from types import SimpleNamespace

from agent.agent import TOOL_FUNCTIONS, TOOL_SCHEMAS
from autonomous.planner import AnalysisPlanner
from tools.contribution_analysis import KPI_CONTRIBUTION_SCHEMA
from tools.period_comparison import PERCENTAGE_CHANGE_SCHEMA


class FakeProvider:
    def __init__(self, content="{}"):
        self.content = content

    def chat(self, messages=None, tools=None, tool_choice=None):
        return SimpleNamespace(content=self.content)


def _description(schema):
    return schema["function"]["description"].lower()


def test_percentage_change_description_excludes_per_group_change_ranking():
    description = _description(PERCENTAGE_CHANGE_SCHEMA)
    assert "does not calculate separate changes or rankings for every group" in description
    assert "group_column with filter_values only scopes" in description
    assert "do not use it to rank which groups" in description


def test_contribution_description_covers_group_decline_growth_and_contribution():
    description = _description(KPI_CONTRIBUTION_SCHEMA)
    assert "signed group-level movement" in description
    assert "declined or grew most" in description
    for concept in ("drove", "contributed", "offset", "accounted for"):
        assert concept in description
    assert "never causes" in description


def test_planner_prompt_contains_cross_tool_selection_and_noncausal_guidance():
    planner = AnalysisPlanner(FakeProvider(), tools_registry=TOOL_FUNCTIONS)
    prompt = planner._build_prompt("Analyze KPI movement", {
        "datasets": ["sales"], "tool_schemas": TOOL_SCHEMAS,
    })[0]["content"]

    assert "percentage_change for overall period-to-period movement" in prompt
    assert "groupby_analysis for group levels" in prompt
    assert "kpi_contribution_analysis for which groups declined or grew most" in prompt
    assert "Pair contribution analysis with time_analysis" in prompt
    assert "correlation_analysis only for statistical association" in prompt
    assert "cannot substitute for contribution or establish causal explanation" in prompt
    assert "Current tools do not establish causality" in prompt


def test_relevant_tool_schema_structures_are_unchanged():
    percentage_parameters = PERCENTAGE_CHANGE_SCHEMA["function"]["parameters"]
    contribution_parameters = KPI_CONTRIBUTION_SCHEMA["function"]["parameters"]

    assert percentage_parameters["required"] == ["date_column", "value_column"]
    assert set(percentage_parameters["properties"]) == {
        "dataset_name", "date_column", "value_column", "period", "agg_function",
        "year_1", "year_2", "group_column", "filter_values",
    }
    assert contribution_parameters["required"] == [
        "date_column", "metric_column", "group_column", "period_a", "period_b",
    ]
    assert set(contribution_parameters["properties"]) == {
        "dataset_name", "date_column", "metric_column", "group_column", "period_a",
        "period_b", "period", "agg_function", "filter_column", "filter_values", "top_n",
    }


def test_fake_provider_accepts_generic_temporal_group_decline_plan():
    payload = {
        "id": "group_change",
        "objective": "Measure product revenue movement between periods",
        "datasets": ["sales"],
        "steps": [{
            "id": "contribution", "tool_name": "kpi_contribution_analysis",
            "kwargs": {
                "dataset_name": "sales", "date_column": "date", "metric_column": "revenue",
                "group_column": "product", "period_a": "2024", "period_b": "2025",
            },
            "read_only": True, "outputs_expected": ["signed product changes"],
        }],
        "constraints": {}, "expected_outputs": ["ranked product movement"],
    }
    planner = AnalysisPlanner(
        FakeProvider(json.dumps(payload)),
        tools_registry={"kpi_contribution_analysis": object()},
        validate_tools=True,
    )

    plan = planner.plan("Which categories declined most between two periods?", {
        "datasets": ["sales"],
    })
    assert plan.steps[0].tool_name == "kpi_contribution_analysis"
