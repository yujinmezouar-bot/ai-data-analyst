import pytest

from autonomous.plan import AnalysisPlan, PlanStep


def test_planstep_from_dict_valid():
    d = {"id": "s1", "tool_name": "statistics", "kwargs": {"col": "sales"}, "read_only": True, "outputs_expected": ["summary"]}
    step = PlanStep.from_dict(d)
    assert step.id == "s1"
    assert step.tool_name == "statistics"
    assert step.kwargs["col"] == "sales"


def test_plan_from_dict_valid():
    data = {
        "id": "plan_1",
        "objective": "Analyze sales",
        "datasets": ["sales"],
        "steps": [
            {"id": "step1", "tool_name": "statistics", "kwargs": {}, "read_only": True, "outputs_expected": ["summary"]}
        ],
        "constraints": {},
        "expected_outputs": ["summary"],
    }
    plan = AnalysisPlan.from_dict(data, max_steps=5, allowed_datasets=["sales"])
    assert plan.id == "plan_1"
    assert plan.objective.startswith("Analyze")
    assert len(plan.steps) == 1


def test_plan_missing_required_fields():
    with pytest.raises(ValueError):
        AnalysisPlan.from_dict({}, max_steps=5)


def test_plan_too_many_steps():
    data = {
        "id": "p",
        "objective": "o",
        "datasets": [],
        "steps": [{"id": f"s{i}", "tool_name": "t"} for i in range(20)],
    }
    with pytest.raises(ValueError):
        AnalysisPlan.from_dict(data, max_steps=5)


def test_plan_invalid_step_structure():
    data = {
        "id": "p",
        "objective": "o",
        "datasets": [],
        "steps": ["not an object"],
    }
    with pytest.raises(ValueError):
        AnalysisPlan.from_dict(data, max_steps=5)


def test_plan_model_still_rejects_scalar_expected_outputs():
    data = {
        "id": "p",
        "objective": "Analyze sales",
        "datasets": ["sales"],
        "steps": [],
        "expected_outputs": "summary",
    }
    with pytest.raises(ValueError, match="expected_outputs.*must be a list"):
        AnalysisPlan.from_dict(data, allowed_datasets=["sales"])
