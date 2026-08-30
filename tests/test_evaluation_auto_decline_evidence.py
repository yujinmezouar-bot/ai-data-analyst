from copy import deepcopy

import pytest

from evaluation.benchmark_cases import benchmark_cases
from evaluation.datasets import build_benchmark_datasets
from evaluation.evaluator import score_case
from evaluation.ground_truth import contribution


def _case():
    return next(case for case in benchmark_cases() if case.id == "auto_decline_why")


def _contribution_result():
    truth = contribution(build_benchmark_datasets()["sales"], "product", "revenue")
    return {
        "period_a": truth["period_a"],
        "period_b": truth["period_b"],
        "contributors": [{
            "group": truth["largest_decline"],
            "value_a": truth["leading_value_a"],
            "value_b": truth["leading_value_b"],
            "absolute_change": truth["leading_absolute_change"],
            "percentage_change": truth["leading_percentage_change"],
            "effect": truth["leading_effect"],
        }],
        "overall": {
            "absolute_change": truth["total_change"],
            "direction": truth["direction"],
        },
    }


def _observed(result=None, extra_tools=()):
    evidence = [{"tool_name": "kpi_contribution_analysis", "result": result or _contribution_result()}]
    evidence.extend({"tool_name": tool, "result": {}} for tool in extra_tools)
    return {
        "answer": "Grounded contribution answer.",
        "trace": [
            {"step": "routing", "decision": "autonomous"},
            {"step": "autonomous_plan", "plan": {"datasets": ["sales"]}},
            {"step": "final_answer", "autonomous": True},
        ],
        "evidence": evidence,
        "has_figure": False,
    }


def _failed_numerical_paths(scored):
    return {
        check["expected"] for check in scored["checks"]
        if check["kind"] == "numerical" and not check["passed"]
    }


def test_auto_decline_tool_contract_requires_only_contribution():
    case = _case()
    assert "2024" in case.question and "2025" in case.question
    assert "observed factors" in case.question
    assert case.expected_tools == ("kpi_contribution_analysis",)
    assert "time_analysis" not in case.forbidden_tools
    assert "time_analysis" not in case.expected_tools
    assert "correlation_analysis" not in case.expected_tools
    assert "correlation_analysis" not in case.forbidden_tools


def test_auto_decline_annual_ground_truth_expectations_remain_exact():
    case = _case()
    expected = {item.path: item.expected for item in case.expected_values}
    assert expected == {
        "period_a": "2024",
        "period_b": "2025",
        "contributors.0.group": "Gamma",
        "contributors.0.value_a": 5408.0,
        "contributors.0.value_b": 1647.0,
        "contributors.0.absolute_change": -3761.0,
        "contributors.0.percentage_change": -69.55,
        "contributors.0.effect": "reinforces_decrease",
        "overall.absolute_change": -2347.0,
        "overall.direction": "decrease",
    }

    truth = contribution(build_benchmark_datasets()["sales"], "product", "revenue")
    assert truth["period_a"] == "2024" and truth["period_b"] == "2025"
    assert truth["largest_decline"] == "Gamma"
    assert truth["leading_absolute_change"] == -3761.0
    assert truth["total_change"] == -2347.0


def test_correct_contribution_evidence_passes_with_optional_tools():
    scored = score_case(_case(), _observed(extra_tools=("time_analysis", "correlation_analysis")))
    assert scored["passed"]


def test_wrong_leading_declining_product_fails():
    result = _contribution_result()
    result["contributors"][0]["group"] = "Delta"
    scored = score_case(_case(), _observed(result))
    assert not scored["passed"]
    assert "Gamma" in _failed_numerical_paths(scored)


def test_wrong_gamma_absolute_change_fails():
    result = _contribution_result()
    result["contributors"][0]["absolute_change"] += 100
    scored = score_case(_case(), _observed(result))
    assert not scored["passed"]
    assert -3761.0 in _failed_numerical_paths(scored)


@pytest.mark.parametrize("field,value", [("period_a", "2023"), ("period_b", "2026")])
def test_wrong_compared_period_fails(field, value):
    result = _contribution_result()
    result[field] = value
    assert not score_case(_case(), _observed(result))["passed"]


def test_missing_required_contribution_evidence_fails():
    result = deepcopy(_contribution_result())
    result["contributors"][0].pop("value_b")
    scored = score_case(_case(), _observed(result))
    assert not scored["passed"]
    assert "numerical correctness" in scored["failure_classifications"]
