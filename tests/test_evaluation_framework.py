import json

import pytest

from evaluation.benchmark_cases import BenchmarkCase, ValueExpectation, benchmark_cases
from evaluation.datasets import build_benchmark_datasets
from evaluation.evaluator import provider_available, run_benchmark, score_case, serialize_run, write_artifacts
from evaluation.ground_truth import contribution, grouped, monthly, yearly
from evaluation.metrics import latency_summary, summarize, values_match


def test_catalog_is_balanced_valid_and_within_target_size():
    cases = benchmark_cases()
    assert 40 <= len(cases) <= 60
    assert len({case.id for case in cases}) == len(cases)
    categories = {case.category for case in cases}
    assert categories == {
        "descriptive", "ranking", "time", "change", "contribution", "correlation_outliers",
        "visualization", "multi_dataset", "autonomous", "ml_classification", "ml_regression",
        "ml_safety", "context", "fallback",
    }
    for case in cases:
        case.validate()


def test_case_validation_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="Unknown"):
        BenchmarkCase("x", "unknown", "question", ("sales",)).validate()
    with pytest.raises(ValueError, match="both"):
        BenchmarkCase("x", "descriptive", "question", ("sales",), expected_tools=("statistics",), forbidden_tools=("statistics",)).validate()


def test_ground_truth_is_independent_and_deterministic():
    first, second = build_benchmark_datasets(), build_benchmark_datasets()
    assert first.keys() == second.keys()
    assert first["sales"].equals(second["sales"])
    assert sum(grouped(first["sales"], "product", "revenue").values()) == pytest.approx(first["sales"]["revenue"].sum())
    assert len(monthly(first["sales"], "revenue")) == 24
    assert set(yearly(first["sales"], "revenue")) == {2024, 2025}
    assert contribution(first["sales"], "product", "revenue")["changes"]


@pytest.mark.parametrize("actual,expected,tolerance,match", [
    (1.00001, 1.0, 1e-4, True), (1.1, 1.0, 1e-4, False), ("North", "North", 0, True),
])
def test_numeric_tolerance(actual, expected, tolerance, match):
    assert values_match(actual, expected, tolerance) is match


def test_routing_tool_numerical_and_safety_scoring():
    case = BenchmarkCase(
        "score", "ml_safety", "question", ("sales",), expected_mode="reactive",
        expected_tools=("train_ml_model",), expected_values=(ValueExpectation("train_ml_model", "split.group_aware", False),),
        expected_warning="Repeated entities",
    )
    observed = {
        "answer": "Safe result", "trace": [{"step":"routing","decision":"reactive"}],
        "evidence": [{"tool_name":"train_ml_model","result":{"split":{"group_aware":False},"error":"Repeated entities"}}],
        "has_figure": False,
    }
    scored = score_case(case, observed)
    assert scored["passed"]
    assert {check["kind"] for check in scored["checks"]} >= {"routing", "tool_selection", "numerical", "safety", "ml"}


def test_summary_category_failure_and_latency_metrics():
    results = [
        {"category":"descriptive","passed":True,"checks":[{"kind":"routing","passed":True}],"latency_ms":10,"trace":[],"failure_classifications":[]},
        {"category":"descriptive","passed":False,"checks":[{"kind":"routing","passed":False,"expected":"reactive","actual":"autonomous"}],"latency_ms":30,"trace":[],"failure_classifications":["routing"]},
    ]
    summary = summarize(results, "deterministic")
    assert summary["overall_pass_rate"] == 50
    assert summary["routing_accuracy"] == 50
    assert summary["false_autonomous_count"] == 1
    assert summary["category_breakdown"]["descriptive"]["pass_rate"] == 50
    assert summary["failure_classification"] == {"routing": 1}
    assert latency_summary([10, 20, 30]) == {"mean_ms":20.0,"median_ms":20.0,"p95_ms":30.0}


def test_deterministic_subset_runs_real_agent_and_serializes_stably(tmp_path):
    run = run_benchmark("deterministic", {"desc_average_quantity", "multi_safe_join", "ml_churn"})
    assert run["status"] == "completed"
    assert run["summary"]["total_cases"] == 3
    serialized = json.dumps(run, sort_keys=True, default=str)
    assert "GROQ_API_KEY" not in serialized
    assert serialize_run(run) == serialize_run(run)
    json_path, markdown_path = write_artifacts(run, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["total_cases"] == 3
    assert "Behavioral Benchmark" in markdown_path.read_text(encoding="utf-8")


def test_provider_unavailable_is_non_failing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert provider_available() is False
    run = run_benchmark("real", {"desc_average_quantity"})
    assert run["status"] == "unavailable"
    assert run["cases"] == []
    assert run["summary"]["total_cases"] == 0


def test_benchmark_retains_autonomous_fallback_diagnostic():
    run = run_benchmark("deterministic", {"fallback_planner"})
    trace = run["cases"][0]["trace"]
    diagnostic = next(item for item in trace if item.get("step") == "autonomous_fallback")
    assert diagnostic["stage"] == "planner_parse"
    assert diagnostic["planner_output_received"] is True
    assert diagnostic["planner_json_parsed"] is False
