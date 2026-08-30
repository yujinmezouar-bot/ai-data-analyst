import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent import (
    MAX_LLM_REQUEST_CHARS,
    MAX_TOOL_RESULT_CHARS,
    TOOL_SCHEMAS,
    Agent,
    _compact_ml_result_for_llm,
    _compact_tool_result,
    _estimate_request_chars,
)
from evaluation.datasets import build_benchmark_datasets
from reports.report_builder import build_analysis_report, render_markdown
from tools.ml_model import train_ml_model


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_message(arguments):
    call = SimpleNamespace(function=SimpleNamespace(
        name="train_ml_model", arguments=json.dumps(arguments)
    ))
    return _message("Run bounded ML.", [call])


def _run_large_reactive_ml():
    sales = build_benchmark_datasets()["sales"]
    with patch("agent.agent.LLMClient") as llm_class:
        llm = llm_class.return_value
        llm.chat.side_effect = [
            _tool_message({"target_column":"returned", "task_type":"classification"}),
            _message("Evidence is sufficient."),
            _message("The held-out evaluation is complete."),
        ]
        result = Agent().run("Predict returned orders.", datasets={"sales": sales}, autonomous=False)
    return sales, result, llm


def test_large_reactive_ml_preserves_complete_bounded_structured_evidence():
    _, result, _ = _run_large_reactive_ml()
    evidence = result["evidence"][0]["result"]

    assert len(json.dumps(evidence)) > MAX_TOOL_RESULT_CHARS
    assert evidence["task_type"] == "classification"
    assert evidence["target_column"] == "returned"
    assert {"rows_received", "rows_used", "rows_dropped"} <= evidence.keys()
    assert evidence["features_used"] and evidence["features_excluded"]
    assert {"strategy", "group_aware", "group_column", "group_overlap_count"} <= evidence["split"].keys()
    assert evidence["target_summary"]["class_balance"]
    assert [model["baseline"] for model in evidence["models"]] == [True, False]
    assert all(model["metrics"] for model in evidence["models"])
    assert evidence["best_model"] and evidence["selection_metric"]
    assert len(evidence["feature_associations"]) <= 20
    assert evidence["warnings"] and any("Repeated entities" in warning for warning in evidence["warnings"])
    assert evidence["limitations"]
    serialized = json.dumps(evidence)
    assert len(serialized) < 50_000
    def all_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(all_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(all_keys(item) for item in value), set())
        return set()
    assert not {"estimator", "pipeline", "transformer", "predictions", "probabilities"} & all_keys(evidence)
    assert "Pipeline(" not in serialized and "ColumnTransformer(" not in serialized


def test_large_ml_uses_separate_bounded_llm_representation():
    sales, result, llm = _run_large_reactive_ml()
    full_result = result["evidence"][0]["result"]
    compact = _compact_tool_result(_compact_ml_result_for_llm(full_result))

    assert len(json.dumps(compact)) <= MAX_TOOL_RESULT_CHARS
    assert compact["models"] == full_result["models"]
    assert compact["best_model"] == full_result["best_model"]
    assert compact["split"] == full_result["split"]
    assert compact["warnings"] and compact["limitations"]
    assert len(compact["feature_associations"]) <= 5

    decision_messages = llm.chat.call_args_list[1].args[0]
    final_messages = llm.chat.call_args_list[2].args[0]
    assert _estimate_request_chars(decision_messages, TOOL_SCHEMAS) <= MAX_LLM_REQUEST_CHARS
    assert _estimate_request_chars(final_messages) <= MAX_LLM_REQUEST_CHARS
    assert "dummy_most_frequent" in final_messages[1]["content"]
    assert "logistic_regression" in final_messages[1]["content"]
    assert "Repeated entities" in final_messages[1]["content"]


def test_reactive_ml_report_receives_complete_metrics_warnings_and_limitations():
    sales, result, _ = _run_large_reactive_ml()
    report = build_analysis_report("Predict returned orders", result, {"sales": sales})
    markdown = render_markdown(report)

    assert "dummy_most_frequent" in markdown
    assert "logistic_regression" in markdown
    assert "Selected model" in markdown
    assert "Repeated entities" in markdown
    assert "Top Predictive Feature Associations" in markdown
    assert "not external validation" in markdown


def test_non_ml_compaction_contract_is_unchanged():
    large = {"result": {f"group_{index}": "x" * 300 for index in range(100)}}
    compact = _compact_tool_result(large)
    assert len(compact["result"]) == 25
    assert "truncated" in compact["note"].lower()


def test_autonomous_ml_result_schema_matches_reactive_evidence_schema():
    dataframe = build_benchmark_datasets()["customers"]
    direct = train_ml_model(dataframe, "churn", "classification")
    assert "error" not in direct
    assert set(direct) >= {
        "task_type", "target_column", "split", "models", "best_model",
        "feature_associations", "warnings", "limitations",
    }
