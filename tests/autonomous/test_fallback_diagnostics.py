import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent import Agent, MAX_AUTONOMOUS_DIAGNOSTIC_MESSAGE_CHARS


def _message(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _plan(*, dataset="default", tool="dataset_info", kwargs=None, step_id="step_1"):
    return {
        "id": "diagnostic_plan",
        "objective": "Exercise autonomous fallback diagnostics",
        "datasets": [dataset],
        "steps": [{
            "id": step_id,
            "tool_name": tool,
            "kwargs": kwargs or {},
            "read_only": True,
            "outputs_expected": [],
        }],
    }


def _fallback_entry(result):
    return next(item for item in result["trace"] if item.get("step") == "autonomous_fallback")


def test_planner_call_failure_records_diagnostic_and_reactive_fallback_runs(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            RuntimeError("api_key=supersecret Bearer hidden-token"),
            _message("Reactive fallback answer."),
        ]
        result = Agent().run("Analyze the dataset", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    assert result["answer"] == "Reactive fallback answer."
    assert diagnostic["stage"] == "planner_call"
    assert diagnostic["planner_output_received"] is False
    assert "supersecret" not in json.dumps(diagnostic)
    assert "hidden-token" not in json.dumps(diagnostic)


def test_malformed_planner_json_records_parse_stage(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [_message("not json"), _message("Reactive.")]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    assert diagnostic["stage"] == "planner_parse"
    assert diagnostic["planner_output_received"] is True
    assert diagnostic["planner_json_parsed"] is False
    assert "not json" not in json.dumps(diagnostic)


def test_plan_validation_failure_records_validation_stage(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(_plan(dataset="missing_dataset"))),
            _message("Reactive."),
        ]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    assert diagnostic["stage"] == "plan_validation"
    assert diagnostic["planner_json_parsed"] is True
    assert diagnostic["plan_validated"] is False


def test_preflight_failure_records_bounded_reason_before_reactive_fallback(sample_df):
    plan = _plan(tool="statistics", kwargs={"column": "missing_column"})
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)), _message("Reactive after preflight failure."),
        ]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    assert result["answer"] == "Reactive after preflight failure."
    assert diagnostic["stage"] == "preflight"
    assert diagnostic["plan_validated"] is True
    assert diagnostic["failing_step_id"] == "step_1"
    assert "missing_column" in diagnostic["reason"]
    assert len(diagnostic["reason"]) <= MAX_AUTONOMOUS_DIAGNOSTIC_MESSAGE_CHARS
    assert diagnostic["plan_summary"]["steps"] == [{"id": "step_1", "tool_name": "statistics"}]
    assert "kwargs" not in json.dumps(diagnostic["plan_summary"])


def test_initial_execution_failure_records_step_without_raw_data(sample_df):
    def fail_tool(_df):
        raise RuntimeError("row value private-row-value password=unsafe")

    with patch("agent.agent.LLMClient") as llm_class, patch.dict(
        "agent.agent.TOOL_FUNCTIONS", {"dataset_info": fail_tool}
    ):
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(_plan(step_id="failing_step"))), _message("Reactive."),
        ]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    serialized = json.dumps(diagnostic)
    assert diagnostic["stage"] == "execution"
    assert diagnostic["failing_step_id"] == "failing_step"
    assert "private-row-value" not in serialized
    assert "unsafe" not in serialized
    assert "kwargs" not in serialized


def test_successful_autonomous_run_has_no_fallback_diagnostic(sample_df):
    review = {"status": "complete", "reason": "Initial evidence is sufficient."}
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(_plan())), _message(json.dumps(review)), _message("Autonomous answer."),
        ]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    assert not any(item.get("step") == "autonomous_fallback" for item in result["trace"])


def test_diagnostic_total_size_is_bounded(sample_df):
    oversized = _plan(
        tool="statistics",
        kwargs={"column": "missing_" + "c" * 5000},
        step_id="s" * 5000,
    )
    oversized["id"] = "p" * 5000
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(oversized)), _message("Reactive."),
        ]
        result = Agent().run("Analyze", sample_df, autonomous=True)

    diagnostic = _fallback_entry(result)
    assert diagnostic["plan_validated"] is True
    assert len(diagnostic["message"]) <= MAX_AUTONOMOUS_DIAGNOSTIC_MESSAGE_CHARS
    assert len(json.dumps(diagnostic)) < 2000
