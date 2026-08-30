import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from agent.agent import Agent


def message(content):
    return SimpleNamespace(content=content, tool_calls=None)


def initial_plan(steps=None, datasets=None, plan_id="initial"):
    return {
        "id": plan_id,
        "objective": "Initial investigation",
        "datasets": datasets or ["default"],
        "steps": steps or [{
            "id": "initial_info", "tool_name": "dataset_info",
            "kwargs": {"dataset_name": "default"}, "read_only": True,
        }],
    }


def review(status="complete", reason="Initial evidence is sufficient.", steps=None):
    payload = {"status": status, "reason": reason}
    if steps is not None:
        payload["steps"] = steps
    return message(json.dumps(payload))


def statistics_step(step_id="adaptive_stats", dataset="default", column="Weekly_Sales", read_only=True):
    return {
        "id": step_id, "tool_name": "statistics",
        "kwargs": {"dataset_name": dataset, "column": column},
        "read_only": read_only, "outputs_expected": ["summary"],
    }


def run_autonomous(sample_df, plan, review_message, synthesis="Final grounded answer."):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            message(json.dumps(plan)), review_message, message(synthesis),
        ]
        result = Agent().run("Investigate sales", sample_df, autonomous=True)
        return result, llm_class.return_value


def test_review_complete_executes_no_follow_up_and_synthesizes_initial_findings(sample_df):
    result, llm = run_autonomous(sample_df, initial_plan(), review())

    assert len(result["findings"]) == 1
    assert any(item == {"step": "adaptive_stop", "reason": "review_complete"} for item in result["trace"])
    assert "dataset_info" in llm.chat.call_args_list[2].args[0][1]["content"]
    assert llm.chat.call_count == 3


def test_one_follow_up_executes_and_adds_provenanced_finding(sample_df):
    result, llm = run_autonomous(
        sample_df,
        initial_plan(),
        review("follow_up", "Quantify the sales distribution.", [statistics_step()]),
    )

    assert [finding.tool_name for finding in result["findings"]] == ["dataset_info", "statistics"]
    adaptive = result["findings"][-1]
    assert adaptive.provenance["parent_plan_id"] == "initial"
    assert adaptive.provenance["adaptive_round"] == 1
    assert adaptive.provenance["reviewer_reason"] == "Quantify the sales distribution."
    assert any(item.get("step") == "adaptive_execution" and item["status"] == "completed" for item in result["trace"])
    assert "statistics" in llm.chat.call_args_list[2].args[0][1]["content"]
    assert llm.chat.call_count == 3


def test_two_follow_up_steps_execute_within_budget(sample_df):
    second = {
        "id": "adaptive_missing", "tool_name": "missing_values",
        "kwargs": {"dataset_name": "default"}, "read_only": True,
    }
    result, _ = run_autonomous(
        sample_df, initial_plan(),
        review("follow_up", "Check distribution and data quality.", [statistics_step(), second]),
    )

    assert [finding.tool_name for finding in result["findings"]] == [
        "dataset_info", "statistics", "missing_values",
    ]
    execution = next(item for item in result["trace"] if item.get("step") == "adaptive_execution")
    assert execution["executed_steps"] == 2


def test_follow_up_can_use_initial_canonical_derived_dataset():
    customers = pd.DataFrame({"customer_id": [1, 2], "region": ["N", "S"]})
    orders = pd.DataFrame({"customer_id": [1, 2], "amount": [50.0, 75.0]})
    plan = initial_plan(
        datasets=["customers", "orders"],
        steps=[{
            "id": "join", "tool_name": "execute_join", "read_only": False,
            "kwargs": {
                "left_dataset": "customers", "right_dataset": "orders",
                "left_on": "customer_id", "right_on": "customer_id",
            },
        }],
    )
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            message(json.dumps(plan)),
            review("follow_up", "Summarize joined order amounts.", [
                statistics_step(dataset="derived_join_1", column="amount")
            ]),
            message("Joined amounts average 62.5."),
        ]
        result = Agent().run("Join customers and orders, then investigate amounts", datasets={
            "customers": customers, "orders": orders,
        }, autonomous=True)

    assert "derived_join_1" in result["findings"][-1].datasets
    assert result["findings"][-1].result["mean"] == 62.5


def test_more_than_two_follow_up_steps_is_invalid_and_preserves_initial(sample_df):
    steps = [statistics_step(f"s{index}") for index in range(3)]
    result, _ = run_autonomous(sample_df, initial_plan(), review("follow_up", "More work", steps))

    assert len(result["findings"]) == 1
    assert any(item.get("status") == "invalid" for item in result["trace"] if item.get("step") == "adaptive_review")


def test_global_step_budget_rejects_follow_up(sample_df):
    steps = [{
        "id": f"initial_{index}", "tool_name": "dataset_info",
        "kwargs": {"dataset_name": "default"}, "read_only": True,
    } for index in range(10)]
    result, _ = run_autonomous(
        sample_df, initial_plan(steps=steps),
        review("follow_up", "One more check", [statistics_step()]),
    )

    assert len(result["findings"]) == 10
    assert {"step": "adaptive_stop", "reason": "global_step_limit"} in result["trace"]


@pytest.mark.parametrize("step, stop_reason", [
    ({
        "id": "join", "tool_name": "execute_join", "kwargs": {}, "read_only": False,
    }, "follow_up_join_prohibited"),
    (statistics_step(read_only=False), "follow_up_mutation_prohibited"),
])
def test_follow_up_join_or_mutation_is_rejected(sample_df, step, stop_reason):
    result, _ = run_autonomous(
        sample_df, initial_plan(), review("follow_up", "Unsafe request", [step])
    )
    assert len(result["findings"]) == 1
    assert {"step": "adaptive_stop", "reason": stop_reason} in result["trace"]


def test_malformed_review_preserves_findings_and_still_synthesizes(sample_df):
    result, llm = run_autonomous(sample_df, initial_plan(), message("not json"))

    assert len(result["findings"]) == 1
    assert result["answer"] == "Final grounded answer."
    assert any(item.get("status") == "invalid" for item in result["trace"] if item.get("step") == "adaptive_review")
    assert llm.chat.call_count == 3


def test_empty_review_content_records_bounded_empty_content_diagnostic(sample_df):
    result, llm = run_autonomous(sample_df, initial_plan(), message(""))
    diagnostic = next(item for item in result["trace"] if item.get("step") == "adaptive_review")

    assert diagnostic["status"] == "invalid"
    assert diagnostic["reason"] == "invalid_review_response"
    assert diagnostic["failure_stage"] == "empty_content"
    assert len(diagnostic["diagnostic_message"]) <= 500
    assert len(result["findings"]) == 1
    assert result["answer"] == "Final grounded answer."
    assert result["trace"][-1]["autonomous"] is True
    assert llm.chat.call_count == 3


def test_invalid_json_review_records_parse_diagnostic_without_raw_response(sample_df):
    raw_response = "not-json raw-review-secret"
    result, _ = run_autonomous(sample_df, initial_plan(), message(raw_response))
    diagnostic = next(item for item in result["trace"] if item.get("step") == "adaptive_review")
    serialized = json.dumps(diagnostic)

    assert diagnostic["failure_stage"] == "json_parse"
    assert "AdaptiveReviewError" in diagnostic["exception_type"]
    assert raw_response not in serialized
    assert "raw-review-secret" not in serialized
    assert "Weekly_Sales" not in serialized
    assert len(result["findings"]) == 1
    assert result["answer"] == "Final grounded answer."


def test_contract_review_diagnostic_has_only_safe_bounded_shape_metadata(sample_df):
    unsafe_key = "api_key=supersecret"
    payload = {
        "status": "complete", "reason": "Enough evidence", unsafe_key: "dataset-private-value",
    }
    result, _ = run_autonomous(sample_df, initial_plan(), message(json.dumps(payload)))
    diagnostic = next(item for item in result["trace"] if item.get("step") == "adaptive_review")
    serialized = json.dumps(diagnostic)

    assert diagnostic["failure_stage"] == "contract_validation"
    assert set(diagnostic["parsed_top_level_types"].values()) == {"str"}
    assert "status" in diagnostic["parsed_top_level_keys"]
    assert "supersecret" not in serialized
    assert "dataset-private-value" not in serialized
    assert "Completed findings" not in serialized
    assert len(diagnostic["diagnostic_message"]) <= 500
    assert len(serialized) < 2500
    assert len(result["findings"]) == 1
    assert result["answer"] == "Final grounded answer."


def test_review_provider_failure_preserves_findings_without_reactive_fallback(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            message(json.dumps(initial_plan())), RuntimeError("review unavailable"), message("Initial answer."),
        ]
        result = Agent().run("Investigate sales", sample_df, autonomous=True)

    assert len(result["findings"]) == 1
    assert result["answer"] == "Initial answer."
    assert any(item.get("status") == "failed" for item in result["trace"] if item.get("step") == "adaptive_review")
    assert llm_class.return_value.chat.call_count == 3


def test_follow_up_preflight_failure_executes_zero_follow_up_tools(sample_df):
    result, _ = run_autonomous(
        sample_df, initial_plan(),
        review("follow_up", "Inspect unavailable profit.", [statistics_step(column="Profit")]),
    )

    assert [finding.tool_name for finding in result["findings"]] == ["dataset_info"]
    execution = next(item for item in result["trace"] if item.get("step") == "adaptive_execution")
    assert execution == {
        "step": "adaptive_execution", "plan_id": "adaptive_initial",
        "executed_steps": 0, "status": "failed",
    }


def test_follow_up_runtime_failure_preserves_initial_findings(sample_df):
    def failing_statistics(df, **kwargs):
        raise RuntimeError("runtime failure")

    with patch.dict("agent.agent.TOOL_FUNCTIONS", {"statistics": failing_statistics}):
        result, _ = run_autonomous(
            sample_df, initial_plan(),
            review("follow_up", "Inspect sales statistics.", [statistics_step()]),
        )

    assert [finding.tool_name for finding in result["findings"]] == ["dataset_info"]
    assert {"step": "adaptive_stop", "reason": "follow_up_failed"} in result["trace"]
