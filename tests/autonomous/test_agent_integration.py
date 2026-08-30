import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent import Agent, MAX_LLM_REQUEST_CHARS, _estimate_request_chars
from autonomous.results import Finding


def _message(content, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _complete_review():
    return _message(json.dumps({"status": "complete", "reason": "The findings are sufficient."}))


def test_agent_run_executes_autonomous_plan(sample_df):
    plan = {
        "id": "plan_1",
        "objective": "Describe the dataset",
        "datasets": ["default"],
        "steps": [{
            "id": "step_1",
            "tool_name": "dataset_info",
            "kwargs": {"dataset_name": "default"},
            "read_only": True,
            "outputs_expected": ["shape"],
        }],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)),
            _complete_review(),
            _message("The dataset contains six rows and six columns."),
        ]
        result = Agent().run("Describe the dataset", sample_df, autonomous=True)

    assert result["answer"] == "The dataset contains six rows and six columns."
    assert result["trace"][-1]["autonomous"] is True
    assert result["findings"][0].tool_name == "dataset_info"
    synthesis_call = llm_class.return_value.chat.call_args_list[2]
    assert synthesis_call.kwargs["tool_choice"] == "none"
    assert "tools" not in synthesis_call.kwargs
    assert "Structured findings" in synthesis_call.args[0][1]["content"]


def test_agent_autonomous_plan_accepts_scalar_expected_outputs(sample_df):
    plan = {
        "id": "scalar-output",
        "objective": "Describe the dataset",
        "datasets": ["default"],
        "steps": [{
            "id": "step_1", "tool_name": "dataset_info",
            "kwargs": {"dataset_name": "default"}, "read_only": True,
            "outputs_expected": ["dataset profile"],
        }],
        "expected_outputs": "dataset summary",
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)), _complete_review(), _message("Autonomous summary."),
        ]
        result = Agent().run("Describe the dataset", sample_df, autonomous=True)

    assert result["answer"] == "Autonomous summary."
    assert result["trace"][-1]["autonomous"] is True
    assert not any(item.get("step") == "autonomous_fallback" for item in result["trace"])
    assert result["trace"][1]["plan"]["expected_outputs"] == ["dataset summary"]


def test_agent_run_falls_back_when_autonomous_plan_is_invalid(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message("not json"),
            _message("Reactive answer."),
        ]
        result = Agent().run("Describe the dataset", sample_df, autonomous=True)

    assert result["answer"] == "Reactive answer."
    assert result["trace"][-1] == {"step": "final_answer", "tool_used": False}


def test_agent_run_default_preserves_reactive_path(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Existing answer.")
        result = Agent().run("Hello", sample_df, autonomous=False)

    assert result["answer"] == "Existing answer."
    assert llm_class.return_value.chat.call_count == 1


def test_autonomous_synthesis_compacts_large_findings(sample_df):
    plan = {
        "id": "large",
        "objective": "Summarize",
        "datasets": ["default"],
        "steps": [{"id": "s1", "tool_name": "dataset_info", "kwargs": {}, "read_only": True}],
    }
    large_finding = Finding(
        id="f1",
        step_id="s1",
        tool_name="dataset_info",
        datasets=["default"],
        result={"result": {f"row_{i}": "x" * 1000 for i in range(100)}},
        provenance={"dataset_names": ["default"]},
    )

    with patch("agent.agent.LLMClient") as llm_class, patch("agent.agent.Executor") as executor_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)),
            _complete_review(),
            _message("The available findings were summarized."),
        ]
        executor_class.return_value.execute.return_value.all.return_value = [large_finding]
        executor_class.return_value.findings_store.all.return_value = [large_finding]
        Agent().run("Summarize", sample_df, autonomous=True)

    synthesis_messages = llm_class.return_value.chat.call_args_list[2].args[0]
    synthesis_text = synthesis_messages[1]["content"]
    assert _estimate_request_chars(synthesis_messages) <= MAX_LLM_REQUEST_CHARS
    assert "Result truncated" in synthesis_text
    assert "row_99" not in synthesis_text


def test_autonomous_synthesis_failure_preserves_findings(sample_df):
    plan = {
        "id": "fallback",
        "objective": "Describe",
        "datasets": ["default"],
        "steps": [{"id": "s1", "tool_name": "dataset_info", "kwargs": {}, "read_only": True}],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)),
            _complete_review(),
            RuntimeError("synthesis unavailable"),
        ]
        result = Agent().run("Describe", sample_df, autonomous=True)

    assert result["answer"].startswith("Autonomous analysis completed.")
    assert result["findings"]
    assert result["trace"][-2] == {"step": "autonomous_synthesis", "success": False}
    assert llm_class.return_value.chat.call_count == 3


def test_auto_mode_routes_clear_multistep_request_to_autonomous(sample_df):
    plan = {
        "id": "auto",
        "objective": "Rank stores and analyze their trend",
        "datasets": ["default"],
        "steps": [{"id": "s1", "tool_name": "dataset_info", "kwargs": {}, "read_only": True}],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)),
            _complete_review(),
            _message("The ranked-store trend analysis is complete."),
        ]
        result = Agent().run("Show the sales trend of the top 3 stores", sample_df, autonomous=None)

    assert result["trace"][1] == {
        "step": "routing",
        "mode": "auto",
        "decision": "autonomous",
        "reason": "clear_multistep_signal",
    }
    assert result["trace"][-1]["autonomous"] is True


def test_auto_mode_keeps_simple_analysis_reactive(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("The average is 225.")
        result = Agent().run("What is the average sales?", sample_df, autonomous=None)

    assert result["answer"] == "The average is 225."
    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "no_clear_multistep_signal"
    assert llm_class.return_value.chat.call_count == 1


def test_auto_mode_keeps_visualization_first_request_reactive(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("I can create that chart.")
        result = Agent().run("Chart the top stores over time", sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "visualization_first_request"
    assert llm_class.return_value.chat.call_count == 1


def test_auto_mode_keeps_context_dependent_followup_reactive(sample_df):
    history = [
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "Sales declined."},
    ]
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("I will use the prior context.")
        result = Agent().run(
            "Why did it decrease?",
            sample_df,
            conversation_history=history,
            autonomous=None,
        )

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "context_dependent_request"
    assert llm_class.return_value.chat.call_count == 1


def test_auto_mode_autonomous_failure_falls_back_to_reactive(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message("invalid plan"),
            _message("Reactive fallback answer."),
        ]
        result = Agent().run("Show the sales trend of the top 3 stores", sample_df, autonomous=None)

    assert result["answer"] == "Reactive fallback answer."
    assert result["trace"][1]["decision"] == "autonomous"
    assert result["trace"][-1] == {"step": "final_answer", "tool_used": False}


@pytest.mark.parametrize("ranking_word", ["most", "least", "biggest", "largest", "smallest"])
def test_auto_mode_routes_ranking_synonyms_with_time_analysis(sample_df, ranking_word):
    plan = {
        "id": "ranking-synonym",
        "objective": "Rank products over time",
        "datasets": ["default"],
        "steps": [{"id": "s1", "tool_name": "dataset_info", "kwargs": {}, "read_only": True}],
    }
    question = f"Which products grew the {ranking_word} over time?"
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)),
            _complete_review(),
            _message("The product growth analysis is complete."),
        ]
        result = Agent().run(question, sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "autonomous"
    assert result["trace"][1]["reason"] == "clear_multistep_signal"


def test_auto_mode_plot_monthly_sales_has_visualization_reason(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("I can plot monthly sales.")
        result = Agent().run("Plot monthly sales.", sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "visualization_first_request"


def test_auto_mode_context_reference_has_context_reason(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Which stores do you mean?")
        result = Agent().run("What about those stores?", sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "context_dependent_request"


def test_auto_mode_simple_most_request_remains_reactive(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Region A has the most sales.")
        result = Agent().run("Which region has the most sales?", sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "no_clear_multistep_signal"
    assert llm_class.return_value.chat.call_count == 1


@pytest.mark.parametrize("question", [
    "Which products drove the decline?",
    "Which products drove the decline from 2024 to 2025?",
    "Which regions contributed most to revenue growth?",
    "Which segments offset the revenue decline?",
    "Within Region A, which products drove the decline?",
    "Which products drove the previous year's decline?",
])
def test_auto_mode_routes_contribution_change_requests_with_specific_reason(sample_df, question):
    plan = {
        "id": "contribution-routing",
        "objective": "Decompose the KPI change",
        "datasets": ["default"],
        "steps": [{"id": "s1", "tool_name": "dataset_info", "kwargs": {}, "read_only": True}],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _message(json.dumps(plan)), _complete_review(), _message("Contribution analysis complete."),
        ]
        result = Agent().run(question, sample_df, autonomous=None)

    assert result["trace"][1] == {
        "step": "routing", "mode": "auto", "decision": "autonomous",
        "reason": "contribution_change_signal",
    }


def test_temporal_previous_is_not_treated_as_context_reference(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Quarter comparison complete.")
        result = Agent().run(
            "Compare this quarter with the previous quarter.", sample_df, autonomous=None
        )

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "no_clear_multistep_signal"


@pytest.mark.parametrize("question", [
    "Analyze the previous result.",
    "Use the previous analysis.",
    "Compare with the previous answer.",
])
def test_contextual_previous_remains_context_dependent(sample_df, question):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Context-dependent answer.")
        result = Agent().run(question, sample_df, autonomous=None)

    assert result["trace"][1]["decision"] == "reactive"
    assert result["trace"][1]["reason"] == "context_dependent_request"
