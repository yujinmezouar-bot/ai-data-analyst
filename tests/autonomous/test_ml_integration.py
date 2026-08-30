import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agent.agent import Agent, TOOL_FUNCTIONS, TOOL_SCHEMAS, _build_execution_plan
from autonomous.executor import Executor, ExecutorError
from autonomous.plan import AnalysisPlan, PlanStep
from autonomous.results import Finding
from reports.report_builder import build_analysis_report, render_markdown


def ml_dataframe(rows=100):
    rng = np.random.default_rng(91)
    spend = rng.normal(100, 20, rows)
    segment = np.where(np.arange(rows) % 2, "business", "consumer")
    churn = (spend < 100).astype(int)
    return pd.DataFrame({"spend": spend, "segment": segment, "churn": churn})


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def tool_message(arguments):
    call = SimpleNamespace(function=SimpleNamespace(
        name="train_ml_model", arguments=json.dumps(arguments)
    ))
    return message(tool_calls=[call])


def ml_step(identifier="ml", **overrides):
    kwargs = {
        "dataset_name": "sales", "target_column": "churn", "task_type": "classification"
    }
    kwargs.update(overrides)
    return PlanStep(id=identifier, tool_name="train_ml_model", kwargs=kwargs, read_only=True)


def test_ml_tool_is_registered_with_canonical_schema_and_routing_is_unchanged():
    assert TOOL_FUNCTIONS["train_ml_model"]
    assert any(schema["function"]["name"] == "train_ml_model" for schema in TOOL_SCHEMAS)
    assert _build_execution_plan("Build a classification model to predict churn") is None


def test_reactive_agent_executes_ml_and_returns_bounded_evidence():
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            tool_message({"target_column": "churn", "task_type": "classification"}),
            message("Training complete."),
            message("The held-out model evaluation is complete."),
        ]
        result = Agent().run("Build a model to predict churn", ml_dataframe(), autonomous=False)

    evidence = result["evidence"][0]
    assert evidence["tool_name"] == "train_ml_model"
    assert evidence["result"]["target_column"] == "churn"
    assert evidence["result"]["models"][1]["name"] == "logistic_regression"


@pytest.mark.parametrize("kwargs,message_text", [
    ({"target_column": "missing"}, "unavailable column"),
    ({"feature_columns": ["missing"]}, "unavailable column"),
    ({"exclude_columns": ["missing"]}, "unavailable column"),
    ({"test_size": 0.9}, "test_size"),
    ({"split_strategy": "temporal"}, "requires time_column"),
    ({"time_column": "spend"}, "only be used"),
    ({"feature_columns": ["churn"]}, "cannot also"),
    ({"feature_columns": ["spend"], "exclude_columns": ["spend"]}, "overlap"),
])
def test_ml_complete_plan_preflight_rejects_structural_errors(kwargs, message_text):
    plan = AnalysisPlan("p", "train", ["sales"], [ml_step(**kwargs)])
    executor = Executor(TOOL_FUNCTIONS, tool_schemas=TOOL_SCHEMAS)
    with pytest.raises(ExecutorError, match=message_text):
        executor.preflight(plan, {"sales": ml_dataframe()})


def test_more_than_one_ml_step_rejected_before_any_tool_executes():
    calls = []

    def should_not_run(df, **kwargs):
        calls.append(kwargs)
        return {}

    registry = dict(TOOL_FUNCTIONS)
    registry["train_ml_model"] = should_not_run
    plan = AnalysisPlan("p", "train twice", ["sales"], [ml_step("one"), ml_step("two")])
    with pytest.raises(ExecutorError, match="at most one"):
        Executor(registry, tool_schemas=TOOL_SCHEMAS).execute(plan, {"sales": ml_dataframe()})
    assert calls == []


def test_initial_autonomous_ml_execution_records_normal_finding():
    plan = {
        "id": "ml_plan", "objective": "Predict churn", "datasets": ["sales"],
        "steps": [{
            "id": "train", "tool_name": "train_ml_model", "read_only": True,
            "kwargs": {"dataset_name": "sales", "target_column": "churn", "task_type": "classification"},
        }],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            message(json.dumps(plan)),
            message(json.dumps({"status": "complete", "reason": "Evaluation is sufficient."})),
            message("Churn evaluation completed."),
        ]
        result = Agent().run(
            "Build a classification model for churn", datasets={"sales": ml_dataframe()}, autonomous=True
        )

    finding = result["findings"][0]
    assert finding.tool_name == "train_ml_model"
    assert finding.result["task_type"] == "classification"
    assert finding.provenance["category"] == "analysis_tool"


def test_adaptive_ml_training_is_prohibited_and_initial_findings_survive():
    initial_plan = {
        "id": "initial", "objective": "Inspect data", "datasets": ["sales"],
        "steps": [{
            "id": "info", "tool_name": "dataset_info", "read_only": True,
            "kwargs": {"dataset_name": "sales"},
        }],
    }
    review = {
        "status": "follow_up", "reason": "Try predictive modeling.",
        "steps": [{
            "id": "adaptive_ml", "tool_name": "train_ml_model", "read_only": True,
            "kwargs": {"dataset_name": "sales", "target_column": "churn", "task_type": "classification"},
        }],
    }
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            message(json.dumps(initial_plan)), message(json.dumps(review)), message("Initial evidence only."),
        ]
        result = Agent().run("Inspect sales", datasets={"sales": ml_dataframe()}, autonomous=True)

    assert [finding.tool_name for finding in result["findings"]] == ["dataset_info"]
    assert {"step": "adaptive_stop", "reason": "follow_up_ml_prohibited"} in result["trace"]


def test_reactive_and_autonomous_ml_evidence_render_report_section_and_safeguards():
    ml_result = TOOL_FUNCTIONS["train_ml_model"](
        ml_dataframe(), target_column="churn", task_type="classification"
    )
    reactive = {
        "answer": "Evaluation complete.", "figure": None, "trace": [],
        "evidence": [{"tool_name": "train_ml_model", "result": ml_result}],
    }
    autonomous = {
        "answer": "Evaluation complete.", "figure": None, "trace": [],
        "findings": [Finding(
            id="finding_1", step_id="train", tool_name="train_ml_model", datasets=["sales"],
            result=ml_result, metadata={"plan_id": "ml_plan"}, provenance={"plan_id": "ml_plan"},
        )],
    }
    for result in (reactive, autonomous):
        markdown = render_markdown(build_analysis_report(
            "Predict churn", result, {"sales": ml_dataframe()}
        ))
        assert "## Machine Learning Results" in markdown
        assert "dummy_most_frequent" in markdown and "logistic_regression" in markdown
        assert "Held-out Test Metrics" in markdown
        assert "predictive" in markdown.lower() and "do not imply causation" in markdown
        assert "not external validation" in markdown


def test_non_ml_report_has_no_ml_section():
    markdown = render_markdown(build_analysis_report(
        "Describe", {"answer": "Done", "figure": None, "trace": [], "evidence": []},
        {"sales": ml_dataframe()},
    ))
    assert "Machine Learning Results" not in markdown
