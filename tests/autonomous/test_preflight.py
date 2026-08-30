from unittest.mock import Mock

import pandas as pd
import pytest

from agent.agent import TOOL_FUNCTIONS, TOOL_SCHEMAS
from autonomous.executor import Executor, ExecutorError
from autonomous.plan import AnalysisPlan, PlanStep


@pytest.fixture
def datasets():
    return {
        "customers": pd.DataFrame({"customer_id": [1, 2], "region": ["N", "S"]}),
        "orders": pd.DataFrame({"order_id": [10, 11], "customer_id": [1, 2], "amount": [50.0, 75.0]}),
    }


def make_executor(registrar=None, registry=None):
    return Executor(
        TOOL_FUNCTIONS if registry is None else registry,
        derived_dataset_register=registrar,
        tool_schemas=TOOL_SCHEMAS,
    )


def make_plan(steps):
    return AnalysisPlan(id="preflight", objective="validate", datasets=["customers", "orders"], steps=steps)


def test_unknown_step_dataset_fails_before_any_tool_executes(datasets):
    tool = Mock(return_value={"ok": True})
    executor = make_executor(registry={"dataset_info": tool})
    plan = make_plan([
        PlanStep(id="valid", tool_name="dataset_info", kwargs={"dataset_name": "customers"}),
        PlanStep(id="invalid", tool_name="dataset_info", kwargs={"dataset_name": "missing"}),
    ])

    with pytest.raises(ExecutorError, match="not available"):
        executor.execute(plan, datasets)
    tool.assert_not_called()


def test_nonexistent_column_fails_preflight(datasets):
    plan = make_plan([
        PlanStep(id="bad", tool_name="statistics", kwargs={"dataset_name": "orders", "column": "profit"})
    ])
    with pytest.raises(ExecutorError, match="Column 'profit'"):
        make_executor().execute(plan, datasets)


@pytest.mark.parametrize("step, message", [
    (PlanStep(id="missing", tool_name="groupby_analysis", kwargs={"dataset_name": "orders", "group_column": "customer_id"}), "missing required"),
    (PlanStep(id="unknown", tool_name="statistics", kwargs={"dataset_name": "orders", "bogus": 1}), "unknown argument"),
    (PlanStep(id="type", tool_name="groupby_analysis", kwargs={"dataset_name": "orders", "group_column": "customer_id", "value_column": "amount", "top_n": "ten"}), "invalid type"),
    (PlanStep(id="enum", tool_name="groupby_analysis", kwargs={"dataset_name": "orders", "group_column": "customer_id", "value_column": "amount", "agg_function": "average"}), "must be one of"),
])
def test_schema_argument_errors_fail_preflight(datasets, step, message):
    with pytest.raises(ExecutorError, match=message):
        make_executor().execute(make_plan([step]), datasets)


@pytest.mark.parametrize("step, message", [
    (PlanStep(id="unsupported", tool_name="invented_tool"), "not available"),
    (PlanStep(id="unsafe", tool_name="dataset_info", kwargs={"dataset_name": "customers"}, read_only=False), "read-only"),
])
def test_unsupported_or_unsafe_step_fails_preflight(datasets, step, message):
    with pytest.raises(ExecutorError, match=message):
        make_executor().execute(make_plan([step]), datasets)


def test_join_derived_dataset_dependency_passes_preflight_and_runtime(datasets):
    plan = make_plan([
        PlanStep(id="join", tool_name="execute_join", kwargs={
            "left_dataset": "customers", "right_dataset": "orders",
            "left_on": "customer_id", "right_on": "customer_id",
        }, read_only=False),
        PlanStep(id="analyze", tool_name="statistics", kwargs={
            "dataset_name": "derived_join_1", "column": "amount",
        }),
    ])

    findings = make_executor().execute(plan, datasets)
    assert findings.find_by_step("analyze")[0].result["mean"] == 62.5


def test_derived_dataset_without_earlier_producer_fails_preflight(datasets):
    plan = make_plan([
        PlanStep(id="bad", tool_name="statistics", kwargs={
            "dataset_name": "derived_join_1", "column": "amount",
        })
    ])
    with pytest.raises(ExecutorError, match="not available"):
        make_executor().execute(plan, datasets)


def test_late_invalid_step_executes_nothing_and_registers_nothing(datasets, monkeypatch):
    join_tool = Mock()
    registrar = Mock()
    monkeypatch.setattr("autonomous.executor.execute_join", join_tool)
    plan = make_plan([
        PlanStep(id="join", tool_name="execute_join", kwargs={
            "left_dataset": "customers", "right_dataset": "orders",
            "left_on": "customer_id", "right_on": "customer_id",
        }, read_only=False),
        PlanStep(id="bad", tool_name="statistics", kwargs={
            "dataset_name": "derived_join_1", "column": "missing_column",
        }),
    ])

    with pytest.raises(ExecutorError, match="missing_column"):
        make_executor(registrar=registrar).execute(plan, datasets)
    join_tool.assert_not_called()
    registrar.assert_not_called()
