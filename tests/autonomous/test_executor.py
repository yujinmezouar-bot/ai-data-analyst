import pandas as pd
import pytest

from autonomous.executor import Executor, ExecutorError
from autonomous.plan import AnalysisPlan, PlanStep
from tools.dataset_info import dataset_info
from tools.statistics import statistics


@pytest.fixture
def sample_datasets():
    df = pd.DataFrame({
        "sales": [10, 20, 30, 40],
        "region": ["north", "south", "north", "south"],
    })
    return {"sales": df}


def test_executor_runs_valid_read_only_plan(sample_datasets):
    plan = AnalysisPlan(
        id="plan_1",
        objective="Inspect the sales dataset",
        datasets=["sales"],
        steps=[
            PlanStep(
                id="step_1",
                tool_name="dataset_info",
                kwargs={"dataset_name": "sales"},
                read_only=True,
                outputs_expected=["dataset_profile"],
                category="analysis",
            )
        ],
        constraints={},
        expected_outputs=["dataset_profile"],
    )
    executor = Executor({"dataset_info": dataset_info})
    findings = executor.execute(plan, sample_datasets)

    assert len(findings) == 1
    assert findings.all()[0].tool_name == "dataset_info"
    assert findings.all()[0].result["num_rows"] == 4


def test_executor_handles_multiple_steps(sample_datasets):
    plan = AnalysisPlan(
        id="plan_2",
        objective="Summarize and inspect",
        datasets=["sales"],
        steps=[
            PlanStep(id="step_1", tool_name="dataset_info", kwargs={"dataset_name": "sales"}, read_only=True, outputs_expected=["profile"]),
            PlanStep(id="step_2", tool_name="statistics", kwargs={"dataset_name": "sales", "column": "sales"}, read_only=True, outputs_expected=["summary"]),
        ],
        constraints={},
        expected_outputs=["profile", "summary"],
    )
    executor = Executor({"dataset_info": dataset_info, "statistics": statistics})
    findings = executor.execute(plan, sample_datasets)

    assert len(findings) == 2
    assert [f.step_id for f in findings.all()] == ["step_1", "step_2"]
    assert findings.all()[1].result["mean"] == 25.0


def test_executor_rejects_unknown_tool(sample_datasets):
    plan = AnalysisPlan(
        id="p",
        objective="x",
        datasets=["sales"],
        steps=[PlanStep(id="s1", tool_name="unknown_tool", kwargs={"dataset_name": "sales"}, read_only=True)],
    )
    executor = Executor({})
    with pytest.raises(ExecutorError, match="not available"):
        executor.execute(plan, sample_datasets)


def test_executor_rejects_missing_dataset(sample_datasets):
    plan = AnalysisPlan(
        id="p",
        objective="x",
        datasets=["sales"],
        steps=[PlanStep(id="s1", tool_name="dataset_info", kwargs={"dataset_name": "missing"}, read_only=True)],
    )
    executor = Executor({"dataset_info": dataset_info})
    with pytest.raises(ExecutorError, match="not available"):
        executor.execute(plan, sample_datasets)


def test_executor_rejects_non_read_only_step(sample_datasets):
    plan = AnalysisPlan(
        id="p",
        objective="x",
        datasets=["sales"],
        steps=[PlanStep(id="s1", tool_name="dataset_info", kwargs={"dataset_name": "sales"}, read_only=False)],
    )
    executor = Executor({"dataset_info": dataset_info})
    with pytest.raises(ExecutorError, match="read-only"):
        executor.execute(plan, sample_datasets)


def test_executor_wraps_tool_exception(sample_datasets):
    def boom(df, **kwargs):
        raise ValueError("bad data")

    plan = AnalysisPlan(
        id="p",
        objective="x",
        datasets=["sales"],
        steps=[PlanStep(id="s1", tool_name="boom", kwargs={"dataset_name": "sales"}, read_only=True)],
    )
    executor = Executor({"boom": boom})
    with pytest.raises(ExecutorError, match="failed during execution"):
        executor.execute(plan, sample_datasets)


def test_executor_does_not_mutate_dataset(sample_datasets):
    original = sample_datasets["sales"].copy(deep=True)
    plan = AnalysisPlan(
        id="p",
        objective="x",
        datasets=["sales"],
        steps=[PlanStep(id="s1", tool_name="dataset_info", kwargs={"dataset_name": "sales"}, read_only=True)],
    )
    executor = Executor({"dataset_info": dataset_info})
    executor.execute(plan, sample_datasets)
    pd.testing.assert_frame_equal(sample_datasets["sales"], original)


def test_executor_rejects_steps_over_limit(sample_datasets):
    steps = [PlanStep(id=f"s{i}", tool_name="dataset_info", kwargs={"dataset_name": "sales"}, read_only=True) for i in range(11)]
    plan = AnalysisPlan(id="p", objective="x", datasets=["sales"], steps=steps)
    executor = Executor({"dataset_info": dataset_info}, max_steps=10)
    with pytest.raises(ExecutorError, match="maximum allowed"):
        executor.execute(plan, sample_datasets)
