import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from agent.agent import TOOL_FUNCTIONS, TOOL_SCHEMAS
from autonomous.executor import Executor, ExecutorError
from autonomous.plan import AnalysisPlan, PlanStep
from reports.report_builder import build_analysis_report, render_markdown
from tools.ml_model import MIN_UNIQUE_GROUPS, train_ml_model


def grouped_dataframe(groups=20, rows_per_group=6, task="classification"):
    rng = np.random.default_rng(314)
    entity = np.repeat([f"customer_{index}" for index in range(groups)], rows_per_group)
    feature = rng.normal(size=len(entity))
    if task == "classification":
        target = (feature + np.repeat(np.arange(groups) % 2, rows_per_group) > 0).astype(int)
    else:
        target = 2.5 * feature + rng.normal(scale=0.2, size=len(entity))
    return pd.DataFrame({"customer_id": entity, "feature": feature, "target": target})


def _grouped_result(task="classification", random_state=42, dataframe=None):
    return train_ml_model(
        dataframe if dataframe is not None else grouped_dataframe(task=task),
        "target", task, group_column="customer_id", random_state=random_state,
    )


def test_grouped_binary_classification_is_isolated_and_bounded():
    observed = {}
    from sklearn.model_selection import GroupShuffleSplit
    original_split = GroupShuffleSplit.split

    def recording_split(self, X, y=None, groups=None):
        train, test = next(original_split(self, X, y, groups))
        observed["train"] = set(groups.iloc[train])
        observed["test"] = set(groups.iloc[test])
        return iter([(train, test)])

    with patch.object(GroupShuffleSplit, "split", recording_split):
        result = _grouped_result()

    assert "error" not in result
    assert observed["train"].isdisjoint(observed["test"])
    assert result["split"] == {
        "strategy": "random", "train_rows": 96, "test_rows": 24, "test_size": 0.2,
        "random_state": 42, "time_column": None, "group_column": "customer_id",
        "group_aware": True, "total_groups": 20, "train_groups": 16,
        "test_groups": 4, "group_overlap_count": 0,
    }
    assert "customer_id" not in result["features_used"]
    assert not any(key.endswith("members") or key.endswith("group_ids") for key in result["split"])


def test_grouped_regression_is_deterministic_and_seed_can_change_partition():
    first = _grouped_result("regression", 11)
    repeated = _grouped_result("regression", 11)
    changed = _grouped_result("regression", 12)

    assert first["models"] == repeated["models"]
    assert first["models"] != changed["models"]
    assert first["prediction_unit"]["evaluation_unit"] == "group"


@pytest.mark.parametrize("kwargs,error", [
    ({"group_column": "missing"}, "not found"),
    ({"group_column": "target"}, "target column"),
    ({"group_column": "customer_id", "feature_columns": ["feature", "customer_id"]}, "evaluation metadata"),
])
def test_group_column_contract_rejects_structural_conflicts(kwargs, error):
    result = train_ml_model(grouped_dataframe(), "target", "classification", **kwargs)
    assert error in result["error"]


def test_missing_group_rows_are_dropped_and_all_null_groups_fail():
    dataframe = grouped_dataframe()
    dataframe.loc[:5, "customer_id"] = None
    result = _grouped_result(dataframe=dataframe)
    assert result["rows_dropped"] == 6
    assert any("missing group values" in warning for warning in result["warnings"])

    dataframe["customer_id"] = None
    assert "no usable values" in _grouped_result(dataframe=dataframe)["error"]


def test_insufficient_unique_groups_are_rejected():
    result = _grouped_result(dataframe=grouped_dataframe(groups=MIN_UNIQUE_GROUPS - 1, rows_per_group=10))
    assert f"at least {MIN_UNIQUE_GROUPS}" in result["error"]


def test_grouped_classification_requires_train_classes_and_warns_for_test_coverage():
    dataframe = grouped_dataframe(groups=10, rows_per_group=10)
    dataframe["target"] = 0
    dataframe.loc[dataframe["customer_id"] == "customer_9", "target"] = 1

    assert "training data does not contain every" in _grouped_result(
        random_state=1, dataframe=dataframe
    )["error"]
    result = _grouped_result(random_state=0, dataframe=dataframe)
    assert "error" not in result
    assert any("grouped test split does not contain every" in warning for warning in result["warnings"])


def test_repeated_entity_diagnostics_warn_without_automatic_grouping():
    dataframe = grouped_dataframe()
    dataframe["store_id"] = np.repeat([f"store_{index}" for index in range(10)], 12)
    result = train_ml_model(dataframe, "target", "classification")

    candidates = result["prediction_unit"]["repeated_entity_candidates"]
    assert {item["column"] for item in candidates} == {"customer_id", "store_id"}
    assert result["split"]["group_aware"] is False
    assert result["prediction_unit"]["group_column"] is None
    assert any("may overestimate generalization" in warning for warning in result["warnings"])


def test_unique_identifier_does_not_trigger_repeated_entity_warning():
    dataframe = grouped_dataframe()
    dataframe["order_id"] = [f"order_{index}" for index in range(len(dataframe))]
    result = train_ml_model(
        dataframe.drop(columns="customer_id"), "target", "classification"
    )
    assert result["prediction_unit"]["repeated_entity_candidates"] == []
    assert not any("Repeated entities" in warning for warning in result["warnings"])


def test_temporal_group_combination_is_rejected():
    dataframe = grouped_dataframe()
    dataframe["date"] = pd.date_range("2024-01-01", periods=len(dataframe))
    result = train_ml_model(
        dataframe, "target", "classification", split_strategy="temporal",
        time_column="date", group_column="customer_id",
    )
    assert "grouped-temporal" in result["error"]


@pytest.mark.parametrize("overrides,message", [
    ({"group_column": "missing"}, "unavailable column"),
    ({"group_column": "target"}, "cannot equal"),
    ({"group_column": "customer_id", "feature_columns": ["feature", "customer_id"]}, "cannot appear"),
    ({"group_column": "customer_id", "split_strategy": "temporal", "time_column": "date"}, "grouped-temporal"),
])
def test_autonomous_preflight_validates_group_column(overrides, message):
    dataframe = grouped_dataframe()
    dataframe["date"] = pd.date_range("2024-01-01", periods=len(dataframe))
    kwargs = {"dataset_name": "data", "target_column": "target", "task_type": "classification", **overrides}
    plan = AnalysisPlan("p", "model", ["data"], [
        PlanStep("ml", "train_ml_model", kwargs=kwargs, read_only=True)
    ])
    with pytest.raises(ExecutorError, match=message):
        Executor(TOOL_FUNCTIONS, tool_schemas=TOOL_SCHEMAS).preflight(plan, {"data": dataframe})


def _report(result):
    return render_markdown(build_analysis_report(
        "model target", {"answer": "Evaluation complete.", "figure": None, "trace": [],
                         "evidence": [{"tool_name": "train_ml_model", "result": result}]},
        {"data": grouped_dataframe()},
    ))


def test_report_renders_group_isolation_and_row_level_warning():
    grouped = _report(_grouped_result())
    assert "Group isolation: `customer_id`" in grouped
    assert "groups do not cross train/test" in grouped

    row_level = _report(train_ml_model(grouped_dataframe(), "target", "classification"))
    assert "may place the same entity in both training and test" in row_level


def test_report_qualifies_dummy_winner_negative_r2_and_associations():
    rng = np.random.default_rng(44)
    dataframe = pd.DataFrame({"feature": rng.normal(size=300), "target": rng.normal(size=300)})
    result = train_ml_model(dataframe, "target", "regression")
    assert result["best_model"] == "dummy_mean"
    markdown = _report(result)
    assert "did not outperform the naive baseline" in markdown
    assert "Negative R²" in markdown
    assert "not validated useful predictors" in markdown


def test_group_schema_is_bounded_and_no_group_is_chosen_implicitly():
    properties = next(
        schema for schema in TOOL_SCHEMAS if schema["function"]["name"] == "train_ml_model"
    )["function"]["parameters"]["properties"]
    assert properties["group_column"]["type"] == ["string", "null"]
    result = train_ml_model(grouped_dataframe(), "target", "classification")
    assert result["split"]["group_aware"] is False
    assert len(json.dumps(result)) < 50_000
