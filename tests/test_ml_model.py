import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tools.ml_model import (
    MAX_ENCODED_FEATURES,
    MAX_FEATURE_ASSOCIATIONS,
    MAX_ML_ROWS,
    MAX_RAW_FEATURES,
    TRAIN_ML_MODEL_SCHEMA,
    train_ml_model,
)


def classification_df(rows=180, classes=2):
    rng = np.random.default_rng(123)
    numeric = rng.normal(size=rows)
    category = np.where(np.arange(rows) % 3 == 0, "North", "South").astype(object)
    if classes == 2:
        target = (numeric + (category == "North") * 0.8 + rng.normal(scale=0.4, size=rows) > 0).astype(int)
    else:
        target = np.arange(rows) % classes
        numeric = target + rng.normal(scale=0.3, size=rows)
    return pd.DataFrame({"numeric": numeric, "category": category, "target": target})


def regression_df(rows=180):
    rng = np.random.default_rng(456)
    numeric = rng.normal(size=rows)
    category = np.where(np.arange(rows) % 2, "A", "B").astype(object)
    target = 3.0 * numeric + (category == "A") * 2.0 + rng.normal(scale=0.25, size=rows)
    return pd.DataFrame({"numeric": numeric, "category": category, "target": target})


def test_binary_classification_models_metrics_baseline_and_associations():
    result = train_ml_model(classification_df(), "target", "classification")

    assert "error" not in result
    assert [model["name"] for model in result["models"]] == ["dummy_most_frequent", "logistic_regression"]
    assert result["selection_metric"].startswith("macro_f1")
    for model in result["models"]:
        assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "confusion_matrix"} <= set(model["metrics"])
        assert {"positive_precision", "positive_recall", "positive_f1", "roc_auc", "pr_auc"} <= set(model["metrics"])
    assert {item["feature"] for item in result["feature_associations"]} == {"numeric", "category"}
    assert all(item["interpretation"] == "predictive association, not causation" for item in result["feature_associations"])


def test_multiclass_classification_is_bounded_and_aggregates_coefficients():
    result = train_ml_model(classification_df(classes=3), "target", "classification")

    assert "error" not in result
    assert len(result["target_summary"]["classes"]) == 3
    assert "roc_auc" not in result["models"][1]["metrics"]
    assert len(result["feature_associations"]) <= MAX_FEATURE_ASSOCIATIONS
    assert [item["feature"] for item in result["feature_associations"]].count("category") <= 1


def test_regression_returns_exact_metric_set_ridge_associations_and_best_rule():
    result = train_ml_model(regression_df(), "target", "regression")

    assert "error" not in result
    assert [model["name"] for model in result["models"]] == ["dummy_mean", "ridge"]
    assert set(result["models"][1]["metrics"]) == {"mae", "rmse", "r2"}
    assert result["best_model"] == "ridge"
    assert result["selection_metric"].startswith("rmse")
    assert result["feature_associations"][0]["feature"] == "numeric"


@pytest.mark.parametrize("task,factory", [("classification", classification_df), ("regression", regression_df)])
def test_repeated_execution_is_deterministic(task, factory):
    first = train_ml_model(factory(), "target", task, random_state=17)
    second = train_ml_model(factory(), "target", task, random_state=17)
    assert first["models"] == second["models"]
    assert first["feature_associations"] == second["feature_associations"]


def test_missing_categorical_numeric_and_infinite_values_are_safely_preprocessed():
    dataframe = regression_df()
    dataframe.loc[0:5, "numeric"] = np.nan
    dataframe.loc[6, "numeric"] = np.inf
    dataframe.loc[7:12, "category"] = None
    result = train_ml_model(dataframe, "target", "regression")

    assert "error" not in result
    assert any("infinite" in warning for warning in result["warnings"])


def test_temporal_unknown_category_succeeds_and_uses_latest_rows():
    dataframe = regression_df(60)
    dataframe["Date"] = pd.date_range("2024-01-01", periods=60)
    dataframe.loc[48:, "category"] = "NeverSeenInTrain"
    result = train_ml_model(
        dataframe, "target", "regression", split_strategy="temporal", time_column="Date"
    )

    assert "error" not in result
    assert result["split"]["train_rows"] == 48
    assert result["split"]["test_rows"] == 12
    assert "Date" not in result["features_used"]
    assert {item["column"]: item["reason"] for item in result["features_excluded"]}["Date"] == "temporal_split_column"


def test_preprocessors_are_fitted_on_training_rows_only():
    from sklearn.impute import SimpleImputer

    observed_rows = []
    original_fit = SimpleImputer.fit

    def recording_fit(self, X, y=None):
        observed_rows.append(len(X))
        return original_fit(self, X, y)

    with patch.object(SimpleImputer, "fit", recording_fit):
        result = train_ml_model(regression_df(50), "target", "regression", test_size=0.2)

    assert "error" not in result
    assert observed_rows and set(observed_rows) == {40}


@pytest.mark.parametrize("dataframe,target,task,error_text", [
    (classification_df(), "missing", "classification", "not found"),
    (pd.DataFrame({"x": range(40), "target": [None] * 40}), "target", "classification", "usable"),
    (pd.DataFrame({"x": np.arange(40, dtype=float), "target": [1] * 40}), "target", "classification", "distinct"),
    (classification_df(29), "target", "classification", "At least 30"),
    (pd.DataFrame({"x": np.arange(40), "target": [f"v{i}" for i in range(40)]}), "target", "regression", "numeric"),
    (pd.DataFrame({"x": np.arange(42, dtype=float), "target": [i % 21 for i in range(42)]}), "target", "classification", "at most 20"),
])
def test_invalid_target_conditions_return_actionable_errors(dataframe, target, task, error_text):
    result = train_ml_model(dataframe, target, task)
    assert error_text.lower() in result["error"].lower()


def test_target_feature_and_exact_duplicate_leakage_are_rejected():
    dataframe = classification_df()
    dataframe["leaked"] = dataframe["target"]

    assert "cannot also" in train_ml_model(
        dataframe, "target", "classification", feature_columns=["target", "numeric"]
    )["error"]
    assert "exactly duplicates" in train_ml_model(
        dataframe, "target", "classification", feature_columns=["numeric", "leaked"]
    )["error"]


def test_identifier_datetime_constant_all_null_and_high_cardinality_auto_exclusions():
    dataframe = regression_df(220)
    dataframe["customer_id"] = [f"C{i}" for i in range(220)]
    dataframe["Date"] = pd.date_range("2020-01-01", periods=220)
    dataframe["constant"] = 1
    dataframe["all_null"] = None
    dataframe["high_card"] = [f"group_{i % 110}" for i in range(220)]
    result = train_ml_model(dataframe, "target", "regression")
    reasons = {item["column"]: item["reason"] for item in result["features_excluded"]}

    assert reasons["customer_id"] == "identifier_like"
    assert reasons["Date"] == "datetime_unsupported"
    assert reasons["constant"] == "constant"
    assert reasons["all_null"] == "all_null"
    assert reasons["high_card"] == "high_cardinality"


def test_explicit_identifier_and_unsafe_time_feature_are_rejected():
    dataframe = regression_df()
    dataframe["customer_id"] = [f"C{i}" for i in range(len(dataframe))]
    dataframe["Date"] = pd.date_range("2020-01-01", periods=len(dataframe))

    assert "identifier-like" in train_ml_model(
        dataframe, "target", "regression", feature_columns=["numeric", "customer_id"]
    )["error"].lower()
    assert "cannot also" in train_ml_model(
        dataframe, "target", "regression", feature_columns=["numeric", "Date"],
        split_strategy="temporal", time_column="Date",
    )["error"]


def test_feature_encoded_and_row_bounds():
    raw = pd.DataFrame({f"x{i}": np.arange(40, dtype=float) + i for i in range(MAX_RAW_FEATURES + 1)})
    raw["target"] = np.arange(40, dtype=float) * 2
    assert "raw features" in train_ml_model(raw, "target", "regression")["error"]

    encoded = pd.DataFrame({
        f"cat{i}": [f"value_{row % 100}" for row in range(600)] for i in range(6)
    })
    encoded["target"] = np.arange(600, dtype=float)
    result = train_ml_model(encoded, "target", "regression")
    assert str(MAX_ENCODED_FEATURES) in result["error"]

    oversized = pd.DataFrame({"x": np.zeros(MAX_ML_ROWS + 1), "target": np.arange(MAX_ML_ROWS + 1)})
    assert str(MAX_ML_ROWS) in train_ml_model(oversized, "target", "regression")["error"]


def test_random_classification_is_stratified_and_imbalance_warns():
    dataframe = classification_df(100)
    dataframe["target"] = [0] * 95 + [1] * 5
    result = train_ml_model(dataframe, "target", "classification")

    assert "error" not in result
    assert result["split"]["test_rows"] == 20
    assert any("minority class" in warning for warning in result["warnings"])
    assert all(len(model["metrics"]["confusion_matrix"]) == 2 for model in result["models"])


def test_invalid_split_configurations():
    dataframe = regression_df()
    dataframe["Date"] = pd.date_range("2024-01-01", periods=len(dataframe))

    assert "between 0.1 and 0.4" in train_ml_model(dataframe, "target", "regression", test_size=0.9)["error"]
    assert "required" in train_ml_model(dataframe, "target", "regression", split_strategy="temporal")["error"]
    assert "only be supplied" in train_ml_model(dataframe, "target", "regression", time_column="Date")["error"]
    invalid_date = dataframe.assign(Date=dataframe["Date"].astype(str))
    assert "datetime dtype" in train_ml_model(
        invalid_date, "target", "regression", split_strategy="temporal", time_column="Date"
    )["error"]


def test_output_is_bounded_json_and_contains_no_runtime_model_objects():
    dataframe = regression_df()
    for index in range(25):
        dataframe[f"feature_{index}"] = dataframe["numeric"] + index
    result = train_ml_model(dataframe, "target", "regression")
    serialized = json.dumps(result)

    assert len(result["feature_associations"]) <= MAX_FEATURE_ASSOCIATIONS
    assert len(serialized) < 50_000
    assert not any(term in serialized for term in ("Pipeline(", "ColumnTransformer(", "predictions", "probabilities"))


def test_schema_contract_is_single_bounded_training_tool():
    function = TRAIN_ML_MODEL_SCHEMA["function"]
    properties = function["parameters"]["properties"]
    assert function["name"] == "train_ml_model"
    assert function["parameters"]["required"] == ["target_column", "task_type"]
    assert properties["task_type"]["enum"] == ["classification", "regression"]
    assert properties["split_strategy"]["enum"] == ["random", "temporal"]
    assert "model" not in properties and "hyperparameters" not in properties
