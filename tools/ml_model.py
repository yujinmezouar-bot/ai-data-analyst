from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tools.dataset_info import build_dataset_profile


MAX_ML_ROWS = 50_000
MIN_ML_ROWS = 30
MAX_RAW_FEATURES = 50
MAX_ENCODED_FEATURES = 500
MAX_CLASSES = 20
MAX_CATEGORY_CARDINALITY = 100
MAX_FEATURE_ASSOCIATIONS = 20
MIN_UNIQUE_GROUPS = 5
MAX_ENTITY_CANDIDATES = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

ENTITY_NAME_HINTS = (
    "customer", "client", "product", "store", "account", "patient", "machine",
    "user", "member", "employee", "device", "vendor", "supplier", "entity",
    "order_id", "transaction_id", "uuid", "guid",
)


def _error(message: str, **details: Any) -> dict[str, Any]:
    return {"error": message, **details}


def _rounded(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _repeated_entity_candidates(dataframe: pd.DataFrame, excluded: set[str]) -> list[dict[str, Any]]:
    """Return bounded diagnostics for repeated, explicitly entity-like columns."""
    candidates: list[dict[str, Any]] = []
    for column in list(dataframe.columns)[:100]:
        if column in excluded:
            continue
        normalized = str(column).lower().replace("-", "_").replace(" ", "_")
        id_like = normalized == "id" or normalized.endswith("_id") or any(
            hint in normalized for hint in ENTITY_NAME_HINTS
        )
        if not id_like:
            continue
        series = dataframe[column].dropna()
        if series.empty:
            continue
        try:
            unique_groups = int(series.nunique())
        except TypeError:
            continue
        repeated_rows = int(len(series) - unique_groups)
        if unique_groups < 2 or repeated_rows <= 0:
            continue
        candidates.append({
            "column": str(column),
            "unique_entities": unique_groups,
            "rows_with_values": int(len(series)),
            "average_rows_per_entity": _rounded(len(series) / unique_groups),
            "repeated_entity_ratio": _rounded(repeated_rows / len(series)),
        })
        if len(candidates) >= MAX_ENTITY_CANDIDATES:
            break
    return candidates


def _classification_metrics(y_true, y_pred, probabilities, classes) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": _rounded(accuracy_score(y_true, y_pred)),
        "macro_precision": _rounded(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": _rounded(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": _rounded(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).astype(int).tolist(),
        "class_labels": [str(value) for value in classes],
    }
    if len(classes) == 2:
        positive = classes[-1]
        binary_true = np.asarray(y_true) == positive
        binary_pred = np.asarray(y_pred) == positive
        metrics.update({
            "positive_class": str(positive),
            "positive_precision": _rounded(precision_score(binary_true, binary_pred, zero_division=0)),
            "positive_recall": _rounded(recall_score(binary_true, binary_pred, zero_division=0)),
            "positive_f1": _rounded(f1_score(binary_true, binary_pred, zero_division=0)),
        })
        if probabilities is not None and len(np.unique(binary_true)) == 2:
            metrics["roc_auc"] = _rounded(roc_auc_score(binary_true, probabilities))
            metrics["pr_auc"] = _rounded(average_precision_score(binary_true, probabilities))
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics = {
        "mae": _rounded(mean_absolute_error(y_true, y_pred)),
        "rmse": _rounded(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(y_true) >= 2:
        metrics["r2"] = _rounded(r2_score(y_true, y_pred))
    return metrics


def _feature_associations(pipeline, numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    coefficients = np.asarray(model.coef_)
    magnitudes = np.abs(coefficients)
    if magnitudes.ndim == 2:
        magnitudes = magnitudes.mean(axis=0)

    raw_feature_map = list(numeric_columns)
    categorical_transformer = pipeline.named_steps["preprocessor"].named_transformers_.get("categorical")
    if categorical_transformer is not None:
        encoder = categorical_transformer.named_steps["encoder"]
        for column, categories in zip(categorical_columns, encoder.categories_):
            raw_feature_map.extend([column] * len(categories))

    if len(raw_feature_map) != len(magnitudes):
        return []
    aggregated: dict[str, float] = {}
    for column, magnitude in zip(raw_feature_map, magnitudes):
        aggregated[column] = aggregated.get(column, 0.0) + float(magnitude)
    total = sum(aggregated.values())
    ranked = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "feature": column,
            "association_score": _rounded(score),
            "relative_share_percentage": _rounded(100.0 * score / total) if total else None,
            "interpretation": "predictive association, not causation",
        }
        for column, score in ranked[:MAX_FEATURE_ASSOCIATIONS]
    ]


def train_ml_model(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    split_strategy: str = "random",
    time_column: str | None = None,
    group_column: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one bounded supervised-learning baseline and predictive model."""
    if df is None:
        return _error("No dataset is loaded.")
    if len(df) > MAX_ML_ROWS:
        return _error(
            f"Dataset has {len(df)} rows; ML V1 supports at most {MAX_ML_ROWS}. Filter the dataset first."
        )
    if not isinstance(target_column, str) or target_column not in df.columns:
        return _error(f"Target column '{target_column}' was not found.", available_columns=list(df.columns)[:100])
    if task_type not in {"classification", "regression"}:
        return _error("task_type must be 'classification' or 'regression'.")
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not 0.1 <= float(test_size) <= 0.4:
        return _error("test_size must be between 0.1 and 0.4.")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        return _error("random_state must be an integer.")
    if split_strategy not in {"random", "temporal"}:
        return _error("split_strategy must be 'random' or 'temporal'.")
    if split_strategy == "temporal" and not time_column:
        return _error("time_column is required for temporal splitting.")
    if split_strategy == "random" and time_column is not None:
        return _error("time_column may only be supplied when split_strategy is 'temporal'.")
    if time_column is not None and time_column not in df.columns:
        return _error(f"Time column '{time_column}' was not found.")
    if group_column is not None and (not isinstance(group_column, str) or group_column not in df.columns):
        return _error(f"Group column '{group_column}' was not found.")
    if group_column == target_column:
        return _error("group_column cannot be the target column.")
    if group_column is not None and group_column == time_column:
        return _error("group_column and time_column must be different columns.")
    if group_column is not None and split_strategy == "temporal":
        return _error(
            "ML V1 supports either row-level temporal evaluation or group-isolated random evaluation, "
            "not a combined grouped-temporal split. Remove group_column or use split_strategy='random'."
        )
    if split_strategy == "temporal" and not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        return _error(f"Time column '{time_column}' must have datetime dtype for temporal splitting.")
    for name, value in (("feature_columns", feature_columns), ("exclude_columns", exclude_columns)):
        if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
            return _error(f"{name} must be a list of column names or null.")

    exclude_columns = list(dict.fromkeys(exclude_columns or []))
    missing_exclusions = [column for column in exclude_columns if column not in df.columns]
    if missing_exclusions:
        return _error("Excluded columns were not found.", missing_columns=missing_exclusions)
    if feature_columns is not None:
        feature_columns = list(dict.fromkeys(feature_columns))
        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            return _error("Feature columns were not found.", missing_columns=missing_features)
        if target_column in feature_columns:
            return _error("The target column cannot also be a predictive feature.")
        overlap = sorted(set(feature_columns) & set(exclude_columns))
        if overlap:
            return _error("Feature and exclusion lists overlap.", overlapping_columns=overlap)
        if time_column and time_column in feature_columns:
            return _error("The temporal split column cannot also be a predictive feature.")
        if group_column and group_column in feature_columns:
            return _error("The group column is evaluation metadata and cannot be a predictive feature.")

    warnings: list[str] = []
    exclusions: list[dict[str, str]] = []
    rows_received = int(len(df))
    working = df.loc[df[target_column].notna()].copy()
    dropped_target = rows_received - len(working)
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return _error("Regression requires a numeric target column.")
        finite_target = np.isfinite(working[target_column].astype(float))
        nonfinite_target = int((~finite_target).sum())
        if nonfinite_target:
            working = working.loc[finite_target].copy()
            dropped_target += nonfinite_target
    if dropped_target:
        warnings.append(f"Dropped {dropped_target} row(s) with missing or non-finite target values.")
    dropped_group = 0
    if group_column is not None:
        dropped_group = int(working[group_column].isna().sum())
        if dropped_group:
            working = working.loc[working[group_column].notna()].copy()
            warnings.append(
                f"Dropped {dropped_group} row(s) with missing group values because group isolation "
                "cannot be guaranteed for an unknown entity."
            )
        if working.empty:
            return _error(f"Group column '{group_column}' has no usable values.")
        try:
            working[group_column].map(hash)
            unique_group_count = int(working[group_column].nunique())
        except (TypeError, ValueError):
            return _error(f"Group column '{group_column}' contains unusable group values.")
        if unique_group_count < MIN_UNIQUE_GROUPS:
            return _error(
                f"Grouped evaluation requires at least {MIN_UNIQUE_GROUPS} unique groups.",
                unique_groups=unique_group_count,
            )
    if len(working) < MIN_ML_ROWS:
        return _error(f"At least {MIN_ML_ROWS} usable target rows are required.", rows_usable=int(len(working)))
    if len(working) < 100:
        warnings.append("Fewer than 100 usable rows are available; test metrics may be unstable.")
    if working[target_column].nunique(dropna=True) < 2:
        return _error("The target must contain at least two distinct values.")

    if task_type == "classification":
        class_counts = working[target_column].value_counts(dropna=False, sort=False)
        if len(class_counts) > MAX_CLASSES:
            return _error(f"Classification supports at most {MAX_CLASSES} target classes.")
        if pd.api.types.is_numeric_dtype(working[target_column]) and len(class_counts) > MAX_CLASSES:
            return _error("The numeric target has too many values for classification.")
        if int(class_counts.min()) < 2 and split_strategy == "random":
            return _error("Every classification class needs at least two rows for a stratified split.")
        minority_share = float(class_counts.min() / class_counts.sum())
        if minority_share < 0.10:
            warnings.append("The minority class is below 10% of usable rows; accuracy may be misleading.")
    else:
        class_counts = None

    profile = build_dataset_profile(working)
    identifier_columns = {item["column"] for item in profile.get("potential_identifiers", [])}
    datetime_columns = set(profile.get("datetime_columns", []))
    constant_columns = set(profile.get("constant_columns", []))
    all_null_columns = set(profile.get("all_null_columns", []))

    candidates = feature_columns if feature_columns is not None else [str(column) for column in df.columns]
    if feature_columns is not None and len(candidates) > MAX_RAW_FEATURES:
        return _error(f"ML V1 supports at most {MAX_RAW_FEATURES} requested raw features.")
    selected = []
    for column in candidates:
        reason = None
        if column == target_column:
            reason = "target_column"
        elif column in exclude_columns:
            reason = "explicitly_excluded"
        elif column == time_column:
            reason = "temporal_split_column"
        elif column == group_column:
            reason = "evaluation_group_column"
        elif column in identifier_columns:
            if feature_columns is not None:
                return _error(f"Identifier-like column '{column}' cannot be used as an explicit feature.")
            reason = "identifier_like"
        elif column in datetime_columns:
            reason = "datetime_unsupported"
        elif column in all_null_columns:
            reason = "all_null"
        elif column in constant_columns:
            reason = "constant"
        else:
            series = working[column]
            supported = (
                pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            )
            if not supported:
                reason = "unsupported_dtype"
            elif not pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > MAX_CATEGORY_CARDINALITY:
                reason = "high_cardinality"
        if reason:
            exclusions.append({"column": column, "reason": reason})
        else:
            selected.append(column)

    if feature_columns is not None:
        unsafe = [item for item in exclusions if item["reason"] not in {"explicitly_excluded"}]
        if unsafe:
            return _error("One or more explicit features are unsafe or unsupported.", rejected_features=unsafe)
    if not selected:
        return _error("No eligible predictive features remain after safety exclusions.", features_excluded=exclusions)
    if len(selected) > MAX_RAW_FEATURES:
        return _error(f"ML V1 supports at most {MAX_RAW_FEATURES} eligible raw features.")

    target = working[target_column]
    for column in selected:
        feature = working[column]
        comparable = feature.notna() & target.notna()
        if comparable.any() and feature.loc[comparable].astype(str).equals(target.loc[comparable].astype(str)):
            return _error(f"Feature '{column}' exactly duplicates the target and would cause leakage.")

    numeric_columns = [
        column for column in selected
        if pd.api.types.is_numeric_dtype(working[column]) and not pd.api.types.is_bool_dtype(working[column])
    ]
    categorical_columns = [column for column in selected if column not in numeric_columns]
    estimated_encoded = len(numeric_columns) + sum(
        max(1, int(working[column].nunique(dropna=True))) for column in categorical_columns
    )
    if estimated_encoded > MAX_ENCODED_FEATURES:
        return _error(
            f"Estimated encoded feature count {estimated_encoded} exceeds the V1 limit of {MAX_ENCODED_FEATURES}."
        )

    features = working[selected].copy()
    infinite_count = 0
    for column in numeric_columns:
        values = features[column].astype(float)
        infinite = np.isinf(values)
        infinite_count += int(infinite.sum())
        if infinite.any():
            features.loc[infinite, column] = np.nan
    if infinite_count:
        warnings.append(f"Converted {infinite_count} infinite numeric feature value(s) to missing before imputation.")

    for column in numeric_columns:
        paired = pd.concat([features[column], target], axis=1).dropna()
        if len(paired) >= 3 and pd.api.types.is_numeric_dtype(target):
            correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            if pd.notna(correlation) and abs(float(correlation)) >= 0.995:
                warnings.append(f"Feature '{column}' has a near-perfect target relationship; investigate possible leakage.")

    entity_candidates = _repeated_entity_candidates(
        working, {target_column, *(set([time_column, group_column]) - {None})}
    )
    prediction_unit_warnings: list[str] = []
    if group_column is None and split_strategy == "random":
        for candidate in entity_candidates:
            warning = (
                f"Repeated entities were detected in column '{candidate['column']}'. Row-level random "
                "splitting may place the same entity in both training and test sets and may overestimate "
                f"generalization. Consider specifying group_column='{candidate['column']}' if predictions "
                "are intended to generalize to unseen entities."
            )
            prediction_unit_warnings.append(warning)
            warnings.append(warning)

    group_metadata = {
        "group_column": group_column,
        "group_aware": False,
        "total_groups": None,
        "train_groups": None,
        "test_groups": None,
        "group_overlap_count": None,
    }

    if group_column is not None:
        from sklearn.model_selection import GroupShuffleSplit

        groups = working[group_column]
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=random_state)
        train_index, test_index = next(splitter.split(features, target, groups=groups))
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        train_group_values = set(groups.iloc[train_index].tolist())
        test_group_values = set(groups.iloc[test_index].tolist())
        overlap = train_group_values & test_group_values
        if overlap:
            return _error("Internal grouped-split safety failure: train and test groups overlap.")
        if not train_group_values or not test_group_values:
            return _error("Grouped evaluation must produce at least one train group and one test group.")
        group_metadata = {
            "group_column": group_column,
            "group_aware": True,
            "total_groups": int(len(train_group_values) + len(test_group_values)),
            "train_groups": int(len(train_group_values)),
            "test_groups": int(len(test_group_values)),
            "group_overlap_count": 0,
        }
        if task_type == "classification":
            missing_train = set(target.unique()) - set(y_train.unique())
            if missing_train:
                return _error("Grouped training data does not contain every target class.")
            missing_test = set(target.unique()) - set(y_test.unique())
            if missing_test:
                warnings.append("The grouped test split does not contain every target class; some metrics are conditional.")
    elif split_strategy == "temporal":
        valid_time = working[time_column].notna()
        missing_time = int((~valid_time).sum())
        if missing_time:
            warnings.append(f"Dropped {missing_time} row(s) with missing temporal split values.")
            working = working.loc[valid_time]
            features = features.loc[valid_time]
            target = target.loc[valid_time]
        order = working[time_column].sort_values(kind="stable").index
        features = features.loc[order]
        target = target.loc[order]
        test_rows = max(1, int(math.ceil(len(features) * float(test_size))))
        train_rows = len(features) - test_rows
        if train_rows < 2 or test_rows < 1:
            return _error("Temporal split leaves insufficient train or test rows.")
        X_train, X_test = features.iloc[:train_rows], features.iloc[train_rows:]
        y_train, y_test = target.iloc[:train_rows], target.iloc[train_rows:]
        if task_type == "classification":
            missing_train = set(target.unique()) - set(y_train.unique())
            if missing_train:
                return _error("Temporal training data does not contain every target class.")
            missing_test = set(target.unique()) - set(y_test.unique())
            if missing_test:
                warnings.append("The temporal test split does not contain every target class; some metrics are conditional.")
    else:
        from sklearn.model_selection import train_test_split

        stratify = target if task_type == "classification" else None
        if task_type == "classification":
            expected_test_rows = int(math.ceil(len(target) * float(test_size)))
            if expected_test_rows < len(class_counts) or len(target) - expected_test_rows < len(class_counts):
                return _error("The requested test size cannot represent every classification class.")
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=float(test_size),
            random_state=random_state,
            stratify=stratify,
        )

    if task_type == "classification":
        test_counts = y_test.value_counts()
        if not test_counts.empty and int(test_counts.min()) < 5:
            warnings.append("At least one classification class has fewer than five test rows.")

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if task_type == "classification":
        estimators = [
            ("dummy_most_frequent", True, DummyClassifier(strategy="most_frequent")),
            ("logistic_regression", False, LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    else:
        estimators = [
            ("dummy_mean", True, DummyRegressor(strategy="mean")),
            ("ridge", False, Ridge()),
        ]

    model_results = []
    predictive_pipeline = None
    classes = np.asarray(sorted(target.unique().tolist(), key=lambda value: str(value))) if task_type == "classification" else None
    for name, baseline, estimator in estimators:
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if task_type == "classification":
            probabilities = None
            if len(classes) == 2 and hasattr(pipeline, "predict_proba"):
                model_classes = list(pipeline.named_steps["model"].classes_)
                positive_index = model_classes.index(classes[-1])
                probabilities = pipeline.predict_proba(X_test)[:, positive_index]
            metrics = _classification_metrics(y_test, predictions, probabilities, classes)
        else:
            metrics = _regression_metrics(y_test, predictions)
        model_results.append({"name": name, "baseline": baseline, "metrics": metrics})
        if not baseline:
            predictive_pipeline = pipeline

    baseline_result, predictive_result = model_results
    if task_type == "classification":
        best_model = (
            predictive_result["name"]
            if predictive_result["metrics"]["macro_f1"] > baseline_result["metrics"]["macro_f1"]
            else baseline_result["name"]
        )
        selection_metric = "macro_f1 (higher is better; ties select the baseline)"
        target_summary = {
            "classes": [str(value) for value in classes],
            "class_balance": {
                str(label): {"count": int(count), "percentage": _rounded(100.0 * count / len(target))}
                for label, count in target.value_counts(sort=False).items()
            },
        }
    else:
        best_model = (
            predictive_result["name"]
            if predictive_result["metrics"]["rmse"] < baseline_result["metrics"]["rmse"]
            else baseline_result["name"]
        )
        selection_metric = "rmse (lower is better; ties select the baseline)"
        target_summary = {
            "min": _rounded(target.min()), "max": _rounded(target.max()),
            "mean": _rounded(target.mean()), "median": _rounded(target.median()),
        }

    associations = _feature_associations(predictive_pipeline, numeric_columns, categorical_columns)
    if exclusions:
        grouped_reasons = ", ".join(
            f"{item['column']} ({item['reason']})" for item in exclusions[:20]
        )
        warnings.append(f"Excluded unsafe or unsupported feature columns: {grouped_reasons}.")
    warnings.append("Metrics use one held-out test split and may vary on new data.")

    return {
        "task_type": task_type,
        "target_column": target_column,
        "rows_received": rows_received,
        "rows_used": int(len(target)),
        "rows_dropped": int(rows_received - len(target)),
        "features_used": selected,
        "features_excluded": exclusions[:50],
        "estimated_encoded_features": estimated_encoded,
        "split": {
            "strategy": split_strategy,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": float(test_size),
            "random_state": random_state,
            "time_column": time_column,
            **group_metadata,
        },
        "prediction_unit": {
            "group_column": group_column,
            "evaluation_unit": "group" if group_column is not None else "row",
            "repeated_entity_candidates": entity_candidates,
            "warnings": prediction_unit_warnings[:MAX_ENTITY_CANDIDATES],
        },
        "target_summary": target_summary,
        "models": model_results,
        "best_model": best_model,
        "selection_metric": selection_metric,
        "feature_associations": associations,
        "warnings": _unique(warnings)[:30],
        "limitations": [
            "Performance is estimated from one held-out split, not external validation.",
            "Feature associations are predictive and do not establish causation.",
            "Automated checks cannot detect every form of semantic target leakage.",
            "Feature validity depends on whether each supplied column would be available at prediction time.",
        ],
    }


TRAIN_ML_MODEL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "train_ml_model",
        "description": (
            "Evaluate bounded classification or regression with an explicit target and task; never invent them. "
            "Omit features for safe selection. group_column isolates explicitly named unseen entities; ask if "
            "ambiguous. Returns held-out metrics and non-causal associations, not models or predictions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Named dataset to model; optional for one dataset."},
                "target_column": {"type": "string", "description": "Explicit column to predict."},
                "task_type": {"type": "string", "enum": ["classification", "regression"]},
                "feature_columns": {"type": ["array", "null"], "items": {"type": "string"}},
                "exclude_columns": {"type": ["array", "null"], "items": {"type": "string"}},
                "test_size": {"type": "number", "description": "Held-out fraction from 0.1 through 0.4; defaults to 0.2."},
                "random_state": {"type": "integer", "description": "Deterministic split/model seed; defaults to 42."},
                "split_strategy": {"type": "string", "enum": ["random", "temporal"]},
                "time_column": {"type": ["string", "null"], "description": "Required only for an explicit temporal split."},
                "group_column": {
                    "type": ["string", "null"],
                    "description": (
                        "Entity metadata for group-isolated random evaluation (minimum 5 groups); excluded from features."
                    ),
                },
            },
            "required": ["target_column", "task_type"],
        },
    },
}
