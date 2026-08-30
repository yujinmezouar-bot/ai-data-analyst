from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from autonomous.plan import AnalysisPlan, PlanStep
from autonomous.results import Finding, FindingsStore
from tools.join_datasets import execute_join, inspect_join_viability
from tools.relationship_discovery import discover_relationships


class ExecutorError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        step_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.step_id = step_id
        self.tool_name = tool_name
        self.original_error = original_error


class Executor:
    """Execute a validated plan by calling deterministic tools directly.

    This version remains conservative and read-only by default. The only
    approved mutation is an explicit safe join workflow using the project’s
    existing join tools.
    """

    DEFAULT_MAX_STEPS = 10
    SAFE_MUTATING_TOOLS = {"execute_join"}

    def __init__(
        self,
        tools_registry: Mapping[str, Callable[..., Any]],
        findings_store: Optional[FindingsStore] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        derived_dataset_register: Optional[Callable[[str, Any], Optional[str]]] = None,
        tool_schemas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.tools_registry = dict(tools_registry)
        self.findings_store = findings_store or FindingsStore()
        self.max_steps = int(max_steps)
        self.derived_dataset_register = derived_dataset_register
        self.tool_schemas = {
            schema.get("function", {}).get("name"): schema.get("function", {}).get("parameters", {})
            for schema in (tool_schemas or [])
            if schema.get("function", {}).get("name")
        }
        self.diagnostic_stage: Optional[str] = None
        self.diagnostic_step_id: Optional[str] = None

    @staticmethod
    def _matches_json_type(value: Any, expected: str) -> bool:
        checks = {
            "string": lambda: isinstance(value, str),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": lambda: isinstance(value, bool),
            "array": lambda: isinstance(value, list),
            "object": lambda: isinstance(value, dict),
            "null": lambda: value is None,
        }
        return checks.get(expected, lambda: True)()

    def _validate_schema_arguments(self, step: PlanStep) -> None:
        schema = self.tool_schemas.get(step.tool_name)
        if not schema:
            return
        properties = schema.get("properties", {})
        missing = [name for name in schema.get("required", []) if name not in step.kwargs]
        if missing:
            raise ExecutorError(
                f"Step is missing required argument(s): {missing}.",
                step_id=step.id, tool_name=step.tool_name,
            )
        unknown = [name for name in step.kwargs if name not in properties]
        if unknown:
            raise ExecutorError(
                f"Step contains unknown argument(s): {unknown}.",
                step_id=step.id, tool_name=step.tool_name,
            )
        for name, value in step.kwargs.items():
            spec = properties.get(name, {})
            expected = spec.get("type")
            expected_types = expected if isinstance(expected, list) else [expected]
            if expected and not any(self._matches_json_type(value, item) for item in expected_types):
                raise ExecutorError(
                    f"Argument '{name}' has invalid type; expected {expected}.",
                    step_id=step.id, tool_name=step.tool_name,
                )
            if "enum" in spec and value not in spec["enum"]:
                raise ExecutorError(
                    f"Argument '{name}' must be one of {spec['enum']}.",
                    step_id=step.id, tool_name=step.tool_name,
                )

    @staticmethod
    def _joined_columns(
        left_columns: set[str], right_columns: set[str], left_on: str, right_on: str
    ) -> set[str]:
        common = left_columns & right_columns
        output: set[str] = set()
        for column in left_columns:
            if column == left_on:
                output.add(column)
            elif column in common:
                output.add(f"{column}_x")
            else:
                output.add(column)
        for column in right_columns:
            if left_on == right_on and column == right_on:
                continue
            if column in common:
                output.add(f"{column}_y")
            else:
                output.add(column)
        return output

    def _validate_plan(self, plan: AnalysisPlan) -> None:
        if len(plan.steps) > self.max_steps:
            raise ExecutorError(
                f"Plan exceeds the maximum allowed execution steps ({self.max_steps}).",
                step_id=getattr(plan.steps[-1], "id", None),
            )

    def preflight(self, plan: AnalysisPlan, datasets: Mapping[str, Any]) -> None:
        """Validate the complete plan structure before any tool executes."""
        self._validate_plan(plan)
        if sum(step.tool_name == "train_ml_model" for step in plan.steps) > 1:
            raise ExecutorError("Autonomous plans may contain at most one ML training step.")
        available = {
            name: {str(column) for column in dataframe.columns}
            for name, dataframe in datasets.items()
        }
        internal_tools = {"discover_relationships", "inspect_join_viability", "execute_join"}

        for dataset_name in plan.datasets:
            if dataset_name not in available:
                raise ExecutorError(f"Dataset '{dataset_name}' is not available for execution.")

        for step in plan.steps:
            if step.tool_name not in internal_tools and step.tool_name not in self.tools_registry:
                raise ExecutorError(
                    f"Tool '{step.tool_name}' is not available in the registry.",
                    step_id=step.id, tool_name=step.tool_name,
                )
            if step.tool_name not in self.SAFE_MUTATING_TOOLS and not step.read_only:
                raise ExecutorError(
                    "Only read-only steps are supported by the executor.",
                    step_id=step.id, tool_name=step.tool_name,
                )
            self._validate_schema_arguments(step)

            if step.tool_name == "train_ml_model":
                dataset_name = step.kwargs.get("dataset_name") or step.kwargs.get("dataset")
                if dataset_name is None:
                    if len(available) != 1:
                        raise ExecutorError(
                            "ML step must name a dataset when multiple datasets are available.",
                            step_id=step.id, tool_name=step.tool_name,
                        )
                    dataset_name = next(iter(available))
                if dataset_name not in available:
                    raise ExecutorError(
                        f"Dataset '{dataset_name}' is not available for execution.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                columns = available[dataset_name]
                target = step.kwargs.get("target_column")
                features = step.kwargs.get("feature_columns") or []
                exclusions = step.kwargs.get("exclude_columns") or []
                time_column = step.kwargs.get("time_column")
                group_column = step.kwargs.get("group_column")
                missing_columns = [
                    column for column in [target, *features, *exclusions, time_column, group_column]
                    if column is not None and column not in columns
                ]
                if missing_columns:
                    raise ExecutorError(
                        f"ML step references unavailable column(s): {sorted(set(missing_columns))}.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if target in features:
                    raise ExecutorError(
                        "ML target cannot also appear in feature_columns.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if group_column == target:
                    raise ExecutorError(
                        "ML group_column cannot equal target_column.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if group_column is not None and group_column == time_column:
                    raise ExecutorError(
                        "ML group_column and time_column must be different.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if group_column in features:
                    raise ExecutorError(
                        "ML group_column cannot appear in feature_columns.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                overlap = sorted(set(features) & set(exclusions))
                if overlap:
                    raise ExecutorError(
                        f"ML feature/exclusion lists overlap: {overlap}.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                test_size = step.kwargs.get("test_size", 0.2)
                if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not 0.1 <= test_size <= 0.4:
                    raise ExecutorError(
                        "ML test_size must be between 0.1 and 0.4.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                split_strategy = step.kwargs.get("split_strategy", "random")
                if split_strategy == "temporal" and not time_column:
                    raise ExecutorError(
                        "Temporal ML split requires time_column.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if split_strategy == "temporal" and group_column is not None:
                    raise ExecutorError(
                        "ML V1 does not support a combined grouped-temporal split.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if split_strategy == "random" and time_column is not None:
                    raise ExecutorError(
                        "time_column may only be used with a temporal ML split.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if time_column is not None and time_column in features:
                    raise ExecutorError(
                        "Temporal split column cannot also be an ML feature.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                continue

            if step.tool_name in {"inspect_join_viability", "execute_join"}:
                left_name = step.kwargs.get("left_dataset")
                right_name = step.kwargs.get("right_dataset")
                if left_name not in available or right_name not in available:
                    missing = [name for name in (left_name, right_name) if name not in available]
                    raise ExecutorError(
                        f"Dataset(s) not available for join execution: {missing}",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                left_on = step.kwargs.get("left_on")
                right_on = step.kwargs.get("right_on")
                if left_on not in available[left_name] or right_on not in available[right_name]:
                    raise ExecutorError(
                        "Join step references a column that is not available.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                if step.tool_name == "execute_join":
                    derived_name = next(
                        f"derived_join_{index}" for index in range(1, 1000)
                        if f"derived_join_{index}" not in available
                    )
                    available[derived_name] = self._joined_columns(
                        available[left_name], available[right_name], left_on, right_on
                    )
                continue

            if step.tool_name == "discover_relationships":
                target_name = step.kwargs.get("dataset_name") or step.kwargs.get("dataset")
                if target_name is not None and target_name not in available:
                    raise ExecutorError(
                        f"Dataset '{target_name}' is not available for execution.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                continue

            if step.tool_name == "create_multi_dataset_visualization":
                for series in step.kwargs.get("series", []):
                    name = series.get("dataset_name") if isinstance(series, dict) else None
                    if name not in available:
                        raise ExecutorError(
                            f"Dataset '{name}' is not available for execution.",
                            step_id=step.id, tool_name=step.tool_name,
                        )
                    for key in ("x_column", "y_column"):
                        column = series.get(key)
                        if column is not None and column not in available[name]:
                            raise ExecutorError(
                                f"Column '{column}' is not available in dataset '{name}'.",
                                step_id=step.id, tool_name=step.tool_name,
                            )
                continue

            dataset_name = step.kwargs.get("dataset_name") or step.kwargs.get("dataset")
            if dataset_name is None:
                if len(available) != 1:
                    raise ExecutorError(
                        "Step does not reference a dataset and multiple datasets are available.",
                        step_id=step.id, tool_name=step.tool_name,
                    )
                dataset_name = next(iter(available))
            if dataset_name not in available:
                raise ExecutorError(
                    f"Dataset '{dataset_name}' is not available for execution.",
                    step_id=step.id, tool_name=step.tool_name,
                )
            for argument, value in step.kwargs.items():
                if argument == "column" or argument.endswith("_column"):
                    if value is not None and value not in available[dataset_name]:
                        raise ExecutorError(
                            f"Column '{value}' is not available in dataset '{dataset_name}'.",
                            step_id=step.id, tool_name=step.tool_name,
                        )
                elif argument == "columns" and isinstance(value, list):
                    missing_columns = [column for column in value if column not in available[dataset_name]]
                    if missing_columns:
                        raise ExecutorError(
                            f"Column(s) {missing_columns} are not available in dataset '{dataset_name}'.",
                            step_id=step.id, tool_name=step.tool_name,
                        )

    def _dataset_name_for_step(self, step: PlanStep, datasets: Mapping[str, Any]) -> str:
        dataset_name = step.kwargs.get("dataset_name") or step.kwargs.get("dataset")
        if dataset_name is None:
            if len(datasets) == 1:
                dataset_name = next(iter(datasets.keys()))
            else:
                raise ExecutorError(
                    "Step does not reference a dataset and multiple datasets are available.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )
        if dataset_name not in datasets:
            raise ExecutorError(
                f"Dataset '{dataset_name}' is not available for execution.",
                step_id=step.id,
                tool_name=step.tool_name,
            )
        return str(dataset_name)

    def _record_finding(
        self,
        plan: AnalysisPlan,
        step: PlanStep,
        *,
        datasets: list[str],
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Finding:
        adaptive_context = {
            key: plan.constraints[key]
            for key in ("parent_plan_id", "adaptive_round", "reviewer_reason")
            if key in plan.constraints
        }
        finding = Finding(
            id=f"finding_{len(self.findings_store.all()) + 1}",
            step_id=step.id,
            tool_name=step.tool_name,
            datasets=list(datasets),
            result=result,
            metadata={
                "plan_id": plan.id,
                "objective": plan.objective,
                "read_only": step.read_only,
                "outputs_expected": list(step.outputs_expected),
                **adaptive_context,
                **(metadata or {}),
            },
            provenance={
                "step_id": step.id,
                "tool_name": step.tool_name,
                "dataset_names": list(datasets),
                "plan_id": plan.id,
                **adaptive_context,
                **(provenance or {}),
            },
        )
        self.findings_store.record(finding)
        return finding

    def _resolve_join_pair(self, step: PlanStep, datasets: Mapping[str, Any]) -> tuple[str, str, Any, Any]:
        left_name = step.kwargs.get("left_dataset")
        right_name = step.kwargs.get("right_dataset")
        if left_name is None or right_name is None:
            raise ExecutorError(
                "Join steps must specify both left_dataset and right_dataset.",
                step_id=step.id,
                tool_name=step.tool_name,
            )
        if left_name not in datasets or right_name not in datasets:
            missing = [name for name in (left_name, right_name) if name not in datasets]
            raise ExecutorError(
                f"Dataset(s) not available for join execution: {missing}",
                step_id=step.id,
                tool_name=step.tool_name,
            )
        return str(left_name), str(right_name), datasets[left_name], datasets[right_name]

    def _execute_step(self, plan: AnalysisPlan, step: PlanStep, working_datasets: dict[str, Any]) -> Any:
        # Allow internal builtin analysis tools to be handled even when they are not
        # present in the external tools_registry. External registry tools are still
        # required to be present.
        INTERNAL_BUILTIN_TOOLS = {"discover_relationships", "inspect_join_viability", "execute_join"}

        if step.tool_name not in INTERNAL_BUILTIN_TOOLS and step.tool_name not in self.tools_registry:
            raise ExecutorError(
                f"Tool '{step.tool_name}' is not available in the registry.",
                step_id=step.id,
                tool_name=step.tool_name,
            )

        if step.tool_name not in self.SAFE_MUTATING_TOOLS and not step.read_only:
            raise ExecutorError(
                "Only read-only steps are supported by the executor.",
                step_id=step.id,
                tool_name=step.tool_name,
            )

        if step.tool_name == "discover_relationships":
            dataset_name = step.kwargs.get("dataset_name") or step.kwargs.get("dataset")
            dataset_map = dict(working_datasets)
            if dataset_name is not None:
                if dataset_name not in dataset_map:
                    raise ExecutorError(
                        f"Dataset '{dataset_name}' is not available for execution.",
                        step_id=step.id,
                        tool_name=step.tool_name,
                    )
                dataset_map = {dataset_name: dataset_map[dataset_name]}
            result = discover_relationships(
                dataset_map,
                target_dataset=dataset_name,
                min_confidence=step.kwargs.get("min_confidence", 0.4),
            )
            self._record_finding(
                plan,
                step,
                datasets=list(dataset_map.keys()),
                result=result,
                metadata={"target_dataset": dataset_name, "min_confidence": step.kwargs.get("min_confidence", 0.4)},
                provenance={"category": "relationship_discovery"},
            )
            return result

        if step.tool_name == "inspect_join_viability":
            left_name, right_name, left_df, right_df = self._resolve_join_pair(step, working_datasets)
            left_on = step.kwargs.get("left_on")
            right_on = step.kwargs.get("right_on")
            if left_on is None or right_on is None:
                raise ExecutorError(
                    "inspect_join_viability requires both left_on and right_on.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )
            result = inspect_join_viability(left_df, right_df, left_on, right_on)
            self._record_finding(
                plan,
                step,
                datasets=[left_name, right_name],
                result=result,
                metadata={"left_on": left_on, "right_on": right_on},
                provenance={"category": "join_viability"},
            )
            return result

        if step.tool_name == "execute_join":
            left_name, right_name, left_df, right_df = self._resolve_join_pair(step, working_datasets)
            left_on = step.kwargs.get("left_on")
            right_on = step.kwargs.get("right_on")
            if left_on is None or right_on is None:
                raise ExecutorError(
                    "execute_join requires both left_on and right_on.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )

            viability = inspect_join_viability(left_df, right_df, left_on, right_on)
            # Perform (and record) a pre-flight viability check. If the plan already
            # contained an explicit inspect_join_viability step earlier, that finding
            # will already exist; but when execute_join is run standalone we must
            # check viability first and record the result.
            viability = inspect_join_viability(left_df, right_df, left_on, right_on)

            # Record the pre-check as its own finding so downstream consumers can see it
            pre_step = PlanStep(id=f"{step.id}_inspect", tool_name="inspect_join_viability", kwargs={}, read_only=True)
            self._record_finding(
                plan,
                pre_step,
                datasets=[left_name, right_name],
                result=viability,
                metadata={"left_on": left_on, "right_on": right_on},
                provenance={"category": "join_viability", "auto_precheck": True},
            )

            if not viability.get("safe_to_join", False):
                raise ExecutorError(
                    f"Join rejected by inspect_join_viability: {viability.get('error', 'unsafe cardinality or incompatible keys')}",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )

            how = step.kwargs.get("how", "inner")
            result = execute_join(left_df, right_df, left_on, right_on, how=how)
            if isinstance(result, dict) and "error" in result:
                raise ExecutorError(
                    f"Join execution failed: {result['error']}",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )

            joined_df = result.get("dataframe")
            if joined_df is None:
                raise ExecutorError(
                    "Join execution returned no DataFrame.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )

            derived_name = next(
                (
                    f"derived_join_{idx}"
                    for idx in range(1, 1000)
                    if f"derived_join_{idx}" not in working_datasets
                ),
                None,
            )
            if derived_name is None:
                raise ExecutorError(
                    "No valid derived dataset name was available.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                )

            canonical_name = derived_name
            if self.derived_dataset_register is not None:
                registered_name = self.derived_dataset_register(derived_name, joined_df)
                # Preserve legacy callbacks that return None while accepting
                # canonical names from owning dataset stores.
                if isinstance(registered_name, str) and registered_name:
                    canonical_name = registered_name

            working_datasets[canonical_name] = joined_df
            if canonical_name != derived_name:
                for later_step in plan.steps:
                    for key in ("dataset_name", "dataset", "left_dataset", "right_dataset"):
                        if later_step.kwargs.get(key) == derived_name:
                            later_step.kwargs[key] = canonical_name

            payload = {
                "status": result.get("status", "success"),
                "shape": result.get("shape"),
                "columns": result.get("columns"),
                "cardinality": result.get("cardinality"),
                "dataset_name": canonical_name,
            }
            self._record_finding(
                plan,
                step,
                datasets=[left_name, right_name, canonical_name],
                result=payload,
                metadata={"left_on": left_on, "right_on": right_on, "how": how},
                provenance={"category": "join_execution", "derived_dataset": canonical_name},
            )
            return payload

        dataset_name = self._dataset_name_for_step(step, working_datasets)
        tool = self.tools_registry[step.tool_name]
        tool_kwargs = dict(step.kwargs)
        tool_kwargs.pop("dataset_name", None)
        tool_kwargs.pop("dataset", None)
        result = tool(working_datasets[dataset_name], **tool_kwargs)
        self._record_finding(
            plan,
            step,
            datasets=[dataset_name],
            result=result,
            metadata={"dataset_name": dataset_name},
            provenance={"category": "analysis_tool"},
        )
        return result

    def execute(self, plan: AnalysisPlan, datasets: Mapping[str, Any]) -> FindingsStore:
        self.diagnostic_stage = "preflight"
        self.diagnostic_step_id = None
        self.preflight(plan, datasets)
        self.diagnostic_stage = "execution"
        working_datasets = dict(datasets)

        for step in plan.steps:
            self.diagnostic_step_id = step.id
            try:
                self._execute_step(plan, step, working_datasets)
            except ExecutorError:
                raise
            except Exception as exc:
                raise ExecutorError(
                    f"Tool '{step.tool_name}' failed during execution.",
                    step_id=step.id,
                    tool_name=step.tool_name,
                    original_error=exc,
                ) from exc

        self.diagnostic_stage = None
        self.diagnostic_step_id = None
        return self.findings_store
