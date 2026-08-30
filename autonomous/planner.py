from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from agent.llm import LLMProvider
from autonomous.plan import AnalysisPlan, PlanStep


class PlannerError(Exception):
    pass


class AdaptiveReviewError(PlannerError):
    """Review-contract failure carrying only safe structural diagnostics."""

    def __init__(self, message: str, *, failure_stage: str, parsed_metadata: Dict[str, Any] | None = None):
        super().__init__(message)
        self.failure_stage = failure_stage
        self.parsed_metadata = parsed_metadata or {}


TEMPORAL_GROUNDING_GUIDANCE = (
    "Interpret relative analytical periods against the latest populated period in the selected "
    "dataset/date column, never today's date. Latest/current means the latest populated bucket; "
    "previous means the populated bucket immediately before it; last/previous year means the "
    "populated year before the latest year; recent growth/decline compares the latest two populated "
    "periods at the requested granularity. Use period labels exposed in the dataset profile whenever "
    "possible; percentage_change explicit-year arguments remain integer years."
)

TOOL_SELECTION_GUIDANCE = (
    "Use percentage_change for overall period-to-period movement of one KPI series; "
    "groupby_analysis for group levels within one scope; and kpi_contribution_analysis for which "
    "groups declined or grew most between periods, or drove, contributed to, or offset additive KPI "
    "movement. Pair contribution analysis with time_analysis when broader trend context is requested. "
    "Use correlation_analysis only for statistical association; it cannot substitute for contribution "
    "or establish causal explanation. Current tools do not establish causality."
)


class AnalysisPlanner:
    """Convert natural-language requests into validated AnalysisPlan objects.

    The planner depends on an LLMProvider for planning, and may optionally
    validate tool names against a provided tools registry.
    """

    DEFAULT_MAX_STEPS = 10
    MAX_DATASET_CONTEXT_CHARS = 1350
    MAX_RELATIONSHIP_CONTEXT_CHARS = 600
    MAX_REVIEW_FINDINGS_CHARS = 10000

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools_registry: Optional[Dict[str, Any]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        validate_tools: bool = False,
    ) -> None:
        self.llm = llm_provider
        self.tools_registry = tools_registry or {}
        self.max_steps = int(max_steps)
        self.validate_tools = bool(validate_tools)

    def _build_prompt(self, user_request: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Keep the prompt concise — tests mock the provider so content is not executed.
        # In production, this prompt must instruct the LLM to return ONLY JSON.
        datasets = context.get("datasets", [])
        dataset_context = str(context.get("dataset_context", ""))[:self.MAX_DATASET_CONTEXT_CHARS]
        relationship_context = str(context.get("relationship_context", ""))[:self.MAX_RELATIONSHIP_CONTEXT_CHARS]
        tool_specs = []
        for schema in context.get("tool_schemas", []) or []:
            function = schema.get("function", {})
            tool_specs.append({
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
            })

        system = (
            "You are an analysis planner. Return ONLY a valid JSON AnalysisPlan object.\n"
            "Top-level expected_outputs and every step-level outputs_expected MUST be JSON arrays "
            "of strings; use [] when none.\n"
            "Complete JSON shape example (replace d/t with listed names): "
            '{"id":"p","objective":"o","datasets":["d"],'
            '"steps":[{"id":"s","tool_name":"t","kwargs":{},"read_only":true,'
            '"outputs_expected":[]}],"constraints":{},"expected_outputs":[]}\n'
            f"Datasets: {datasets}\n"
            f"Dataset schema context:\n{dataset_context}\n"
            f"Relationship context:\n{relationship_context}\n"
            f"Tool specifications:\n{json.dumps(tool_specs, default=str, separators=(',', ':'))}\n"
            f"Temporal grounding rule: {TEMPORAL_GROUNDING_GUIDANCE}\n"
            f"Tool-selection rule: {TOOL_SELECTION_GUIDANCE}\n"
            "Use only listed dataset and column names. A successful execute_join produces the next "
            "available derived_join_N dataset, which later steps may reference.\n"
            f"Maximum steps: {self.max_steps}\n"
        )

        user = f"User request:\n{user_request}\nProvide a compact machine-readable plan."
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def plan(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> AnalysisPlan:
        context = context or {}
        messages = self._build_prompt(user_request, context)

        try:
            response = self.llm.chat(messages=messages, tools=None, tool_choice="none")
        except Exception as e:
            raise PlannerError(f"LLM provider failed: {e}") from e

        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise PlannerError("LLM provider returned no textual content")

        # Parse JSON (accepting the content must be pure JSON)
        try:
            plan_obj = json.loads(content)
        except Exception as e:
            raise PlannerError(f"LLM returned invalid JSON: {e}") from e

        # Accept only the harmless scalar representation observed in a real
        # provider run. AnalysisPlan remains the strict source of truth for
        # every other type and structure.
        if isinstance(plan_obj, dict):
            expected_outputs = plan_obj.get("expected_outputs")
            if isinstance(expected_outputs, str) and expected_outputs.strip():
                plan_obj = dict(plan_obj)
                plan_obj["expected_outputs"] = [expected_outputs]

        # Validate basic structure and datasets
        allowed_datasets = context.get("datasets") if context and "datasets" in context else None

        try:
            plan = AnalysisPlan.from_dict(plan_obj, max_steps=self.max_steps, allowed_datasets=allowed_datasets)
        except Exception as e:
            raise PlannerError(f"Plan validation failed: {e}") from e

        # Optionally validate tool names in steps
        if self.validate_tools and self.tools_registry:
            for s in plan.steps:
                if s.tool_name not in self.tools_registry:
                    raise PlannerError(f"Unknown tool referenced in plan: {s.tool_name}")

        return plan

    def build_review_prompt(
        self,
        user_request: str,
        compact_findings: Any,
        dataset_context: str,
        tool_schemas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tool_specs = []
        for schema in tool_schemas:
            function = schema.get("function", {})
            tool_specs.append({
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters", {}),
            })
        findings_text = json.dumps(compact_findings, default=str)[:self.MAX_REVIEW_FINDINGS_CHARS]
        system = (
            "Review completed deterministic findings and return ONLY valid JSON. "
            "status must be exactly \"complete\" or \"follow_up\"; reason is always required as a "
            "non-empty string. complete means current findings are sufficient and no more analysis is needed; "
            "normally omit steps. follow_up requires a non-empty steps array with 1-2 entries. "
            "Each step may contain only id (string), tool_name (string), kwargs (object), read_only (boolean), "
            "outputs_expected (array), and optional category. Unknown top-level or step fields are invalid. "
            "Follow-ups must be justified, read-only, and use available datasets, columns, and tools; never "
            "request execute_join or train_ml_model. "
            "Complete example: {\"status\":\"complete\",\"reason\":\"The current findings are sufficient to answer the objective.\"}\n"
            "Follow-up example (replace dataset/column names as needed): "
            "{\"status\":\"follow_up\",\"reason\":\"A bounded statistic is still needed.\",\"steps\":[{\"id\":\"follow_up_1\",\"tool_name\":\"statistics\",\"kwargs\":{\"dataset_name\":\"sales\",\"column\":\"revenue\"},\"read_only\":true,\"outputs_expected\":[\"descriptive statistics\"]}]}"
            f" Temporal grounding rule: {TEMPORAL_GROUNDING_GUIDANCE}"
        )
        user = (
            f"Original question:\n{user_request}\n\n"
            f"Available dataset schemas:\n{dataset_context[:self.MAX_DATASET_CONTEXT_CHARS]}\n\n"
            f"Completed findings:\n{findings_text}\n\n"
            f"Tool specifications:\n{json.dumps(tool_specs, default=str, separators=(',', ':'))}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def parse_review(content: str, max_follow_up_steps: int = 2) -> Dict[str, Any]:
        try:
            review = json.loads(content)
        except Exception as exc:
            raise AdaptiveReviewError(
                f"Adaptive review returned invalid JSON: {exc}", failure_stage="json_parse"
            ) from exc

        if isinstance(review, dict):
            keys = sorted(str(key) for key in review)[:20]
            metadata = {
                "top_level_keys": keys,
                "top_level_types": {
                    str(key): type(value).__name__ for key, value in list(review.items())[:20]
                },
            }
        else:
            metadata = {"top_level_keys": [], "top_level_types": {"$root": type(review).__name__}}

        def invalid(message: str) -> None:
            raise AdaptiveReviewError(
                message, failure_stage="contract_validation", parsed_metadata=metadata
            )

        if not isinstance(review, dict):
            invalid("Adaptive review must be a JSON object")
        allowed_fields = {"status", "reason", "steps"}
        unknown = set(review) - allowed_fields
        if unknown:
            invalid(f"Adaptive review contains unknown fields: {sorted(unknown)}")
        status = review.get("status")
        if status not in {"complete", "follow_up"}:
            invalid("Adaptive review status must be 'complete' or 'follow_up'")
        reason = review.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            invalid("Adaptive review requires a non-empty reason")
        if status == "complete":
            if "steps" in review and review["steps"] not in (None, []):
                invalid("Complete adaptive review must not include non-empty steps")
            return {"status": status, "reason": reason.strip(), "steps": []}

        steps = review.get("steps")
        if not isinstance(steps, list) or not steps:
            invalid("Follow-up adaptive review requires non-empty steps")
        if len(steps) > max_follow_up_steps:
            invalid(f"Adaptive review exceeds the {max_follow_up_steps}-step limit")
        allowed_step_fields = {"id", "tool_name", "kwargs", "read_only", "outputs_expected", "category"}
        normalized_steps = []
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                invalid("Adaptive review steps must be JSON objects")
            step_unknown = set(raw_step) - allowed_step_fields
            if step_unknown:
                invalid(f"Adaptive step contains unknown fields: {sorted(step_unknown)}")
            if "read_only" in raw_step and not isinstance(raw_step["read_only"], bool):
                invalid("Adaptive step 'read_only' must be a boolean")
            try:
                step = PlanStep.from_dict(raw_step)
            except Exception as exc:
                raise AdaptiveReviewError(
                    f"Invalid adaptive step: {exc}",
                    failure_stage="contract_validation",
                    parsed_metadata=metadata,
                ) from exc
            normalized_steps.append({
                "id": step.id,
                "tool_name": step.tool_name,
                "kwargs": step.kwargs,
                "read_only": step.read_only,
                "outputs_expected": step.outputs_expected,
                "category": step.category,
            })
        return {"status": status, "reason": reason.strip(), "steps": normalized_steps}
