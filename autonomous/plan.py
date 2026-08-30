from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    id: str
    tool_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    read_only: bool = True
    outputs_expected: List[str] = field(default_factory=list)
    category: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PlanStep":
        if "id" not in d or not isinstance(d["id"], str) or not d["id"].strip():
            raise ValueError("PlanStep missing non-empty 'id'")
        if "tool_name" not in d or not isinstance(d["tool_name"], str) or not d["tool_name"].strip():
            raise ValueError("PlanStep missing non-empty 'tool_name'")

        kwargs = d.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise ValueError("PlanStep 'kwargs' must be an object/dict")

        read_only = bool(d.get("read_only", True))
        outputs_expected = d.get("outputs_expected", []) or []
        if not isinstance(outputs_expected, list):
            raise ValueError("PlanStep 'outputs_expected' must be a list")

        category = d.get("category")

        return PlanStep(
            id=d["id"].strip(),
            tool_name=d["tool_name"].strip(),
            kwargs=kwargs,
            read_only=read_only,
            outputs_expected=list(outputs_expected),
            category=category,
        )


@dataclass
class AnalysisPlan:
    id: str
    objective: str
    datasets: List[str]
    steps: List[PlanStep]
    constraints: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any], max_steps: int = 10, allowed_datasets: Optional[List[str]] = None) -> "AnalysisPlan":
        # Basic shape checks
        if not isinstance(d, dict):
            raise ValueError("Plan must be a JSON object/dict")

        if "id" not in d or not isinstance(d["id"], str) or not d["id"].strip():
            raise ValueError("Plan missing non-empty 'id'")

        if "objective" not in d or not isinstance(d["objective"], str) or not d["objective"].strip():
            raise ValueError("Plan missing non-empty 'objective'")

        datasets = d.get("datasets", []) or []
        if not isinstance(datasets, list):
            raise ValueError("Plan 'datasets' must be a list")

        # If allowed_datasets provided, ensure referenced datasets are subset
        if allowed_datasets is not None:
            for ds in datasets:
                if ds not in allowed_datasets:
                    raise ValueError(f"Unknown dataset referenced in plan: {ds}")

        steps_raw = d.get("steps", []) or []
        if not isinstance(steps_raw, list):
            raise ValueError("Plan 'steps' must be a list")

        if len(steps_raw) > max_steps:
            raise ValueError(f"Plan contains too many steps ({len(steps_raw)}), max is {max_steps}")

        steps: List[PlanStep] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                raise ValueError("Each step must be an object/dict")
            step = PlanStep.from_dict(s)
            steps.append(step)

        constraints = d.get("constraints", {}) or {}
        if not isinstance(constraints, dict):
            raise ValueError("Plan 'constraints' must be an object/dict")

        expected_outputs = d.get("expected_outputs", []) or []
        if not isinstance(expected_outputs, list):
            raise ValueError("Plan 'expected_outputs' must be a list")

        return AnalysisPlan(
            id=d["id"].strip(),
            objective=d["objective"].strip(),
            datasets=[str(x) for x in datasets],
            steps=steps,
            constraints=constraints,
            expected_outputs=list(expected_outputs),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "datasets": list(self.datasets),
            "steps": [
                {
                    "id": s.id,
                    "tool_name": s.tool_name,
                    "kwargs": s.kwargs,
                    "read_only": s.read_only,
                    "outputs_expected": s.outputs_expected,
                    "category": s.category,
                }
                for s in self.steps
            ],
            "constraints": dict(self.constraints),
            "expected_outputs": list(self.expected_outputs),
        }