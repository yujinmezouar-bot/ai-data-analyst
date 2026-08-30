from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    id: str
    step_id: str
    tool_name: str
    datasets: List[str] = field(default_factory=list)
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


class FindingsStore:
    """In-memory, deterministic store for execution findings.

    This is intentionally small and non-persistent. It captures tool output,
    dataset provenance, and metadata so future reporting/ML components can
    consume structured findings without relying on the LLM.
    """

    def __init__(self) -> None:
        self._findings: List[Finding] = []

    def record(self, finding: Finding | Dict[str, Any]) -> Finding:
        if isinstance(finding, Finding):
            normalized = finding
        elif isinstance(finding, dict):
            normalized = Finding(
                id=str(finding.get("id") or "finding_" + str(len(self._findings) + 1)),
                step_id=str(finding.get("step_id", "")),
                tool_name=str(finding.get("tool_name", "")),
                datasets=list(finding.get("datasets", []) or []),
                result=finding.get("result"),
                metadata=dict(finding.get("metadata", {}) or {}),
                provenance=dict(finding.get("provenance", {}) or {}),
            )
        else:
            raise TypeError("Finding must be a Finding instance or dict")

        self._findings.append(normalized)
        return normalized

    def get(self, finding_id: str) -> Optional[Finding]:
        for finding in self._findings:
            if finding.id == finding_id:
                return finding
        return None

    def all(self) -> List[Finding]:
        return list(self._findings)

    def find_by_step(self, step_id: str) -> List[Finding]:
        return [finding for finding in self._findings if finding.step_id == step_id]

    def find_by_tool(self, tool_name: str) -> List[Finding]:
        return [finding for finding in self._findings if finding.tool_name == tool_name]

    def clear(self) -> None:
        self._findings.clear()

    def __len__(self) -> int:
        return len(self._findings)

    def __iter__(self):
        return iter(self._findings)
