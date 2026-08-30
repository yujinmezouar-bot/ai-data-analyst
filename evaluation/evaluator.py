from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.agent import Agent
from agent.llm import DEFAULT_MODEL, LLMClient
from evaluation.benchmark_cases import BENCHMARK_VERSION, BenchmarkCase, benchmark_cases
from evaluation.datasets import build_benchmark_datasets
from evaluation.metrics import summarize, values_match


def _message(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(name: str, arguments: dict[str, Any], index: int) -> Any:
    return SimpleNamespace(
        id=f"benchmark_call_{index}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class ScriptedProvider:
    """Controlled provider for reproducible end-to-end orchestration runs."""

    def __init__(self, case: BenchmarkCase) -> None:
        self.case = case
        self.planning_calls = 0
        self.reactive_calls = 0
        self.tool_batch_sent = False

    def _plan(self) -> dict[str, Any]:
        return {
            "id": f"benchmark_{self.case.id}",
            "objective": self.case.question,
            "datasets": list(self.case.datasets),
            "steps": [
                {"id": f"step_{index}", "tool_name": name, "kwargs": kwargs,
                 "read_only": name != "execute_join", "outputs_expected": []}
                for index, (name, kwargs) in enumerate(self.case.autonomous_steps, 1)
            ],
            "constraints": {"benchmark_case": self.case.id},
        }

    def chat(self, messages, tools=None, tool_choice=None):
        if tools is not None:
            self.reactive_calls += 1
            if self.case.fault == "malformed_planner":
                return _message("Reactive fallback completed.")
            if not self.tool_batch_sent and self.case.scripted_calls:
                self.tool_batch_sent = True
                calls = [_tool_call(name, kwargs, index) for index, (name, kwargs) in enumerate(self.case.scripted_calls, 1)]
                return _message("Running deterministic benchmark tools.", calls)
            direct = "Context is required to resolve the previous result."
            if self.case.id == "ml_ambiguous":
                direct = "Please specify the prediction target and whether the task is classification or regression."
            return _message(direct)

        self.planning_calls += 1
        if self.case.autonomous_steps or self.case.fault in {"malformed_planner", "synthesis_failure"}:
            if self.planning_calls == 1:
                if self.case.fault == "malformed_planner":
                    return _message("not valid planner json")
                return _message(json.dumps(self._plan()))
            if self.planning_calls == 2:
                return _message(json.dumps({"status": "complete", "reason": "Benchmark findings are sufficient."}))
            if self.case.fault == "synthesis_failure":
                raise RuntimeError("Injected benchmark synthesis failure")
        return _message("Completed benchmark analysis using deterministic evidence; associations do not imply causation.")


class _DiagnosticProvider:
    """Transparent real-provider wrapper recording only safe request metadata."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, tool_choice=None):
        call = {
            "index": len(self.calls) + 1,
            "phase": "tool_decision" if tools is not None else "tools_disabled_llm_call",
            "message_count": len(messages),
            "message_chars": sum(len(str(message.get("content") or "")) for message in messages),
            "tool_count": len(tools or []),
            "tool_schema_chars": len(json.dumps(tools, default=str)) if tools is not None else 0,
            "tool_choice": tool_choice,
        }
        self.calls.append(call)
        try:
            response = self.provider.chat(messages, tools=tools, tool_choice=tool_choice)
        except Exception as exc:
            call["outcome"] = "error"
            call["error_type"] = f"{type(exc).__module__}.{type(exc).__name__}"
            raise
        call["outcome"] = "success"
        return response


def _exception_diagnostic(exc: BaseException) -> list[dict[str, Any]]:
    """Return a bounded, secret-safe exception cause chain for benchmark artifacts."""
    diagnostic = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(diagnostic) < 8:
        seen.add(id(current))
        item: dict[str, Any] = {
            "type": f"{type(current).__module__}.{type(current).__name__}",
            "message": str(current)[:1000],
        }
        status_code = getattr(current, "status_code", None)
        request_id = getattr(current, "request_id", None)
        response = getattr(current, "response", None)
        if status_code is not None:
            item["http_status"] = status_code
        if request_id:
            item["request_id"] = str(request_id)[:200]
        elif response is not None:
            header_id = getattr(response, "headers", {}).get("x-request-id")
            if header_id:
                item["request_id"] = str(header_id)[:200]
        diagnostic.append(item)
        current = current.__cause__ or current.__context__
    return diagnostic


def _normalise_result(result: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    if result.get("findings"):
        for finding in result["findings"]:
            evidence.append({
                "tool_name": getattr(finding, "tool_name", ""),
                "datasets": list(getattr(finding, "datasets", []) or []),
                "result": getattr(finding, "result", None),
                "provenance": getattr(finding, "provenance", {}),
            })
    else:
        evidence = list(result.get("evidence") or [])
    return {
        "answer": str(result.get("answer") or ""),
        "trace": list(result.get("trace") or []),
        "evidence": evidence,
        "has_figure": result.get("figure") is not None,
    }


def _path(value: Any, path: str) -> Any:
    current = value
    for part in path.split(".") if path else []:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current


def _evidence_for(observed: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    return [item for item in observed["evidence"] if item.get("tool_name") == tool]


def _actual_mode(trace: list[dict[str, Any]]) -> str:
    routing = next((item for item in trace if item.get("step") == "routing"), None)
    if routing:
        return str(routing.get("decision"))
    return "autonomous" if any(item.get("step") == "final_answer" and item.get("autonomous") for item in trace) else "reactive"


def score_case(case: BenchmarkCase, observed: dict[str, Any], provider_error: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    tools = [item.get("tool_name") for item in observed["evidence"]]

    def add(kind: str, passed: bool, expected: Any, actual: Any) -> None:
        checks.append({"kind": kind, "passed": bool(passed), "expected": expected, "actual": actual})

    if case.expected_mode:
        actual_mode = _actual_mode(observed["trace"])
        add("routing", actual_mode == case.expected_mode, case.expected_mode, actual_mode)
    if case.expected_tools or case.forbidden_tools:
        passed = set(case.expected_tools).issubset(tools) and not set(case.forbidden_tools).intersection(tools)
        add("tool_selection", passed, {"required": case.expected_tools, "forbidden": case.forbidden_tools}, tools)
    for expectation in case.expected_values:
        matches = _evidence_for(observed, expectation.tool)
        actual = _path(matches[0].get("result"), expectation.path) if matches else None
        add("numerical", values_match(actual, expectation.expected, expectation.tolerance), expectation.expected, actual)

    safety_required = bool(case.expected_warning or case.expected_error or case.forbidden_tools or case.category == "ml_safety")
    if safety_required:
        serialized = json.dumps(observed["evidence"], default=str)
        expected_text = case.expected_warning or case.expected_error
        safe = (expected_text.lower() in serialized.lower()) if expected_text else not set(case.forbidden_tools).intersection(tools)
        add("safety", safe, expected_text or case.forbidden_tools, serialized[:1000])

    if case.expected_mode == "autonomous" and case.category != "fallback":
        plan_entry = next((item for item in observed["trace"] if item.get("step") == "autonomous_plan"), None)
        plan = plan_entry.get("plan", {}) if plan_entry else {}
        successful = (
            bool(plan_entry)
            and set(plan.get("datasets", [])).issubset(case.datasets)
            and bool(observed["evidence"])
            and any(item.get("step") == "final_answer" and item.get("autonomous") for item in observed["trace"])
        )
        add(
            "autonomous", successful,
            {"valid_plan": True, "datasets": case.datasets, "findings": True},
            {"valid_plan": bool(plan_entry), "datasets": plan.get("datasets", []), "findings": len(observed["evidence"])},
        )

    if case.category.startswith("ml_") and _evidence_for(observed, "train_ml_model"):
        ml_result = _evidence_for(observed, "train_ml_model")[0].get("result")
        valid = isinstance(ml_result, dict) and (
            "error" in ml_result or (
                len(ml_result.get("models", [])) == 2
                and all("metrics" in model for model in ml_result.get("models", []))
                and "best_model" in ml_result
            )
        )
        add("ml", valid, "bounded ML result or safe error", "valid" if valid else "invalid")

    answer_lower = observed["answer"].lower()
    for text in case.answer_contains:
        add("synthesis", text.lower() in answer_lower, text, observed["answer"][:500])
    for text in case.answer_forbids:
        add("synthesis", text.lower() not in answer_lower, f"must not contain {text}", observed["answer"][:500])

    failures = []
    mapping = {"routing":"routing", "tool_selection":"tool selection", "numerical":"numerical correctness", "synthesis":"synthesis", "safety":"ambiguity"}
    failures.extend(mapping.get(check["kind"], check["kind"]) for check in checks if not check["passed"])
    if provider_error:
        failures.append("provider")
    if any(not check["passed"] and check["kind"] == "autonomous" for check in checks):
        failures.append("planner")
    if any(not check["passed"] and check["kind"] == "tool_selection" for check in checks):
        failures.append("tool selection")
    for entry in observed["trace"]:
        if (
            entry.get("step") == "adaptive_execution" and entry.get("status") == "failed"
        ) or (
            entry.get("step") == "tool_call" and entry.get("success") is False
            and not case.expected_error
        ):
            failures.append("tool execution")
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "expected": asdict(case),
        "routing": _actual_mode(observed["trace"]),
        "tools_used": tools,
        "key_evidence": observed["evidence"][:5],
        "final_answer": observed["answer"],
        "trace": observed["trace"],
        "checks": checks,
        "passed": not failures,
        "failure_classifications": list(dict.fromkeys(failures)),
        "provider_error": provider_error,
    }


def _datasets_for(case: BenchmarkCase) -> dict[str, Any]:
    available = build_benchmark_datasets()
    available["small_customers"] = available["customers"].iloc[:20].copy()
    return {name: available[name].copy() for name in case.datasets}


def run_case(case: BenchmarkCase, layer: str = "deterministic") -> dict[str, Any]:
    agent = Agent()
    diagnostic_provider = None
    if layer == "deterministic":
        agent.llm = ScriptedProvider(case)
    else:
        diagnostic_provider = _DiagnosticProvider(LLMClient())
        agent.llm = diagnostic_provider
    started = time.perf_counter()
    provider_error = None
    try:
        result = agent.run(case.question, datasets=_datasets_for(case), autonomous=None)
        observed = _normalise_result(result)
    except Exception as exc:
        provider_error = str(exc)
        provider_diagnostic = _exception_diagnostic(exc)
        observed = {"answer": "", "trace": [], "evidence": [], "has_figure": False}
    else:
        provider_diagnostic = []
    scored = score_case(case, observed, provider_error)
    scored["provider_diagnostic"] = provider_diagnostic
    scored["provider_calls"] = diagnostic_provider.calls if diagnostic_provider else []
    scored["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return scored


def provider_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def run_benchmark(layer: str = "deterministic", case_ids: set[str] | None = None) -> dict[str, Any]:
    if layer not in {"deterministic", "real"}:
        raise ValueError("layer must be deterministic or real")
    selected = [case for case in benchmark_cases() if case_ids is None or case.id in case_ids]
    if layer == "real" and not provider_available():
        return {
            "metadata": {"benchmark_version": BENCHMARK_VERSION, "layer": layer, "provider": "groq", "model": DEFAULT_MODEL},
            "status": "unavailable",
            "reason": "GROQ_API_KEY is not configured.",
            "summary": summarize([], layer),
            "cases": [],
        }
    results = [run_case(case, layer) for case in selected]
    return {
        "metadata": {
            "benchmark_version": BENCHMARK_VERSION,
            "layer": layer,
            "provider": "scripted" if layer == "deterministic" else "groq",
            "model": None if layer == "deterministic" else DEFAULT_MODEL,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "case_ids": [case.id for case in selected],
        },
        "status": "completed",
        "summary": summarize(results, layer),
        "cases": results,
    }


def serialize_run(run: dict[str, Any]) -> str:
    """Serialize one completed run deterministically for review and comparison."""
    return json.dumps(run, indent=2, sort_keys=True, default=str)


def write_artifacts(run: dict[str, Any], output_directory: str | Path = "evaluation/results") -> tuple[Path, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"v{BENCHMARK_VERSION.replace('.', '_')}_{run['metadata']['layer']}_{stamp}"
    json_path, markdown_path = directory / f"{stem}.json", directory / f"{stem}.md"
    json_path.write_text(serialize_run(run), encoding="utf-8")
    summary = run["summary"]
    lines = [
        f"# Behavioral Benchmark V{BENCHMARK_VERSION}", "",
        f"Layer: **{run['metadata']['layer']}**", "",
        f"Status: **{run['status']}**", "",
        f"Cases: {summary['passed_cases']}/{summary['total_cases']} passed ({summary['overall_pass_rate']}%).", "",
        "## Summary", "", "```json", json.dumps(summary, indent=2), "```", "", "## Cases", "",
    ]
    for item in run["cases"]:
        lines.extend([
            f"### {item['id']} — {'PASS' if item['passed'] else 'FAIL'}", "",
            f"Question: {item['question']}", "",
            f"Routing: {item['routing']}; tools: {', '.join(item['tools_used']) or 'none'}; latency: {item['latency_ms']} ms", "",
            f"Answer: {item['final_answer']}", "",
            f"Failures: {', '.join(item['failure_classifications']) or 'none'}", "",
        ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path
