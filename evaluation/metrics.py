from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable


def values_match(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    return actual == expected


def latency_summary(latencies: Iterable[float]) -> dict[str, float | None]:
    values = sorted(float(value) for value in latencies)
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(values[index], 3) if len(values) >= 2 else None,
    }


def summarize(case_results: list[dict[str, Any]], layer: str) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(bool(item.get("passed")) for item in case_results)
    category = defaultdict(lambda: {"total": 0, "passed": 0})
    for item in case_results:
        bucket = category[item["category"]]
        bucket["total"] += 1
        bucket["passed"] += int(bool(item.get("passed")))

    def rate(check_name: str) -> float | None:
        checks = [check for item in case_results for check in item.get("checks", []) if check["kind"] == check_name]
        return round(100 * sum(check["passed"] for check in checks) / len(checks), 2) if checks else None

    routing_checks = [check for item in case_results for check in item.get("checks", []) if check["kind"] == "routing"]
    false_auto = sum(check.get("actual") == "autonomous" and check.get("expected") == "reactive" for check in routing_checks)
    false_reactive = sum(check.get("actual") == "reactive" and check.get("expected") == "autonomous" for check in routing_checks)
    failure_counts = Counter(reason for item in case_results for reason in item.get("failure_classifications", []))
    timing = latency_summary(item.get("latency_ms", 0) for item in case_results if item.get("latency_ms") is not None)
    return {
        "layer": layer,
        "total_cases": total,
        "passed_cases": passed,
        "overall_pass_rate": round(100 * passed / total, 2) if total else 0.0,
        "routing_accuracy": rate("routing"),
        "false_autonomous_count": false_auto,
        "false_reactive_count": false_reactive,
        "tool_selection_accuracy": rate("tool_selection"),
        "numerical_correctness_rate": rate("numerical"),
        "safety_pass_rate": rate("safety"),
        "autonomous_plan_success_rate": rate("autonomous"),
        "ml_behavior_pass_rate": rate("ml"),
        "fallback_count": sum(any(entry.get("step") == "routing" and entry.get("decision") == "autonomous" for entry in item.get("trace", [])) and not any(entry.get("step") == "final_answer" and entry.get("autonomous") for entry in item.get("trace", [])) for item in case_results),
        "provider_error_count": sum("provider" in item.get("failure_classifications", []) for item in case_results),
        **timing,
        "category_breakdown": {
            name: {**counts, "pass_rate": round(100 * counts["passed"] / counts["total"], 2)}
            for name, counts in sorted(category.items())
        },
        "failure_classification": dict(sorted(failure_counts.items())),
    }
