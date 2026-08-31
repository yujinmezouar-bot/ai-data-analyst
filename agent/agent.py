import json
import re
from collections import OrderedDict
from typing import Any

import pandas as pd

from agent.llm import LLMClient
from autonomous.executor import Executor
from autonomous.plan import AnalysisPlan
from autonomous.planner import AdaptiveReviewError, AnalysisPlanner, PlannerError

from tools.dataset_info import dataset_info, DATASET_INFO_SCHEMA, format_dataset_context, format_datasets_context
from tools.missing_values import missing_values, MISSING_VALUES_SCHEMA
from tools.statistics import statistics, STATISTICS_SCHEMA
from tools.groupby import groupby_analysis, GROUPBY_ANALYSIS_SCHEMA
from tools.visualization import create_visualization, CREATE_VISUALIZATION_SCHEMA, create_multi_dataset_visualization, MULTI_DATASET_VISUALIZATION_SCHEMA
from tools.time_analysis import time_analysis, TIME_ANALYSIS_SCHEMA
from tools.correlation import correlation_analysis, CORRELATION_ANALYSIS_SCHEMA
from tools.outliers import outlier_analysis, OUTLIER_ANALYSIS_SCHEMA
from tools.period_comparison import percentage_change, PERCENTAGE_CHANGE_SCHEMA
from tools.join_datasets import inspect_join_viability, execute_join, INSPECT_JOIN_SCHEMA, EXECUTE_JOIN_SCHEMA
from tools.relationship_discovery import discover_relationships as _discover_relationships_tool, build_schema_graph_summary, DISCOVER_RELATIONSHIPS_SCHEMA
from tools.contribution_analysis import kpi_contribution_analysis, KPI_CONTRIBUTION_SCHEMA
from tools.ml_model import train_ml_model, TRAIN_ML_MODEL_SCHEMA


# ============================================================
# TOOL REGISTRY -- the LLM can only ever call what's in here.
# ============================================================

TOOL_FUNCTIONS = {
    "dataset_info": dataset_info,
    "missing_values": missing_values,
    "statistics": statistics,
    "groupby_analysis": groupby_analysis,
    "create_visualization": create_visualization,
    "time_analysis": time_analysis,
    "correlation_analysis": correlation_analysis,
    "outlier_analysis": outlier_analysis,
    "percentage_change": percentage_change,
    "inspect_join_viability": inspect_join_viability,
    "execute_join": execute_join,
    "discover_relationships": _discover_relationships_tool,
    "create_multi_dataset_visualization": create_multi_dataset_visualization,
    "kpi_contribution_analysis": kpi_contribution_analysis,
    "train_ml_model": train_ml_model,
}

TOOL_SCHEMAS = [
    DATASET_INFO_SCHEMA,
    MISSING_VALUES_SCHEMA,
    STATISTICS_SCHEMA,
    GROUPBY_ANALYSIS_SCHEMA,
    CREATE_VISUALIZATION_SCHEMA,
    TIME_ANALYSIS_SCHEMA,
    CORRELATION_ANALYSIS_SCHEMA,
    OUTLIER_ANALYSIS_SCHEMA,
    PERCENTAGE_CHANGE_SCHEMA,
    INSPECT_JOIN_SCHEMA,
    EXECUTE_JOIN_SCHEMA,
    DISCOVER_RELATIONSHIPS_SCHEMA,
    MULTI_DATASET_VISUALIZATION_SCHEMA,
    KPI_CONTRIBUTION_SCHEMA,
    TRAIN_ML_MODEL_SCHEMA,
]

# Required parameters per tool, derived from the schemas above, used only
# to give the LLM a targeted retry hint when a tool call fails due to
# missing/invalid arguments (V4.1) -- not a second source of truth for
# validation, which still happens inside each tool.
_REQUIRED_PARAMS = {
    schema["function"]["name"]: schema["function"]["parameters"].get("required", [])
    for schema in TOOL_SCHEMAS
}

_DATASET_SCOPED_TOOLS = {
    schema["function"]["name"]
    for schema in TOOL_SCHEMAS
    if "dataset_name" in schema["function"]["parameters"].get("properties", {})
}


# ============================================================
# CONVERSATION MEMORY / MULTI-STEP LIMITS
# ============================================================

# 20 messages = 10 user/assistant turns kept in the tool-decision call.
MAX_HISTORY_MESSAGES = 20

# Hard cap on how many rounds of tool calls a single question can trigger.
# Prevents any possibility of an unbounded loop.
MAX_TOOL_ITERATIONS = 4

# If a compact tool result's JSON would exceed this many characters,
# it gets truncated before being sent to either LLM call.
MAX_TOOL_RESULT_CHARS = 4000
MAX_TRUNCATED_ENTRIES = 25
MAX_RETURNED_EVIDENCE = 20
# Conservative character budget (~6k tokens) for a model environment with an
# 8k TPM limit. Tool schemas are included because providers count them too.
MAX_LLM_REQUEST_CHARS = 24000

CONTRIBUTION_CHANGE_EXECUTION_PLAN = (
    "Decompose the observed KPI change into ranked group contributions and offsets."
)


def _resolve_explicit_dataset_reference(
    question: str, dataset_names: list[str]
) -> str | None:
    """Return one canonically named active dataset explicitly mentioned by the user."""
    matches = [
        name for name in dataset_names
        if re.search(
            rf"(?<!\w){re.escape(str(name))}(?!\w)",
            question,
            flags=re.IGNORECASE,
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _requests_join_with_downstream_analysis(question: str) -> bool:
    """Recognize an explicit join request that also asks for later analysis."""
    match = re.search(r"\bjoin\b", question, flags=re.IGNORECASE)
    if match is None:
        return False
    remainder = question[match.end():]
    return bool(re.search(
        r"\b(then|calculate|aggregate|group|sum|average|total|identify|analy[sz]e|"
        r"highest|lowest|trend|correlat|plot|chart)\b",
        remainder,
        flags=re.IGNORECASE,
    ))


def _build_execution_plan(question: str) -> str | None:
    """Return a concise, user-safe plan only for clearly multi-step requests."""
    text = question.lower()
    time_terms = ("trend", "over time", "monthly", "weekly", "quarter", "yearly")
    ranking_terms = (
        "top ", "bottom ", "best", "worst", "highest", "lowest",
        "most", "least", "biggest", "largest", "smallest",
    )
    change_terms = (
        "change", "grew", "growth", "declined", "decline", "increased", "increase",
        "decreased", "decrease", "dropped", "drop", "rose", "rise",
    )
    contribution_terms = (
        "drove", "drive", "driver", "contribute", "contributed", "contribution",
        "offset", "offsetting", "account for", "accounted for",
    )
    simple_relative_period_comparison = re.search(
        r"\bcompare\s+(?:this|current|latest)\s+(year|quarter|month|week|period)\s+"
        r"(?:with|to|versus|vs\.?)\s+(?:the\s+)?previous\s+\1\b",
        text,
    )

    if any(term in text for term in contribution_terms) and any(term in text for term in change_terms):
        return CONTRIBUTION_CHANGE_EXECUTION_PLAN
    if "why" in text and any(term in text for term in change_terms):
        return "Identify the observed change, then examine relevant groups for associated contributors."
    if any(term in text for term in time_terms) and any(term in text for term in ranking_terms):
        return "Rank the requested entities, then analyze the selected entities over time."
    if not simple_relative_period_comparison and any(
        term in text for term in ("compare", "relationship", "correlation")
    ) and any(
        term in text for term in time_terms
    ):
        return "Run the requested analyses in sequence and combine the relevant results."
    return None


def _estimate_request_chars(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Conservative, dependency-free request-size estimate."""
    return len(json.dumps({"messages": messages, "tools": tools or []}, default=str))


def _compact_messages_for_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    current_question: str | None = None,
) -> list[dict[str, Any]]:
    """Keep system, current question, and newest context within the request budget."""
    if _estimate_request_chars(messages, tools) <= MAX_LLM_REQUEST_CHARS:
        return messages

    required = {0, len(messages) - 1}
    if current_question is not None:
        for index in range(len(messages) - 1, 0, -1):
            if messages[index].get("role") == "user" and messages[index].get("content") == current_question:
                required.add(index)
                break

    selected = set(required)
    context_indexes = [index for index in range(1, len(messages) - 1) if index not in required]
    if context_indexes:
        newest_index = context_indexes[-1]
        newest = dict(messages[newest_index])
        complete_candidate = [
            message for index, message in enumerate(messages)
            if index in selected | {newest_index}
        ]
        if _estimate_request_chars(complete_candidate, tools) <= MAX_LLM_REQUEST_CHARS:
            selected.add(newest_index)
        else:
            # The tool schema can leave less room than one history message.
            # Reserve a bounded prefix of the newest item before considering
            # older context so follow-up references are not lost entirely.
            content = str(newest.get("content", ""))
            low, high = 0, len(content)
            while low < high:
                midpoint = (low + high + 1) // 2
                newest["content"] = content[:midpoint] + "\n[Context compacted for request size.]"
                candidate = [
                    newest if index == newest_index else message
                    for index, message in enumerate(messages)
                    if index in selected | {newest_index}
                ]
                if _estimate_request_chars(candidate, tools) <= MAX_LLM_REQUEST_CHARS:
                    low = midpoint
                else:
                    high = midpoint - 1
            if low:
                messages = list(messages)
                messages[newest_index] = {
                    **newest,
                    "content": content[:low] + "\n[Context compacted for request size.]",
                }
                selected.add(newest_index)

    for index in range(len(messages) - 2, 0, -1):
        if index in selected:
            continue
        candidate = [message for i, message in enumerate(messages) if i in selected | {index}]
        if _estimate_request_chars(candidate, tools) <= MAX_LLM_REQUEST_CHARS:
            selected.add(index)

    compacted = [message for index, message in enumerate(messages) if index in selected]
    # A very large newest tool-result message is reduced only after preserving
    # the current question, system instructions, and its leading key findings.
    while _estimate_request_chars(compacted, tools) > MAX_LLM_REQUEST_CHARS and len(compacted) > 2:
        removable = next((i for i, message in enumerate(compacted[1:-1], start=1)
                          if message.get("content") != current_question), None)
        if removable is None:
            break
        compacted.pop(removable)
    if _estimate_request_chars(compacted, tools) > MAX_LLM_REQUEST_CHARS:
        last = compacted[-1]
        content = str(last.get("content", ""))
        suffix = "\n[Context compacted for request size.]"
        low, high = 0, len(content)
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate_last = {**last, "content": content[:midpoint] + suffix}
            candidate = [*compacted[:-1], candidate_last]
            if _estimate_request_chars(candidate, tools) <= MAX_LLM_REQUEST_CHARS:
                low = midpoint
            else:
                high = midpoint - 1
        compacted[-1] = {**last, "content": content[:low] + suffix}
    return compacted


def _compact_tool_result(result: Any) -> Any:
    """
    Safety net against oversized tool outputs. The tools themselves
    already cap group counts internally (MAX_GROUPS_RETURNED / similar
    caps in correlation, outliers, period_comparison), but this catches
    anything unexpectedly large regardless of which tool produced it.
    """
    try:
        serialized = json.dumps(result, default=str)
    except Exception:
        return result

    if len(serialized) <= MAX_TOOL_RESULT_CHARS:
        return result

    if isinstance(result, dict) and isinstance(result.get("result"), dict):
        items = list(result["result"].items())
        compact = dict(result)
        compact["result"] = dict(items[:MAX_TRUNCATED_ENTRIES])
        compact["note"] = (
            f"Result truncated to the first {MAX_TRUNCATED_ENTRIES} of {len(items)} "
            "entries to keep the response compact. Ask a more specific question "
            "(top_n, a filter, or a narrower time range) to see the rest."
        )
        return compact

    # Same truncation strategy for other list-shaped result fields
    # (e.g. correlation's "top_correlations", period_comparison's "changes").
    for list_key in ("top_correlations", "changes"):
        if isinstance(result, dict) and isinstance(result.get(list_key), list):
            items = result[list_key]
            compact = dict(result)
            compact[list_key] = items[:MAX_TRUNCATED_ENTRIES]
            compact["note"] = (
                f"'{list_key}' truncated to the first {MAX_TRUNCATED_ENTRIES} of {len(items)} "
                "entries to keep the response compact."
            )
            return compact

    return {
        "note": "Tool result was too large and has been truncated.",
        "truncated_preview": serialized[:MAX_TOOL_RESULT_CHARS],
    }


def _compact_ml_result_for_llm(result: Any) -> Any:
    """Bound ML prompt content without degrading retained structured evidence."""
    if not isinstance(result, dict) or "error" in result:
        return result
    return {
        "task_type": result.get("task_type"),
        "target_column": result.get("target_column"),
        "rows_received": result.get("rows_received"),
        "rows_used": result.get("rows_used"),
        "rows_dropped": result.get("rows_dropped"),
        "features_used": list(result.get("features_used") or [])[:20],
        "features_excluded": list(result.get("features_excluded") or [])[:20],
        "split": result.get("split"),
        "target_summary": result.get("target_summary"),
        "models": list(result.get("models") or [])[:2],
        "best_model": result.get("best_model"),
        "selection_metric": result.get("selection_metric"),
        "feature_associations": list(result.get("feature_associations") or [])[:5],
        "warnings": [str(value)[:300] for value in (result.get("warnings") or [])[:8]],
        "limitations": [str(value)[:300] for value in (result.get("limitations") or [])[:5]],
    }


# ============================================================
# TOOL-RESULT SUMMARISER  (deterministic, no LLM calls)
# ============================================================

def _summarise_tool_results(
    all_tool_results: list[dict[str, Any]],
    has_figure: bool,
) -> str:
    """
    Produce a compact, human-readable summary of key analytical findings
    extracted deterministically from structured tool outputs.

    This is injected into the final explanation prompt alongside the raw
    tool result JSON so the LLM receives pre-digested analytical hints
    and doesn't have to re-parse raw JSON to find the most important
    numbers.  The LLM must still base exact quoted figures on the raw
    results; this text only highlights *which* fields are most relevant.
    """
    hints: list[str] = []
    any_truncated = False
    any_error = False
    sufficiency: list[str] = []

    for item in all_tool_results:
        tool = item["tool_name"]
        res = item["result"]
        if not isinstance(res, dict):
            continue

        if "error" in res:
            any_error = True
            hints.append(f"[{tool}] FAILED: {res['error']}")
            error_text = str(res["error"]).lower()
            if any(marker in error_text for marker in (
                "no rows", "no data", "not enough", "no numeric", "no valid", "constant", "no variance",
            )):
                sufficiency.append(f"{tool}: insufficient data for this analysis")
            continue

        if "note" in res and "truncated" in str(res.get("note", "")).lower():
            any_truncated = True

        if tool == "time_analysis":
            td = res.get("trend_direction")
            if td and td != "insufficient_data":
                hints.append(f"[time_analysis] Trend direction: {td}")
            best = res.get("best_period")
            worst = res.get("worst_period")
            if best:
                hints.append(f"[time_analysis] Best period: {best}")
            if worst and worst != best:
                hints.append(f"[time_analysis] Worst period: {worst}")
            oc = res.get("overall_change")
            op = res.get("overall_percentage_change")
            if oc is not None:
                pct_str = f" ({op:+.1f}%)" if op is not None else ""
                hints.append(f"[time_analysis] Overall change: {oc:+g}{pct_str}")
            periods = res.get("total_periods")
            if isinstance(periods, int) and periods < 3:
                sufficiency.append(f"time_analysis: limited data ({periods} period(s))")

        elif tool == "percentage_change":
            cs = res.get("comparison_summary")
            ac = res.get("absolute_change")
            pc = res.get("percentage_change")
            if cs and ac is not None:
                pct_str = f" ({pc:+.1f}%)" if pc is not None else " (previous value was zero)"
                hints.append(f"[percentage_change] {cs.capitalize()}: absolute change {ac:+g}{pct_str}")
            li = res.get("largest_increase")
            ld = res.get("largest_decrease")
            if isinstance(li, dict):
                hints.append(
                    f"[percentage_change] Largest increase: "
                    f"{li.get('current_period')} ({li.get('percentage_change'):+.1f}%)"
                )
            if isinstance(ld, dict):
                hints.append(
                    f"[percentage_change] Largest decrease: "
                    f"{ld.get('current_period')} ({ld.get('percentage_change'):+.1f}%)"
                )
            oc = res.get("overall_change")
            op_pct = res.get("overall_percentage_change")
            if oc is not None and "comparison_summary" not in res:
                op_str = f" ({op_pct:+.1f}%)" if op_pct is not None else ""
                hints.append(f"[percentage_change] Overall change across all periods: {oc:+g}{op_str}")

        elif tool == "groupby_analysis":
            bg = res.get("best_group")
            wg = res.get("worst_group")
            if bg:
                hints.append(f"[groupby_analysis] Best group: {bg}")
            if wg and wg != bg:
                hints.append(f"[groupby_analysis] Worst group: {wg}")
            comp = res.get("comparison")
            if isinstance(comp, dict):
                pct = comp.get("percentage_change")
                pct_str = f" (pct change: {pct:+.1f}%)" if pct is not None else ""
                hints.append(
                    f"[groupby_analysis] {comp['group_1']} vs {comp['group_2']}: "
                    f"abs diff {comp['absolute_difference']:g}{pct_str}"
                )

            ranking = res.get("ranking")
            values = res.get("result")
            if isinstance(ranking, list) and isinstance(values, dict) and ranking:
                top_ranked = ranking[:3]
                ranking_text = ", ".join(
                    f"{group} ({values[group]:g})"
                    for group in top_ranked
                    if isinstance(values.get(group), (int, float))
                )
                if ranking_text:
                    hints.append(f"[groupby_analysis] Leading ranking: {ranking_text}")

        elif tool == "statistics":
            single_col = res.get("column")
            if single_col:
                count = res.get("count")
                if isinstance(count, int) and count < 3:
                    sufficiency.append(f"statistics: limited data ({count} observation(s))")
                skew = res.get("skewness")
                cv = res.get("coefficient_of_variation")
                if skew is not None and abs(float(skew)) > 1:
                    direction = "right (positively)" if float(skew) > 0 else "left (negatively)"
                    hints.append(
                        f"[statistics] '{single_col}' is notably {direction} skewed "
                        f"(skewness={float(skew):.2f})"
                    )
                if cv is not None and float(cv) > 1.0:
                    hints.append(f"[statistics] '{single_col}' has high variability (CV={float(cv):.2f})")

        elif tool == "missing_values":
            total_missing = res.get("total_missing_values")
            missing_columns = res.get("columns_with_missing")
            if total_missing == 0:
                hints.append("[missing_values] No missing values were found.")
            elif isinstance(missing_columns, dict):
                most_affected = sorted(
                    missing_columns.items(),
                    key=lambda item: item[1].get("missing_percentage", 0),
                    reverse=True,
                )[:3]
                summary = ", ".join(
                    f"{column} ({info.get('missing_count')} missing; "
                    f"{info.get('missing_percentage')}%)"
                    for column, info in most_affected
                    if isinstance(info, dict)
                )
                if summary:
                    hints.append(
                        f"[missing_values] {total_missing} missing value(s); most affected: {summary}"
                    )

        elif tool == "correlation_analysis":
            sp = res.get("strongest_positive")
            sn = res.get("strongest_negative")
            spp = res.get("strongest_positive_pair")
            snp = res.get("strongest_negative_pair")
            if sp:
                hints.append(
                    f"[correlation] Strongest positive association with target: "
                    f"{sp[0]} (r={sp[1]:.3f})"
                )
            if sn:
                hints.append(
                    f"[correlation] Strongest negative association with target: "
                    f"{sn[0]} (r={sn[1]:.3f})"
                )
            if isinstance(spp, dict):
                hints.append(
                    f"[correlation] Strongest positive pair: "
                    f"{spp['column_1']} & {spp['column_2']} (r={spp['correlation']:.3f})"
                )
            if isinstance(snp, dict):
                hints.append(
                    f"[correlation] Strongest negative pair: "
                    f"{snp['column_1']} & {snp['column_2']} (r={snp['correlation']:.3f})"
                )

        elif tool == "outlier_analysis":
            # Single-column results are returned at the top level, while
            # multi-column results use the `results` mapping.
            if res.get("column") and res.get("outlier_count", 0) > 0:
                examples = res.get("example_outlier_values", [])
                example_text = f"; examples: {examples}" if examples else ""
                hints.append(
                    f"[outlier_analysis] {res['column']} has {res['outlier_count']} "
                    f"outlier(s) ({res.get('outlier_percentage')}%){example_text}"
                )

            sc = res.get("results")
            if isinstance(sc, dict):
                notable = [
                    (col, info)
                    for col, info in sc.items()
                    if isinstance(info, dict) and info.get("outlier_count", 0) > 0
                ]
                if notable:
                    top = sorted(
                        notable, key=lambda kv: kv[1].get("outlier_count", 0), reverse=True
                    )[:3]
                    summary = ", ".join(
                        f"{c} ({i['outlier_count']} outlier(s))" for c, i in top
                    )
                    hints.append(f"[outlier_analysis] Columns with outliers: {summary}")

        elif tool == "kpi_contribution_analysis":
            overall = res.get("overall", {})
            if isinstance(overall, dict):
                pct = overall.get("percentage_change")
                pct_text = f", {pct}%" if pct is not None else ""
                hints.append(
                    "[kpi_contribution_analysis] Observed total KPI "
                    f"{overall.get('direction')}: {overall.get('value_a')} to {overall.get('value_b')} "
                    f"(change {overall.get('absolute_change')}{pct_text})."
                )
            contributors = res.get("contributors")
            if isinstance(contributors, list):
                drivers = [item for item in contributors if str(item.get("effect", "")).startswith("reinforces_")]
                offsets = [item for item in contributors if str(item.get("effect", "")).startswith("offsets_")]
                if drivers:
                    driver = drivers[0]
                    hints.append(
                        "[kpi_contribution_analysis] Largest returned mathematical driver: "
                        f"{driver.get('group')} ({driver.get('absolute_change')}; "
                        f"{driver.get('contribution_to_total_change_percentage')}% of net change)."
                    )
                if offsets:
                    offset = min(
                        offsets,
                        key=lambda item: item.get("contribution_to_total_change_percentage") or 0,
                    )
                    hints.append(
                        "[kpi_contribution_analysis] Largest returned offset: "
                        f"{offset.get('group')} ({offset.get('absolute_change')})."
                    )

        elif tool == "train_ml_model":
            models = res.get("models") if isinstance(res.get("models"), list) else []
            hints.append(
                f"[train_ml_model] {res.get('task_type')} target '{res.get('target_column')}', "
                f"{res.get('rows_used')} usable rows; selected model: {res.get('best_model')}."
            )
            split = res.get("split") if isinstance(res.get("split"), dict) else {}
            if split.get("group_aware"):
                hints.append(
                    f"[train_ml_model] Group-isolated evaluation by '{split.get('group_column')}': "
                    f"{split.get('train_groups')} train groups, {split.get('test_groups')} test groups, "
                    "zero group overlap."
                )
            for model in models:
                if isinstance(model, dict):
                    hints.append(
                        f"[train_ml_model] {model.get('name')} test metrics: "
                        f"{json.dumps(model.get('metrics', {}), default=str)}"
                    )

        filter_applied = res.get("filter_applied")
        if isinstance(filter_applied, dict):
            column = filter_applied.get("column")
            values = filter_applied.get("values")
            if column and isinstance(values, list):
                hints.append(f"[{tool}] Filtered scope: {column} in {values}")

    lines: list[str] = []
    if hints:
        lines.append("KEY ANALYTICAL FINDINGS (extracted by Python):\n" + "\n".join(hints))
    if has_figure:
        lines.append(
            "NOTE: A chart was generated. Describe what it visually shows "
            "based on the data results."
        )
    if any_truncated:
        lines.append(
            "NOTE: Some results were truncated due to size limits. "
            "Do not present truncated results as complete — acknowledge this "
            "limitation in the answer."
        )
    if any_error:
        lines.append(
            "NOTE: One or more tools reported an error. For failed analyses, explain "
            "clearly what could not be determined and why, rather than inventing a result."
        )
    if sufficiency:
        lines.append(
            "DATA SUFFICIENCY: " + "; ".join(dict.fromkeys(sufficiency)) + ". "
            "State this as a limitation and do not draw a stronger conclusion."
        )

    return "\n\n".join(lines)


def _compact_autonomous_findings(findings: list[Any]) -> list[dict[str, Any]]:
    """Convert structured findings into bounded final-explanation evidence."""
    compact: list[dict[str, Any]] = []
    for finding in findings:
        compact_result = _compact_tool_result(finding.result)
        serialized_result = json.dumps(compact_result, default=str)
        if len(serialized_result) > MAX_TOOL_RESULT_CHARS:
            compact_result = {
                "note": "Result truncated to keep the final explanation compact.",
                "truncated_preview": serialized_result[:MAX_TOOL_RESULT_CHARS - 200],
            }
        compact.append({
            "tool_name": finding.tool_name,
            "datasets": list(finding.datasets),
            "result": compact_result,
            "provenance": _compact_tool_result(finding.provenance),
        })
    return compact


def _autonomous_fallback_answer(compact_findings: list[dict[str, Any]]) -> str:
    """Return a bounded deterministic answer when final synthesis is unavailable."""
    if not compact_findings:
        return "The autonomous analysis completed, but no findings were produced."

    summary = _summarise_tool_results(
        [{"tool_name": item["tool_name"], "result": item["result"]} for item in compact_findings],
        has_figure=False,
    )
    if summary:
        return summary

    previews = []
    for item in compact_findings[:3]:
        result_text = json.dumps(item["result"], default=str)
        previews.append(
            f"{item['tool_name']} on {', '.join(item['datasets'])}: {result_text[:500]}"
        )
    return "Autonomous analysis completed. " + "\n".join(previews)


MAX_AUTONOMOUS_DIAGNOSTIC_MESSAGE_CHARS = 500


def _sanitize_autonomous_diagnostic_text(value: Any) -> str:
    """Return bounded exception text without common credential representations."""
    text = " ".join(str(value).split())
    text = re.sub(
        r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer [redacted]", text)
    return text[:MAX_AUTONOMOUS_DIAGNOSTIC_MESSAGE_CHARS]


def _autonomous_plan_summary(plan: AnalysisPlan) -> dict[str, Any]:
    """Summarize a validated plan without arguments or user/provider content."""
    return {
        "id": str(plan.id)[:100],
        "datasets": [str(name)[:100] for name in plan.datasets[:20]],
        "step_count": len(plan.steps),
        "steps": [
            {"id": str(step.id)[:100], "tool_name": str(step.tool_name)[:100]}
            for step in plan.steps[:10]
        ],
    }


def _planner_failure_stage(exc: Exception) -> tuple[str, bool, bool]:
    """Classify the existing PlannerError contract without retaining raw output."""
    message = str(exc)
    if isinstance(exc, PlannerError):
        if message.startswith("LLM provider failed:"):
            return "planner_call", False, False
        if message == "LLM provider returned no textual content":
            return "planner_call", True, False
        if message.startswith("LLM returned invalid JSON:"):
            return "planner_parse", True, False
        if message.startswith("Plan validation failed:") or message.startswith("Unknown tool referenced in plan:"):
            return "plan_validation", True, True
    return "planner_call", False, False


# ============================================================
# FINAL EXPLANATION SYSTEM PROMPT (V5.4)
# ============================================================

FINAL_EXPLANATION_SYSTEM_PROMPT = """\
You are a professional data analyst assistant producing the FINAL answer to the user.
All numerical calculations have already been performed by deterministic Python tools.
You must NEVER invent, estimate, or recompute numbers — use ONLY the values returned by the tools.

YOUR ROLE:
- Transform tool results into a concise, useful, evidence-based insight.
- Ground every factual statement in the provided tool results.
- Highlight the most relevant analytical findings for the user's question.
- Adapt answer length to the question: simple questions get brief answers; analytical questions get structured explanations.

GROUNDING RULES (mandatory):
1. Use ONLY the exact numerical values returned by Python tools. Do not recompute or round differently.
2. Do NOT invent category names, column names, trends, or causes absent from the results.
3. Do NOT contradict tool results.
4. If a tool call failed, explain what could NOT be determined — do not fabricate an alternative answer.
5. If results were truncated, acknowledge that the conclusion is based on the available subset.
6. Do NOT claim that any analysis was performed if no tool result confirms it.
7. When DATA SUFFICIENCY is provided, state whether the evidence is limited or insufficient; do not invent a numerical confidence score.
8. When a filtered scope is provided, make clear that the finding applies to that subset rather than the entire dataset.
9. KPI contribution results are mathematical decomposition, not causation. Say a group "accounted for" or "contributed to" the observed change; never say it caused the change.
10. ML metrics are held-out test estimates, not external validation. Feature associations are predictive and non-causal. Never claim they caused the target.
11. For ML, group_column controls evaluation isolation and is never a predictive feature. Grouped evaluation estimates performance on unseen groups; ordinary row splitting does not. Repeated entities can inflate row-level random-split metrics.

INSIGHT EXTRACTION — when relevant, surface:
• Comparisons: best/worst group, absolute difference, percentage change, percentage difference (they are NOT the same thing)
• Trends: trend direction (use the tool's trend_direction field verbatim), best/worst period, overall change
• Distribution: mean vs median differences, high variability (high CV), skewness direction
• Correlation: strongest positive/negative relationships (state as statistical associations, NEVER as causation)
• Outliers: columns with notable outliers, extreme observations actually returned

ANSWER STRUCTURE — for analytical questions, prefer:
1. Direct answer / headline conclusion.
2. Key supporting findings with numbers.
3. Limitation or caveat if relevant (truncation, failed tool, insufficient data).

Do NOT include bullet lists for simple scalar answers.

"WHY" QUESTIONS — special handling:
- Distinguish between OBSERVED DATA PATTERNS and CAUSAL EXPLANATIONS.
- If data supports observational statements only, say so explicitly:
  "The available data does not establish the underlying cause."
- Describe observed contributors factually: "Category B showed the largest decline."
- Do NOT invent external causes (competitor actions, market forces, customer preferences) unless a relevant variable was analyzed and supports it.
- Correlation is NOT causation — never describe a correlation as a cause.

VISUALIZATION — if a chart was generated:
- Describe what it visually shows based on the data results.
- Only state visual patterns supported by the actual data values.

FORBIDDEN PATTERNS:
- Repeating the same number three times without adding value.
- "This is because customers prefer..." (invented cause).
- "Based on industry trends..." (outside the dataset).
- "Confidence: 85%" (no confidence scores were calculated).
- "The correlation proves that..." (correlation ≠ causation).
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are a data analyst using deterministic pandas tools; the dataset is available to them. "
    "For greetings or non-data questions answer directly. For data facts, use the minimum sufficient tool: "
    "rankings and category comparisons use groupby_analysis; descriptive distributions use statistics; missing-data checks use missing_values; "
    "trends use time_analysis; period-over-period comparisons use percentage_change; relationships use correlation_analysis; unusual values use outlier_analysis; "
    "questions about which groups drove, contributed to, or accounted for a KPI increase/decrease use kpi_contribution_analysis; "
    "supervised model training/evaluation uses train_ml_model only when the target and classification/regression task are explicit; "
    "omit feature_columns for safe automatic selection and never fabricate a target or feature list; "
    "for explicitly named unseen entities, use an unambiguous group_column; otherwise ask, never invent it; "
    "dataset structure uses dataset_info only when the compact dataset context is insufficient. "
    "and charts=create_visualization only when they materially help. Set aggregation (and date period) for aggregated charts. "
    "Use compact context and recent conversation to resolve columns and references. For multi-step work, pass exact returned "
    "entities into filter_values. Never invent filter values. Reuse available rankings, trends, comparisons, and correlations; stop when sufficient. "
    "If intent remains ambiguous, ask one concise clarification. Python provides exact metrics: do not calculate or guess. "
    "Distinguish absolute difference, percentage difference, and percentage change; read trend_direction directly. "
    "Correlation is association, never causation. For 'why' questions investigate after identifying the change with time_analysis or percentage_change, then report observed contributors as associations only. "
    "Explain tool errors, limited data, filters, and truncation honestly. Do not repeat tool calls or dump raw data. "
    "Call create_visualization when useful; do not generate charts for simple scalar lookups."
)


class Agent:
    """
    Orchestrates the LLM <-> tool-calling process, including multi-step
    tool calling within a single question.

    run(question, df, conversation_history) returns:
        {"answer": str, "figure": Plotly Figure or None, "trace": list[dict]}

    conversation_history is read-only here; app.py owns persisting it.
    It must only ever contain plain {"role": "user"/"assistant", "content": str}
    entries -- never tool_calls, never role="tool". The multi-step loop
    below follows the same rule: intermediate tool rounds are appended as
    plain assistant/user text, never as tool_calls/tool-role messages.
    This is deliberate -- it is what avoids Groq's
    "Tool choice is none, but model called a tool" error, and the final
    explanation call below remains a fully separate, tools-free
    conversation for the same reason.

    "trace" is a lightweight, observability-only record of which tools
    were called, with what arguments, and whether they succeeded. It
    never contains model reasoning/chain-of-thought -- only tool-call
    bookkeeping -- and is safe to log or display in a debug panel.
    """

    _MAX_DERIVED_DATASETS = 3

    def __init__(self) -> None:
        # Lazy-initialize LLMClient so tests can patch LLMClient after Agent instantiation.
        self.llm = None
        self.derived_datasets: OrderedDict[str, Any] = OrderedDict()
        self._derived_count = 0

    def register_derived_dataset(self, df: pd.DataFrame, suggested_name: str | None = None) -> str:
        """Register a derived dataset into the canonical agent store.

        This is the single canonical place for derived-dataset naming, LRU
        enforcement, and persistent storage. It mirrors the behavior used
        internally when execute_join is invoked through the agent tool path.

        The signature accepts the DataFrame and an optional suggested_name.
        If suggested_name is already present, a counter-based name will be
        assigned to avoid collisions.

        Returns the derived dataset name assigned.
        """
        # Ensure counters exist and increment for deterministic naming
        self._derived_count = getattr(self, "_derived_count", 0) + 1

        if suggested_name:
            derived_name = suggested_name
            if derived_name in self.derived_datasets:
                # Fall back to counter-based name to avoid clobbering
                derived_name = f"derived_join_{self._derived_count}"
        else:
            derived_name = f"derived_join_{self._derived_count}"

        # Evict oldest if at capacity
        while len(self.derived_datasets) >= self._MAX_DERIVED_DATASETS:
            self.derived_datasets.popitem(last=False)

        # Store it canonically
        self.derived_datasets[derived_name] = df
        return derived_name

    def derived_dataset_register(self, name: str, df: pd.DataFrame) -> str:
        """Adapter matching Executor's expected signature: (name, df) -> str.

        This delegates to register_derived_dataset and returns the canonical
        derived name assigned (may differ if the suggested name collided).
        """
        return self.register_derived_dataset(df, suggested_name=name)

    def run(
        self,
        question: str,
        df: pd.DataFrame = None,
        conversation_history: list[dict[str, str]] | None = None,
        datasets: dict[str, pd.DataFrame] = None,
        autonomous: bool | None = False,
    ) -> dict[str, Any]:

        history = conversation_history or []
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        # Ensure llm is initialized lazily to allow tests to patch LLMClient before first use.
        if self.llm is None:
            self.llm = LLMClient()

        # Reset derived datasets at the start of each run
        self.derived_datasets = OrderedDict()
        self._derived_count = 0

        # Normalize datasets
        self.active_datasets = datasets or {}
        if df is not None:
            self.active_datasets["default"] = df

        # Determine the primary dataset for backward compatibility with tools
        self.primary_df = next(iter(self.active_datasets.values())) if self.active_datasets else None
        explicit_dataset_name = _resolve_explicit_dataset_reference(
            question, list(self.active_datasets.keys())
        )
        join_pipeline_requested = _requests_join_with_downstream_analysis(question)

        execution_plan_signal = _build_execution_plan(question)
        routing_trace = None
        attempt_autonomous = autonomous is True
        if autonomous is None:
            text = f" {question.lower()} "
            visualization_terms = (" chart", " plot", " graph", "visualize", "visualise", "visualization", "visualisation")
            context_references = (" those ", " them ", " their ", " these ", " it ", " that ", " they ", " above ")
            without_temporal_previous = re.sub(
                r"\bprevious\s+(?:year(?:'s)?|quarter|month|week|period)\b", "", text
            )
            has_context_reference = (
                any(term in text for term in context_references)
                or " previous " in without_temporal_previous
            )

            if any(term in text for term in visualization_terms):
                routing_reason = "visualization_first_request"
            elif has_context_reference:
                routing_reason = "context_dependent_request"
            elif not self.active_datasets:
                routing_reason = "no_datasets"
            elif execution_plan_signal is None:
                routing_reason = "no_clear_multistep_signal"
            else:
                attempt_autonomous = True
                routing_reason = (
                    "contribution_change_signal"
                    if execution_plan_signal == CONTRIBUTION_CHANGE_EXECUTION_PLAN
                    else "clear_multistep_signal"
                )

            routing_trace = {
                "step": "routing",
                "mode": "auto",
                "decision": "autonomous" if attempt_autonomous else "reactive",
                "reason": routing_reason,
            }

        autonomous_fallback_trace = None
        if attempt_autonomous and self.active_datasets:
            autonomous_stage = "planner_call"
            planner_output_received = False
            planner_json_parsed = False
            plan_validated = False
            autonomous_plan = None
            executor = None
            try:
                planner = AnalysisPlanner(
                    self.llm,
                    tools_registry=TOOL_FUNCTIONS,
                    validate_tools=True,
                )
                autonomous_plan = planner.plan(
                    question,
                    context={
                        "datasets": list(self.active_datasets.keys()),
                        "dataset_context": format_datasets_context(self.active_datasets),
                        "relationship_context": (
                            build_schema_graph_summary(self.active_datasets)
                            if len(self.active_datasets) >= 2 else ""
                        ),
                        "tool_schemas": TOOL_SCHEMAS,
                    },
                )
                planner_output_received = True
                planner_json_parsed = True
                plan_validated = True
                autonomous_stage = "plan_validation"
                if not autonomous_plan.steps:
                    raise ValueError("Autonomous plan contained no executable steps.")

                executor = Executor(
                    TOOL_FUNCTIONS,
                    derived_dataset_register=self.derived_dataset_register,
                    tool_schemas=TOOL_SCHEMAS,
                )
                autonomous_stage = "preflight"
                findings_store = executor.execute(autonomous_plan, self.active_datasets)
                autonomous_stage = "post_execution"
                findings = findings_store.all()
                adaptive_trace: list[dict[str, Any]] = []
                adaptive_limitation = None
                initial_compact_findings = _compact_autonomous_findings(findings)
                review_messages = planner.build_review_prompt(
                    question,
                    initial_compact_findings,
                    format_datasets_context(self.active_datasets, self.derived_datasets),
                    TOOL_SCHEMAS,
                )
                review_messages = _compact_messages_for_request(
                    review_messages, current_question=question
                )

                review = None
                try:
                    review_message = self.llm.chat(review_messages, tool_choice="none")
                    review_content = getattr(review_message, "content", None)
                    if not isinstance(review_content, str) or not review_content.strip():
                        raise ValueError("Adaptive review returned no content.")
                    review = planner.parse_review(review_content, max_follow_up_steps=2)
                except Exception as exc:
                    failure_status = "failed" if not isinstance(exc, ValueError) else "invalid"
                    if exc.__class__.__name__ == "PlannerError":
                        failure_status = "invalid"
                    if isinstance(exc, AdaptiveReviewError):
                        failure_status = "invalid"
                    review_trace = {
                        "step": "adaptive_review",
                        "status": failure_status,
                        "reason": "review_provider_failure" if failure_status == "failed" else "invalid_review_response",
                        "proposed_steps": 0,
                    }
                    if failure_status == "invalid":
                        failure_stage = (
                            getattr(exc, "failure_stage", None)
                            or ("empty_content" if isinstance(exc, ValueError) else "contract_validation")
                        )
                        review_trace.update({
                            "failure_stage": failure_stage,
                            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}"[:200],
                            "diagnostic_message": _sanitize_autonomous_diagnostic_text(exc),
                        })
                        parsed_metadata = getattr(exc, "parsed_metadata", {})
                        if parsed_metadata:
                            safe_keys = [
                                _sanitize_autonomous_diagnostic_text(key)[:100]
                                for key in parsed_metadata.get("top_level_keys", [])[:20]
                            ]
                            safe_types = {
                                _sanitize_autonomous_diagnostic_text(key)[:100]: str(value)[:50]
                                for key, value in list(parsed_metadata.get("top_level_types", {}).items())[:20]
                            }
                            review_trace["parsed_top_level_keys"] = safe_keys
                            review_trace["parsed_top_level_types"] = safe_types
                    adaptive_trace.extend([
                        review_trace,
                        {"step": "adaptive_stop", "reason": failure_status + "_review"},
                    ])
                    adaptive_limitation = "Adaptive review could not be completed; the answer uses the initial findings."

                if review is not None:
                    proposed_steps = len(review["steps"])
                    adaptive_trace.append({
                        "step": "adaptive_review",
                        "status": review["status"],
                        "reason": review["reason"],
                        "proposed_steps": proposed_steps,
                    })
                    if review["status"] == "complete":
                        adaptive_trace.append({"step": "adaptive_stop", "reason": "review_complete"})
                    else:
                        remaining_steps = max(0, 10 - len(autonomous_plan.steps))
                        follow_up_steps = review["steps"]
                        validation_reason = None
                        if not remaining_steps or len(follow_up_steps) > remaining_steps:
                            validation_reason = "global_step_limit"
                        elif any(step["tool_name"] == "execute_join" for step in follow_up_steps):
                            validation_reason = "follow_up_join_prohibited"
                        elif any(step["tool_name"] == "train_ml_model" for step in follow_up_steps):
                            validation_reason = "follow_up_ml_prohibited"
                        elif any(not step["read_only"] for step in follow_up_steps):
                            validation_reason = "follow_up_mutation_prohibited"

                        adaptive_plan_id = f"adaptive_{autonomous_plan.id}"
                        if validation_reason is None:
                            available_datasets = {
                                **self.active_datasets,
                                **self.derived_datasets,
                            }
                            try:
                                adaptive_plan = AnalysisPlan.from_dict(
                                    {
                                        "id": adaptive_plan_id,
                                        "objective": review["reason"],
                                        "datasets": list(available_datasets.keys()),
                                        "steps": follow_up_steps,
                                        "constraints": {
                                            "parent_plan_id": autonomous_plan.id,
                                            "adaptive_round": 1,
                                            "reviewer_reason": review["reason"],
                                        },
                                    },
                                    max_steps=remaining_steps,
                                    allowed_datasets=list(available_datasets.keys()),
                                )
                                before_count = len(executor.findings_store)
                                executor.execute(adaptive_plan, available_datasets)
                                executed_steps = len(executor.findings_store) - before_count
                                adaptive_trace.extend([
                                    {
                                        "step": "adaptive_execution",
                                        "plan_id": adaptive_plan_id,
                                        "executed_steps": executed_steps,
                                        "status": "completed",
                                    },
                                    {"step": "adaptive_stop", "reason": "follow_up_complete"},
                                ])
                            except Exception:
                                executed_steps = len(executor.findings_store) - before_count if "before_count" in locals() else 0
                                adaptive_trace.extend([
                                    {
                                        "step": "adaptive_execution",
                                        "plan_id": adaptive_plan_id,
                                        "executed_steps": executed_steps,
                                        "status": "failed",
                                    },
                                    {"step": "adaptive_stop", "reason": "follow_up_failed"},
                                ])
                                adaptive_limitation = "The adaptive follow-up could not be completed; available findings were preserved."
                        else:
                            adaptive_trace.extend([
                                {
                                    "step": "adaptive_execution",
                                    "plan_id": adaptive_plan_id,
                                    "executed_steps": 0,
                                    "status": "failed",
                                },
                                {"step": "adaptive_stop", "reason": validation_reason},
                            ])
                            adaptive_limitation = "The proposed adaptive follow-up was not executed because it exceeded safety limits."

                findings = executor.findings_store.all()
                compact_findings = _compact_autonomous_findings(findings)
                evidence = [
                    {"tool_name": item["tool_name"], "result": item["result"]}
                    for item in compact_findings
                ]
                analytical_summary = _summarise_tool_results(evidence, has_figure=False)
                synthesis_parts = [
                    f"User question:\n{question}",
                    f"Completed autonomous plan:\n{json.dumps(autonomous_plan.to_dict(), default=str)}",
                ]
                if analytical_summary:
                    synthesis_parts.append(analytical_summary)
                if adaptive_limitation:
                    synthesis_parts.append(adaptive_limitation)
                synthesis_parts.append(
                    "Structured findings (use these for exact values and provenance):\n"
                    + json.dumps(compact_findings, default=str)
                )
                synthesis_parts.append("Now write the final answer to the user.")
                synthesis_messages = _compact_messages_for_request([
                    {"role": "system", "content": FINAL_EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n\n".join(synthesis_parts)},
                ], current_question=question)

                synthesis_succeeded = True
                try:
                    final_message = self.llm.chat(synthesis_messages, tool_choice="none")
                    answer = final_message.content
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Final synthesis returned no answer.")
                except Exception:
                    synthesis_succeeded = False
                    answer = _autonomous_fallback_answer(compact_findings)

                return {
                    "answer": answer,
                    "figure": None,
                    "trace": [
                        {"step": "question", "question": question},
                        *([routing_trace] if routing_trace else []),
                        {"step": "autonomous_plan", "plan": autonomous_plan.to_dict()},
                        {"step": "autonomous_execution", "findings": len(initial_compact_findings)},
                        *adaptive_trace,
                        {"step": "autonomous_synthesis", "success": synthesis_succeeded},
                        {"step": "final_answer", "tool_used": True, "autonomous": True},
                    ],
                    "findings": findings,
                }
            except Exception as exc:
                # Autonomous execution is optional; the established reactive
                # loop remains the fallback for invalid or unsupported plans.
                if autonomous_stage == "planner_call":
                    autonomous_stage, planner_output_received, planner_json_parsed = _planner_failure_stage(exc)
                elif executor is not None and getattr(executor, "diagnostic_stage", None):
                    autonomous_stage = executor.diagnostic_stage
                failing_step_id = getattr(exc, "step_id", None)
                if not failing_step_id and executor is not None and autonomous_stage == "execution":
                    failing_step_id = getattr(executor, "diagnostic_step_id", None)
                diagnostic_message = _sanitize_autonomous_diagnostic_text(exc)
                autonomous_fallback_trace = {
                    "step": "autonomous_fallback",
                    "stage": autonomous_stage,
                    "exception_type": f"{type(exc).__module__}.{type(exc).__name__}"[:200],
                    "message": diagnostic_message,
                    "planner_output_received": planner_output_received,
                    "planner_json_parsed": planner_json_parsed,
                    "plan_validated": plan_validated,
                }
                if plan_validated and autonomous_plan is not None:
                    autonomous_fallback_trace["plan_summary"] = _autonomous_plan_summary(autonomous_plan)
                if failing_step_id:
                    autonomous_fallback_trace["failing_step_id"] = str(failing_step_id)[:100]
                if autonomous_stage == "preflight":
                    autonomous_fallback_trace["reason"] = diagnostic_message

        dataset_context = format_datasets_context(self.active_datasets)
        # V8: append compact schema relationship map when 2+ datasets are present
        if len(self.active_datasets) >= 2:
            schema_graph = build_schema_graph_summary(self.active_datasets)
            if schema_graph:
                dataset_context = (dataset_context + "\n\n" + schema_graph).strip()
        system_content = SYSTEM_PROMPT
        if dataset_context:
            system_content += f"\n\n{dataset_context}"

        working_messages: list[dict[str, Any]] = (
            [{"role": "system", "content": system_content}]
            + history
            + [{"role": "user", "content": question}]
        )

        figure = None
        all_tool_results: list[dict[str, Any]] = []
        all_evidence_results: list[dict[str, Any]] = []
        executed_calls_cache: dict[tuple[str, str], Any] = {}
        trace: list[dict[str, Any]] = [{"step": "question", "question": question}]
        if routing_trace:
            trace.append(routing_trace)
        if autonomous_fallback_trace:
            trace.append(autonomous_fallback_trace)
        if execution_plan_signal:
            trace.append({"step": "plan", "summary": execution_plan_signal})
        iterations = 0
        safe_join_inspected = False
        joined_dataset_name = None
        joined_dataset_analyzed = False
        join_continuation_stages: set[str] = set()

        # ------------------------------------------------------------
        # MULTI-STEP TOOL-DECISION LOOP (bounded by MAX_TOOL_ITERATIONS)
        # ------------------------------------------------------------
        while iterations < MAX_TOOL_ITERATIONS:

            decision_messages = _compact_messages_for_request(
                working_messages, tools=TOOL_SCHEMAS, current_question=question
            )
            response_message = self.llm.chat(decision_messages, tools=TOOL_SCHEMAS)
            tool_calls = getattr(response_message, "tool_calls", None)

            if not tool_calls:
                if iterations == 0:
                    # No tool was ever needed -- answer directly.
                    trace.append({"step": "final_answer", "tool_used": False})
                    return {
                        "answer": response_message.content,
                        "figure": None,
                        "trace": trace,
                        "evidence": [],
                    }
                continuation_stage = None
                if join_pipeline_requested and safe_join_inspected and not joined_dataset_name:
                    continuation_stage = "execute_join"
                elif join_pipeline_requested and joined_dataset_name and not joined_dataset_analyzed:
                    continuation_stage = "analyze_join"
                if continuation_stage and continuation_stage not in join_continuation_stages:
                    join_continuation_stages.add(continuation_stage)
                    working_messages.append({
                        "role": "assistant",
                        "content": response_message.content or "(join workflow paused)",
                    })
                    if continuation_stage == "execute_join":
                        reminder = (
                            "The safe join check is only an intermediate result. The original request also "
                            "requires executing the safe join and analyzing its derived dataset. Continue "
                            "with execute_join, then perform the requested downstream analysis."
                        )
                    else:
                        reminder = (
                            f"The joined dataset is registered as '{joined_dataset_name}', but the original "
                            "downstream analysis is still incomplete. Call the appropriate analytical tool "
                            f"with dataset_name='{joined_dataset_name}' now."
                        )
                    working_messages.append({"role": "user", "content": reminder})
                    continue
                break  # LLM is done requesting tools; go explain results.

            round_results = []
            round_evidence_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                try:
                    tool_args = json.loads(raw_arguments)
                    if not isinstance(tool_args, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    args_malformed = False
                except (json.JSONDecodeError, ValueError):
                    tool_args = {}
                    args_malformed = True

                if args_malformed:
                    # Give the model a targeted, recoverable error instead
                    # of silently calling the tool with empty/wrong args.
                    required = _REQUIRED_PARAMS.get(tool_name, [])
                    tool_result: Any = {
                        "error": (
                            f"Arguments for '{tool_name}' were not valid JSON and could not "
                            "be parsed."
                        ),
                        "required_parameters": required,
                        "hint": "Retry this tool call with a valid JSON arguments object matching its schema.",
                    }
                    success = False
                    reused = False
                else:
                    selected_dataset_name = tool_args.get("dataset_name")
                    if (
                        explicit_dataset_name
                        and tool_name in _DATASET_SCOPED_TOOLS
                        and selected_dataset_name not in self.derived_datasets
                    ):
                        tool_args = dict(tool_args)
                        tool_args["dataset_name"] = explicit_dataset_name
                    call_key = (tool_name, json.dumps(tool_args, sort_keys=True))
                    if call_key in executed_calls_cache:
                        cached_result = executed_calls_cache[call_key]
                        if isinstance(cached_result, dict):
                            tool_result = dict(cached_result)
                            tool_result["duplicate_call_note"] = (
                                f"Tool '{tool_name}' was already executed with identical arguments in a previous step. "
                                "Reusing the computed result. Do not repeat identical calls."
                            )
                        else:
                            tool_result = cached_result
                        success = not (isinstance(tool_result, dict) and "error" in tool_result)
                        reused = True
                    else:
                        tool_result = self._execute_tool(tool_name, tool_args, self.primary_df)
                        success = not (isinstance(tool_result, dict) and "error" in tool_result)
                        reused = False
                        if isinstance(tool_result, dict):
                            executed_calls_cache[call_key] = dict(tool_result)
                        else:
                            executed_calls_cache[call_key] = tool_result

                if isinstance(tool_result, dict) and "figure" in tool_result:
                    figure = tool_result["figure"]
                    tool_result = {k: v for k, v in tool_result.items() if k != "figure"}

                if success and isinstance(tool_result, dict):
                    if tool_name == "inspect_join_viability" and tool_result.get("safe_to_join") is True:
                        safe_join_inspected = True
                    elif tool_name == "execute_join" and tool_result.get("status") == "success":
                        joined_dataset_name = tool_result.get("dataset_name")
                    elif (
                        joined_dataset_name
                        and tool_name not in {"inspect_join_viability", "execute_join"}
                        and selected_dataset_name == joined_dataset_name
                    ):
                        joined_dataset_analyzed = True

                evidence_result = tool_result
                llm_result = (
                    _compact_ml_result_for_llm(tool_result)
                    if tool_name == "train_ml_model" else tool_result
                )
                llm_result = _compact_tool_result(llm_result)
                round_results.append({"tool_name": tool_name, "result": llm_result})
                round_evidence_results.append({
                    "tool_name": tool_name,
                    "result": evidence_result if tool_name == "train_ml_model" else llm_result,
                })

                trace.append({
                    "step": "tool_call",
                    "iteration": iterations,
                    "tool": tool_name,
                    "arguments": tool_args if not args_malformed else raw_arguments,
                    "success": success,
                    "reused": reused,
                    "error": llm_result.get("error") if isinstance(llm_result, dict) else None,
                })

            all_tool_results.extend(round_results)
            all_evidence_results.extend(round_evidence_results)

            # Feed results back as PLAIN TEXT only -- never role="tool",
            # never response_message.tool_calls persisted.
            assistant_note = response_message.content or "(requested tool execution)"
            working_messages.append({"role": "assistant", "content": assistant_note})

            results_text = "\n\n".join(
                f"Tool: {item['tool_name']}\nResult: {json.dumps(item['result'], default=str)}"
                for item in round_results
            )
            working_messages.append({
                "role": "user",
                "content": (
                    f"[Tool results for your previous request]\n{results_text}\n\n"
                    "Instructions:\n"
                    "- If these results are sufficient to fully answer the user's question, respond with your final answer text now without calling any more tools.\n"
                    "- If a tool call failed (see 'error'), check the error message and hint, and retry once with corrected arguments.\n"
                    "- If another analytical step is genuinely required to complete the multi-step analysis (e.g., using entities found above in a subsequent chart or time breakdown), call the next appropriate tool now.\n"
                    "- Do NOT call the same tool again with identical arguments."
                ),
            })

            iterations += 1

        # ------------------------------------------------------------
        # FINAL EXPLANATION CALL (V5.4) -- isolated conversation,
        # no tools, tool_choice="none".  The prompt now includes:
        # 1. The improved grounding/insight system prompt.
        # 2. A deterministic pre-digested hint block (_summarise_tool_results).
        # 3. The raw tool result JSON (unchanged from previous versions).
        # ------------------------------------------------------------
        tool_results_text = "\n\n".join(
            f"Tool: {item['tool_name']}\nResult: {json.dumps(item['result'], default=str)}"
            for item in all_tool_results
        )

        # Deterministic hint block — no LLM call, pure Python extraction.
        analytical_summary = _summarise_tool_results(all_tool_results, figure is not None)

        user_content_parts = [
            f"User question:\n{question}",
        ]
        if dataset_context:
            user_content_parts.append(f"Dataset context:\n{dataset_context}")
        if analytical_summary:
            user_content_parts.append(analytical_summary)
        user_content_parts.append(
            f"Full tool results (use these for exact numbers):\n{tool_results_text}"
        )
        user_content_parts.append("Now write the final answer to the user.")

        final_messages = [
            {
                "role": "system",
                "content": FINAL_EXPLANATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "\n\n".join(user_content_parts),
            },
        ]

        final_messages = _compact_messages_for_request(
            final_messages, current_question=question
        )
        final_message = self.llm.chat(final_messages, tool_choice="none")

        trace.append({"step": "final_answer", "tool_used": True, "iterations": iterations})

        return {
            "answer": final_message.content,
            "figure": figure,
            "trace": trace,
            "evidence": list(all_evidence_results[:MAX_RETURNED_EVIDENCE]),
        }

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any], df: pd.DataFrame) -> Any:
        """Execute only tools explicitly registered in TOOL_FUNCTIONS."""
        tool_function = TOOL_FUNCTIONS.get(tool_name)

        if tool_function is None:
            return {
                "error": f"Unknown tool requested: '{tool_name}'",
                "available_tools": sorted(TOOL_FUNCTIONS.keys()),
                "hint": "Please select from one of the available registered tools.",
            }

        # ------------------------------------------------------------------
        # V8: RELATIONSHIP DISCOVERY INTERCEPT (read-only, multi-dataset)
        # discover_relationships receives all available datasets as a dict.
        # ------------------------------------------------------------------
        if tool_name == "discover_relationships":
            all_available = dict(getattr(self, "active_datasets", {}))
            all_available.update(getattr(self, "derived_datasets", {}))
            target_ds = tool_args.pop("dataset_name", None)
            min_conf = tool_args.pop("min_confidence", 0.4)
            try:
                return _discover_relationships_tool(
                    all_available,
                    target_dataset=target_ds,
                    min_confidence=float(min_conf) if min_conf is not None else 0.4,
                )
            except Exception as e:
                return {"error": f"Tool 'discover_relationships' failed: {e}"}

        # ------------------------------------------------------------------
        # V8: MULTI-DATASET VISUALIZATION INTERCEPT
        # create_multi_dataset_visualization receives the full datasets dict.
        # ------------------------------------------------------------------
        if tool_name == "create_multi_dataset_visualization":
            all_available = dict(getattr(self, "active_datasets", {}))
            all_available.update(getattr(self, "derived_datasets", {}))
            try:
                result = create_multi_dataset_visualization(
                    datasets=all_available,
                    **tool_args,
                )
            except TypeError as e:
                return {
                    "error": f"Tool 'create_multi_dataset_visualization' was called with invalid or missing arguments: {e}",
                    "hint": "Provide a 'series' list with dataset_name, x_column, and y_column for each trace.",
                }
            except Exception as e:
                return {"error": f"Tool 'create_multi_dataset_visualization' failed: {e}"}
            return result

        # ------------------------------------------------------------------
        # JOIN TOOL INTERCEPT (V7)
        # inspect_join_viability and execute_join require two named datasets
        # (left_dataset, right_dataset) resolved from active or derived stores.
        # ------------------------------------------------------------------
        if tool_name in ("inspect_join_viability", "execute_join"):
            left_name = tool_args.pop("left_dataset", None)
            right_name = tool_args.pop("right_dataset", None)

            all_available = dict(getattr(self, "active_datasets", {}))
            all_available.update(getattr(self, "derived_datasets", {}))

            if not left_name or left_name not in all_available:
                return {
                    "error": f"Left dataset '{left_name}' not found.",
                    "available_datasets": list(all_available.keys()),
                    "hint": "Provide a valid left_dataset name.",
                }
            if not right_name or right_name not in all_available:
                return {
                    "error": f"Right dataset '{right_name}' not found.",
                    "available_datasets": list(all_available.keys()),
                    "hint": "Provide a valid right_dataset name.",
                }

            left_df = all_available[left_name]
            right_df = all_available[right_name]

            try:
                result = tool_function(left_df, right_df, **tool_args)
            except TypeError as e:
                required = _REQUIRED_PARAMS.get(tool_name, [])
                return {
                    "error": f"Tool '{tool_name}' was called with invalid or missing arguments: {e}",
                    "required_parameters": required,
                    "hint": "Retry with all required parameters present and correctly named.",
                }
            except Exception as e:
                return {"error": f"Tool '{tool_name}' failed: {e}"}

            # Intercept raw DataFrame from successful execute_join
            if tool_name == "execute_join" and isinstance(result, dict) and result.get("status") == "success":
                joined_df = result.pop("dataframe")

                derived_name = self.register_derived_dataset(joined_df)

                return {
                    "status": "success",
                    "dataset_name": derived_name,
                    "shape": result.get("shape"),
                    "columns": result.get("columns"),
                    "cardinality": result.get("cardinality"),
                    "hint": f"Joined dataset registered as '{derived_name}'. Pass dataset_name='{derived_name}' to any analytical tool to query it.",
                }

            return result

        # ------------------------------------------------------------------
        # STANDARD SINGLE-DATASET ROUTING (V6)
        # ------------------------------------------------------------------
        dataset_name = tool_args.pop("dataset_name", None)
        target_df = df

        if dataset_name:
            # Handle explicit null or empty string gracefully
            active = getattr(self, "active_datasets", {})
            derived = getattr(self, "derived_datasets", {})
            if dataset_name in active:
                target_df = active[dataset_name]
            elif dataset_name in derived:
                target_df = derived[dataset_name]
            else:
                available = list(active.keys()) + list(derived.keys()) if (active or derived) else ["default"]
                return {
                    "error": f"Dataset '{dataset_name}' not found.",
                    "available_datasets": available,
                    "hint": "Check the spelling of the dataset_name or leave it empty to use the primary dataset."
                }

        try:
            return tool_function(target_df, **tool_args)
        except TypeError as e:
            # Distinguish "wrong/missing arguments" from a genuine internal
            # failure, and point the model at exactly what it needs to fix.
            required = _REQUIRED_PARAMS.get(tool_name, [])
            return {
                "error": f"Tool '{tool_name}' was called with invalid or missing arguments: {e}",
                "required_parameters": required,
                "hint": "Retry with all required parameters present and correctly named.",
            }
        except Exception as e:
            return {"error": f"Tool '{tool_name}' failed: {e}"}
