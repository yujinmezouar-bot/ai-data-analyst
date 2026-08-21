import json
from collections import OrderedDict
from typing import Any

import pandas as pd

from agent.llm import LLMClient

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
]

# Required parameters per tool, derived from the schemas above, used only
# to give the LLM a targeted retry hint when a tool call fails due to
# missing/invalid arguments (V4.1) -- not a second source of truth for
# validation, which still happens inside each tool.
_REQUIRED_PARAMS = {
    schema["function"]["name"]: schema["function"]["parameters"].get("required", [])
    for schema in TOOL_SCHEMAS
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
# Conservative character budget (~6k tokens) for a model environment with an
# 8k TPM limit. Tool schemas are included because providers count them too.
MAX_LLM_REQUEST_CHARS = 24000


def _build_execution_plan(question: str) -> str | None:
    """Return a concise, user-safe plan only for clearly multi-step requests."""
    text = question.lower()
    time_terms = ("trend", "over time", "monthly", "weekly", "quarter", "yearly")
    ranking_terms = ("top ", "bottom ", "best", "worst", "highest", "lowest")

    if "why" in text and any(term in text for term in ("change", "increase", "decrease", "grew", "decline")):
        return "Identify the observed change, then examine relevant groups for associated contributors."
    if any(term in text for term in time_terms) and any(term in text for term in ranking_terms):
        return "Rank the requested entities, then analyze the selected entities over time."
    if any(term in text for term in ("compare", "relationship", "correlation")) and any(
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
    for index in range(len(messages) - 2, 0, -1):
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
        allowance = max(500, MAX_LLM_REQUEST_CHARS - _estimate_request_chars(compacted[:-1], tools) - 200)
        last = dict(last)
        last["content"] = content[:allowance] + "\n[Context compacted for request size.]"
        compacted[-1] = last
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

    def run(
        self,
        question: str,
        df: pd.DataFrame = None,
        conversation_history: list[dict[str, str]] | None = None,
        datasets: dict[str, pd.DataFrame] = None,
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
        executed_calls_cache: dict[tuple[str, str], Any] = {}
        trace: list[dict[str, Any]] = [{"step": "question", "question": question}]
        plan = _build_execution_plan(question)
        if plan:
            trace.append({"step": "plan", "summary": plan})
        iterations = 0

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
                    return {"answer": response_message.content, "figure": None, "trace": trace}
                break  # LLM is done requesting tools; go explain results.

            round_results = []
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

                tool_result = _compact_tool_result(tool_result)
                round_results.append({"tool_name": tool_name, "result": tool_result})

                trace.append({
                    "step": "tool_call",
                    "iteration": iterations,
                    "tool": tool_name,
                    "arguments": tool_args if not args_malformed else raw_arguments,
                    "success": success,
                    "reused": reused,
                    "error": tool_result.get("error") if isinstance(tool_result, dict) else None,
                })

            all_tool_results.extend(round_results)

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

        return {"answer": final_message.content, "figure": figure, "trace": trace}

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

                self._derived_count = getattr(self, "_derived_count", 0) + 1
                derived_name = f"derived_join_{self._derived_count}"

                while len(self.derived_datasets) >= self._MAX_DERIVED_DATASETS:
                    self.derived_datasets.popitem(last=False)

                self.derived_datasets[derived_name] = joined_df

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
