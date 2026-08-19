import json
from typing import Any

import pandas as pd

from agent.llm import LLMClient

from tools.dataset_info import dataset_info, DATASET_INFO_SCHEMA, format_dataset_context
from tools.missing_values import missing_values, MISSING_VALUES_SCHEMA
from tools.statistics import statistics, STATISTICS_SCHEMA
from tools.groupby import groupby_analysis, GROUPBY_ANALYSIS_SCHEMA
from tools.visualization import create_visualization, CREATE_VISUALIZATION_SCHEMA
from tools.time_analysis import time_analysis, TIME_ANALYSIS_SCHEMA
from tools.correlation import correlation_analysis, CORRELATION_ANALYSIS_SCHEMA
from tools.outliers import outlier_analysis, OUTLIER_ANALYSIS_SCHEMA
from tools.period_comparison import percentage_change, PERCENTAGE_CHANGE_SCHEMA


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

    for item in all_tool_results:
        tool = item["tool_name"]
        res = item["result"]
        if not isinstance(res, dict):
            continue

        if "error" in res:
            any_error = True
            hints.append(f"[{tool}] FAILED: {res['error']}")
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
    "You are a professional data analyst assistant working with a pandas DataFrame. "

    "IMPORTANT: The actual dataset is available to the Python tools. You must NEVER "
    "claim that the actual data values are unavailable. "

    "CONVERSATIONAL QUESTIONS: For greetings, general capabilities, or questions not "
    "requiring dataset calculations (e.g. 'hello', 'what can you do?'), respond directly "
    "with helpful text without calling any tools. "

    "TOOL SELECTION GUIDELINES: "
    "- Match the question to the minimum sufficient analysis: rankings and category comparisons use groupby_analysis; "
    "descriptive distributions use statistics; missing-data checks use missing_values; trends use time_analysis; "
    "period-over-period comparisons use percentage_change; relationships use correlation_analysis; "
    "unusual values use outlier_analysis; and dataset structure uses dataset_info only when the compact dataset context is insufficient. "
    "- Use dataset_info ONLY for questions about dataset structure (rows, columns, "
    "column names, data types, detected date columns, memory usage). "

    "- Use missing_values for questions about missing, null, or NaN values and completeness. "

    "- Use statistics for descriptive statistics (mean, median, std, min, max, "
    "quartiles) of one or all numeric columns, when there is NO grouping or time "
    "breakdown involved. "

    "- Use groupby_analysis when the user asks for a calculation BY, PER, FOR EACH, "
    "or ACROSS a CATEGORICAL column (not a time breakdown). Set top_n + sort_order "
    "for 'top N' / 'bottom N' / 'highest' / 'lowest' questions. Set filter_values "
    "to restrict to specific categories mentioned earlier in the conversation. "

    "- Use time_analysis whenever the question involves day/week/month/quarter/year "
    "breakdowns, trends over time, a specific year, a date range, comparing years, "
    "or which period was highest/lowest/best/worst. time_analysis returns "
    "best_period and worst_period directly -- read them, do not compute them "
    "yourself. To compare specific categories over time (e.g. 'compare those "
    "stores by month'), set time_analysis's group_column and filter_values. "

    "- Use correlation_analysis for questions about relationships between numeric "
    "variables, e.g. 'what is correlated with sales?' or 'strongest correlations "
    "in the dataset'. "

    "- Use outlier_analysis for questions about unusual/extreme values, e.g. 'are "
    "there outliers in sales?' or 'which columns have many outliers?'. "

    "- Use percentage_change for questions comparing periods, e.g. 'how did sales "
    "change from 2024 to 2025?', 'percentage increase in sales', 'compare this "
    "month to the previous month', or 'which month had the largest increase?'. "
    "It already computes absolute_change and percentage_change (including safe "
    "handling when the previous value is zero) -- read them, do not compute them "
    "yourself. "

    "- Use create_visualization for charts. If the request implies an aggregated "
    "view ('average sales by store', 'sales by month', 'top 10 stores'), you MUST "
    "set agg_function (and period, for a datetime x_column) -- otherwise the chart "
    "plots raw rows instead of the aggregated values, which is wrong. "

    "EFFICIENCY AND FOLLOW-UPS: "
    "- Use the compact dataset context and recent conversation to resolve column names, entities, and references such as 'their', 'those stores', or 'the worst one'. "
    "- If a prior tool result already contains the requested ranking, best/worst group, comparison, trend, or correlation, use it; do not call another tool merely to recompute it. "
    "- For an entity/subset workflow, first obtain the entities with groupby_analysis when needed, then pass the exact returned names as filter_values to the next tool. Never invent filter values. "
    "- Stop as soon as available results answer the question. Use create_visualization only when a chart materially improves a trend, distribution, multi-category comparison, or numeric relationship. "
    "- If the dataset context and conversation cannot identify the relevant column, entity, or comparison unambiguously, ask one concise clarification rather than guessing. "

    "MULTI-STEP REASONING: "
    "For questions that require multiple steps (for example: identifying the top 3 "
    "categories and then charting or trending them over time): "
    "1. Call the initial tool (e.g. groupby_analysis with top_n=3). "
    "2. Inspect the result to extract the top category names. "
    "3. Call the next tool (e.g. time_analysis or create_visualization) passing those "
    "names into filter_values. "
    "4. Stop calling tools once you have collected all data necessary to answer. "

    "STATISTICAL & COMPARATIVE REASONING: "
    "- Clearly distinguish difference, percentage difference, and percentage change. "
    "Difference = B - A; Percentage difference = |B - A| / ((A+B)/2); Percentage change = (B - A) / A. "
    "The Python tools calculate these exact values for you. "
    "- Clearly distinguish correlation from causation. A strong statistical correlation "
    "does NOT prove causation. State findings as associations or relationships, never causation. "
    "- Read trend directions (strictly_increasing, increasing, decreasing, stable, fluctuating) "
    "directly from time_analysis results rather than eyeballing or extrapolating. "
    "- For 'why did X change?' questions, investigate contributing sub-categories (e.g. by comparing "
    "group breakdowns) after identifying the change with time_analysis or percentage_change when applicable. "
    "Report observed contributors as associations only; do not claim they caused the change or invent external real-world causes. "
    "- Call create_visualization when the user asks for a chart or when a visual trend/distribution "
    "adds strong analytical value; do not generate charts for simple scalar lookups. "

    "GENERAL PRINCIPLES: "
    "- Always use a tool when the question requires facts or numbers from the dataset. "
    "- Never guess, invent, or manually calculate numbers yourself -- the Python tools "
    "perform every calculation. "
    "- Do not call unrelated tools or repeat the same tool call if the result is already available. "
    "- Resolve references from conversation context (e.g. 'those stores', 'the top one', 'for 2024') "
    "to the correct column names or values before calling tools."
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

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(
        self,
        question: str,
        df: pd.DataFrame,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:

        history = conversation_history or []
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]

        dataset_context = format_dataset_context(df) if df is not None and not df.empty else ""
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
        iterations = 0

        # ------------------------------------------------------------
        # MULTI-STEP TOOL-DECISION LOOP (bounded by MAX_TOOL_ITERATIONS)
        # ------------------------------------------------------------
        while iterations < MAX_TOOL_ITERATIONS:

            response_message = self.llm.chat(working_messages, tools=TOOL_SCHEMAS)
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
                    else:
                        tool_result = self._execute_tool(tool_name, tool_args, df)
                        success = not (isinstance(tool_result, dict) and "error" in tool_result)
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

        try:
            return tool_function(df, **tool_args)
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
