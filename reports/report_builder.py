from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from tools.dataset_info import build_dataset_profile


MAX_REPORT_FINDINGS = 20
MAX_REPORT_DATASETS = 10
MAX_REPORT_COLUMNS = 20
MAX_REPORT_CONTRIBUTORS = 10
MAX_REPORT_RESULT_ITEMS = 25
MAX_EXECUTIVE_SUMMARY_CHARS = 6000


@dataclass
class AnalysisReport:
    title: str
    objective: str
    executive_summary: str
    dataset_overview: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    methodology: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _list_values(values: Any) -> str:
    if not isinstance(values, list):
        return _value(values)
    return ", ".join(_value(value) for value in values[:10])


def _bounded_result(value: Any, depth: int = 0) -> Any:
    """Retain exact bounded evidence while excluding render-heavy runtime objects."""
    if depth >= 5:
        return "Further nested detail omitted."
    if isinstance(value, pd.DataFrame):
        return {"note": "Tabular artifact omitted from report evidence.", "rows": len(value), "columns": len(value.columns)}
    if hasattr(value, "to_plotly_json"):
        return {"note": "Interactive Plotly object omitted from Markdown evidence."}
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {
            str(key): _bounded_result(item, depth + 1)
            for key, item in items[:MAX_REPORT_RESULT_ITEMS]
        }
        if len(items) > MAX_REPORT_RESULT_ITEMS:
            bounded["report_note"] = f"Mapping truncated to {MAX_REPORT_RESULT_ITEMS} entries."
        return bounded
    if isinstance(value, (list, tuple)):
        bounded = [_bounded_result(item, depth + 1) for item in value[:MAX_REPORT_RESULT_ITEMS]]
        if len(value) > MAX_REPORT_RESULT_ITEMS:
            bounded.append(f"List truncated to {MAX_REPORT_RESULT_ITEMS} entries.")
        return bounded
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:1000] if isinstance(value, str) else value
    return str(value)[:1000]


def _dataset_overview(datasets: Mapping[str, pd.DataFrame] | None) -> list[dict[str, Any]]:
    overview = []
    for name, dataframe in list((datasets or {}).items())[:MAX_REPORT_DATASETS]:
        profile = build_dataset_profile(dataframe)
        if "error" in profile:
            continue
        temporal = []
        for column in profile.get("datetime_columns", [])[:3]:
            period_profile = profile.get("date_column_details", {}).get(column, {}).get("period_profile")
            if period_profile:
                temporal.append({
                    "column": column,
                    "min_date": period_profile.get("min_date"),
                    "max_date": period_profile.get("max_date"),
                })
        missing = [
            {"column": column, **info}
            for column, info in list(profile.get("missing_summary", {}).items())[:10]
        ]
        columns = [str(column) for column in profile.get("column_names", [])]
        overview.append({
            "name": str(name),
            "rows": profile.get("num_rows", 0),
            "columns": profile.get("num_columns", 0),
            "column_names": columns[:MAX_REPORT_COLUMNS],
            "columns_truncated": len(columns) > MAX_REPORT_COLUMNS,
            "numeric_columns": profile.get("numeric_columns", [])[:10],
            "datetime_coverage": temporal,
            "missing": missing,
        })
    return overview


def _finding_value(finding: Any, key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _trace_tool_calls(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in trace if entry.get("step") == "tool_call"]


def _normalize_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    findings = result.get("findings") or []
    if findings:
        for index, finding in enumerate(findings[:MAX_REPORT_FINDINGS], 1):
            metadata = dict(_finding_value(finding, "metadata", {}) or {})
            provenance = dict(_finding_value(finding, "provenance", {}) or {})
            normalized.append({
                "evidence_id": f"F{index}",
                "finding_id": str(_finding_value(finding, "id", "")),
                "step_id": str(_finding_value(finding, "step_id", "")),
                "tool_name": str(_finding_value(finding, "tool_name", "")),
                "datasets": list(_finding_value(finding, "datasets", []) or []),
                "plan_id": metadata.get("plan_id") or provenance.get("plan_id"),
                "result": _bounded_result(_finding_value(finding, "result")),
                "metadata": metadata,
                "provenance": provenance,
            })
        return normalized

    trace_calls = _trace_tool_calls(result.get("trace", []) or [])
    for index, evidence in enumerate((result.get("evidence") or [])[:MAX_REPORT_FINDINGS], 1):
        tool_name = str(evidence.get("tool_name", ""))
        trace_entry = trace_calls[index - 1] if index <= len(trace_calls) else {}
        arguments = trace_entry.get("arguments") if isinstance(trace_entry.get("arguments"), dict) else {}
        dataset_name = arguments.get("dataset_name") or arguments.get("dataset")
        normalized.append({
            "evidence_id": f"F{index}",
            "finding_id": "",
            "step_id": "",
            "tool_name": tool_name,
            "datasets": [dataset_name] if dataset_name else [],
            "plan_id": None,
            "result": _bounded_result(evidence.get("result")),
            "metadata": {"arguments": arguments},
            "provenance": {},
        })
    return normalized


def _generic_summary(tool: str, result: Any) -> str:
    if not isinstance(result, dict):
        return f"{tool} produced structured evidence."
    if "error" in result:
        return f"{tool} did not complete: {_value(result['error'])}."
    if tool == "statistics":
        column = result.get("column", "selected columns")
        values = [
            f"{key}={_value(result[key])}" for key in ("count", "mean", "median", "min", "max")
            if result.get(key) is not None
        ]
        return f"Statistics for {column}: {', '.join(values)}." if values else f"Statistics for {column}."
    if tool == "groupby_analysis":
        best, worst = result.get("best_group"), result.get("worst_group")
        parts = [
            f"best group={best}" if best is not None else "",
            f"worst group={worst}" if worst is not None and worst != best else "",
        ]
        return "Grouped analysis: " + ", ".join(part for part in parts if part) + "."
    if tool == "time_analysis":
        parts = [
            f"trend={result.get('trend_direction')}" if result.get("trend_direction") else "",
            f"overall change={_value(result.get('overall_change'))}" if result.get("overall_change") is not None else "",
            f"best period={result.get('best_period')}" if result.get("best_period") else "",
            f"worst period={result.get('worst_period')}" if result.get("worst_period") else "",
        ]
        return "Time analysis: " + ", ".join(part for part in parts if part) + "."
    if tool == "percentage_change":
        return (
            f"Period comparison: absolute change={_value(result.get('absolute_change', result.get('overall_change')))}, "
            f"percentage change={_value(result.get('percentage_change', result.get('overall_percentage_change')))}%."
        )
    if tool == "missing_values":
        return f"Missing-value analysis found {_value(result.get('total_missing_values'))} missing value(s)."
    if tool == "correlation_analysis":
        pair = result.get("strongest_positive_pair") or result.get("strongest_negative_pair")
        if isinstance(pair, dict):
            return (
                f"Strongest reported association: {pair.get('column_1')} and {pair.get('column_2')} "
                f"(r={_value(pair.get('correlation'))})."
            )
        return "Correlation analysis measured statistical associations between numeric variables."
    if tool == "kpi_contribution_analysis":
        overall = result.get("overall", {})
        return (
            f"KPI contribution analysis measured an observed {overall.get('direction', 'change')} "
            f"from {_value(overall.get('value_a'))} to {_value(overall.get('value_b'))}."
        )
    if tool == "train_ml_model":
        return (
            f"Supervised {result.get('task_type')} evaluated target "
            f"{result.get('target_column')} on {result.get('rows_used')} usable rows; "
            f"selected model={result.get('best_model')}."
        )
    scalar_parts = []
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar_parts.append(f"{key}={_value(value)}")
        if len(scalar_parts) == 5:
            break
    return f"{tool}: " + (", ".join(scalar_parts) + "." if scalar_parts else "structured result recorded.")


def _contains_marker(value: Any, marker: str) -> bool:
    if isinstance(value, dict):
        return any(marker in str(key).lower() or _contains_marker(item, marker) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_marker(item, marker) for item in value)
    return marker in str(value).lower()


def _is_truncated(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("truncated") is True or "truncated_preview" in value:
            return True
        if "truncat" in str(value.get("note", "")).lower():
            return True
        return any(_is_truncated(item) for key, item in value.items() if key != "truncated")
    if isinstance(value, list):
        return any(_is_truncated(item) for item in value)
    return "truncat" in str(value).lower()


def _derive_limitations(
    result: dict[str, Any], evidence: list[dict[str, Any]], dataset_overview: list[dict[str, Any]]
) -> list[str]:
    limitations = []
    tools = {item["tool_name"] for item in evidence}
    if not evidence:
        limitations.append("Structured analytical evidence was unavailable; this report is limited to the recorded response and dataset metadata.")
    if "correlation_analysis" in tools:
        limitations.append("Correlation findings describe statistical association and do not establish causation.")
    if "kpi_contribution_analysis" in tools:
        limitations.append("KPI contributions are mathematical allocations of observed net change and do not establish causation.")
    if "train_ml_model" in tools:
        limitations.append("ML performance is estimated on one held-out test split, not external validation; predictive feature associations do not establish causation.")
    for item in evidence:
        evidence_result = item.get("result")
        if _is_truncated(evidence_result):
            limitations.append("One or more analytical results were truncated; displayed evidence may not include every group or period.")
        if _contains_marker(evidence_result, "insufficient"):
            limitations.append(f"{item['tool_name']} reported insufficient evidence or data for a complete conclusion.")
        if isinstance(evidence_result, dict) and evidence_result.get("error"):
            limitations.append(f"{item['tool_name']} failed: {_value(evidence_result['error'])}.")
        excluded = evidence_result.get("excluded_rows") if isinstance(evidence_result, dict) else None
        if isinstance(excluded, dict) and excluded.get("total_excluded", 0):
            limitations.append(f"{excluded['total_excluded']} row(s) were excluded from {item['tool_name']} because required values were missing.")
        if item["tool_name"] == "train_ml_model" and isinstance(evidence_result, dict):
            for limitation in (evidence_result.get("limitations") or [])[:10]:
                limitations.append(str(limitation))
    for entry in result.get("trace", []) or []:
        if entry.get("step") == "tool_call" and entry.get("success") is False:
            limitations.append(f"Tool step {entry.get('tool', 'unknown')} failed and could not provide evidence.")
        if entry.get("step") in {"adaptive_review", "adaptive_execution"} and entry.get("status") in {"failed", "invalid"}:
            limitations.append(f"Adaptive {entry['step'].replace('adaptive_', '')} did not complete successfully.")
        if entry.get("step") == "adaptive_stop" and entry.get("reason") not in {"review_complete"}:
            limitations.append(f"Adaptive investigation stopped: {entry.get('reason')}.")
        if entry.get("step") == "autonomous_synthesis" and entry.get("success") is False:
            limitations.append("Final autonomous synthesis failed; the recorded answer used deterministic fallback evidence.")
        if entry.get("status") in {"failed", "skipped"} and entry.get("step") not in {
            "adaptive_review", "adaptive_execution"
        }:
            limitations.append(f"Analysis step {entry.get('step', 'unknown')} was {entry.get('status')}.")
    if any(dataset.get("missing") for dataset in dataset_overview):
        limitations.append("Uploaded data contains missing values; affected columns are listed in the dataset overview.")
    if result.get("figure") is not None:
        limitations.append("The interactive chart shown in Streamlit is not embedded in the downloaded Markdown report.")
    return _unique(limitations)


def _visualization_metadata(result: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figure = result.get("figure")
    if figure is None:
        return []
    title = None
    trace_count = None
    try:
        title = getattr(getattr(figure.layout, "title", None), "text", None)
        trace_count = len(figure.data)
    except Exception:
        pass
    metadata = next(
        (item.get("result") for item in evidence if item.get("tool_name") in {
            "create_visualization", "create_multi_dataset_visualization"
        } and isinstance(item.get("result"), dict)),
        {},
    )
    return [{
        "title": title or metadata.get("description") or "Analysis visualization",
        "chart_type": metadata.get("chart_type"),
        "description": metadata.get("description"),
        "trace_count": trace_count,
    }]


def _provenance_line(item: dict[str, Any]) -> str:
    identity = item["evidence_id"]
    if item.get("finding_id"):
        identity += f" ({item['finding_id']})"
    line = f"{identity} — {item['tool_name']}"
    if item.get("datasets"):
        line += " — " + ", ".join(str(name) for name in item["datasets"])
    result = item.get("result")
    if item["tool_name"] == "kpi_contribution_analysis" and isinstance(result, dict):
        line += (
            f" — {result.get('group_column', 'group')} contributions"
            f" — {result.get('period_a', '?')} → {result.get('period_b', '?')}"
        )
    if item.get("step_id"):
        line += f" — step {item['step_id']}"
    if item.get("plan_id"):
        line += f" — plan {item['plan_id']}"
    return line


def build_analysis_report(
    question: str,
    analysis_result: dict[str, Any],
    datasets: Mapping[str, pd.DataFrame] | None = None,
    title: str = "AI Data Analysis Report",
) -> AnalysisReport:
    """Build report presentation state from completed artifacts without executing analysis."""
    objective = question
    overview = _dataset_overview(datasets)
    evidence = _normalize_evidence(analysis_result)
    for item in evidence:
        item["summary"] = _generic_summary(item["tool_name"], item.get("result"))
    tools = _unique([item["tool_name"] for item in evidence])
    visualizations = _visualization_metadata(analysis_result, evidence)
    limitations = _derive_limitations(analysis_result, evidence, overview)
    provenance = [_provenance_line(item) for item in evidence]
    return AnalysisReport(
        title=title,
        objective=str(objective or question),
        executive_summary=str(analysis_result.get("answer") or "No final answer was recorded.")[:MAX_EXECUTIVE_SUMMARY_CHARS],
        dataset_overview=overview,
        findings=evidence,
        visualizations=visualizations,
        limitations=limitations,
        methodology=tools or ["No deterministic analytical tool execution was recorded."],
        provenance=provenance,
    )


def _render_contribution(item: dict[str, Any]) -> tuple[list[str], list[str]]:
    result = item.get("result")
    if not isinstance(result, dict):
        return [], []
    overall = result.get("overall", {}) if isinstance(result.get("overall"), dict) else {}
    kpi = [
        f"### [{item['evidence_id']}] {_value(result.get('metric_column'))}: {_value(result.get('period_a'))} → {_value(result.get('period_b'))}",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Period A total | {_value(overall.get('value_a'))} |",
        f"| Period B total | {_value(overall.get('value_b'))} |",
        f"| Absolute change | {_value(overall.get('absolute_change'))} |",
        f"| Percentage change | {_value(overall.get('percentage_change'))}% |",
        f"| Direction | {_value(overall.get('direction'))} |",
    ]
    if result.get("filter_applied"):
        applied = result["filter_applied"]
        kpi.extend(["", f"Filter: `{_value(applied.get('column'))}` in {_list_values(applied.get('values'))}."])
    drivers = []
    contributors = result.get("contributors") if isinstance(result.get("contributors"), list) else []
    for contributor in contributors[:MAX_REPORT_CONTRIBUTORS]:
        effect = str(contributor.get("effect", ""))
        if effect.startswith("reinforces_"):
            wording = "mathematical contributor to the observed " + str(overall.get("direction", "change"))
        elif effect.startswith("offsets_"):
            wording = "offset part of the observed " + str(overall.get("direction", "change"))
        else:
            wording = "group movement"
        drivers.append(
            f"- **{_value(contributor.get('group'))}** — {wording}: "
            f"change {_value(contributor.get('absolute_change'))}; "
            f"share of net change {_value(contributor.get('contribution_to_total_change_percentage'))}%."
        )
    excluded = result.get("excluded_rows")
    if isinstance(excluded, dict) and excluded.get("total_excluded", 0):
        kpi.extend(["", f"Excluded rows: {_value(excluded.get('total_excluded'))}."])
    if result.get("truncated"):
        kpi.extend(["", "Contributor details were truncated; overall totals still use all valid groups."])
    return kpi, drivers


def _render_ml_result(item: dict[str, Any]) -> list[str]:
    result = item.get("result")
    if not isinstance(result, dict):
        return []
    split = result.get("split", {}) if isinstance(result.get("split"), dict) else {}
    lines = [
        f"### [{item['evidence_id']}] {_value(result.get('task_type'))}: `{_value(result.get('target_column'))}`",
        "",
        f"- Rows used: {_value(result.get('rows_used'))}",
        f"- Features used: {_list_values(result.get('features_used'))}",
        f"- Split: {_value(split.get('strategy'))}; train={_value(split.get('train_rows'))}, "
        f"test={_value(split.get('test_rows'))}, test size={_value(split.get('test_size'))}, "
        f"random state={_value(split.get('random_state'))}",
        f"- Selected model: **{_value(result.get('best_model'))}** using {_value(result.get('selection_metric'))}",
        "",
        "#### Held-out Test Metrics",
        "",
    ]
    if split.get("group_aware"):
        lines[6:6] = [
            f"- Group isolation: `{_value(split.get('group_column'))}`; "
            f"total={_value(split.get('total_groups'))}, train={_value(split.get('train_groups'))}, "
            f"test={_value(split.get('test_groups'))}; groups do not cross train/test.",
        ]
    models = [model for model in (result.get("models") or [])[:2] if isinstance(model, dict)]
    predictive_model = next((model for model in models if not model.get("baseline")), None)
    predictive_won = bool(predictive_model and result.get("best_model") == predictive_model.get("name"))
    if predictive_model is not None and not predictive_won:
        lines.extend([
            "The predictive model did not outperform the naive baseline on the selected evaluation metric.",
            "",
        ])
    for model in models:
        role = "baseline" if model.get("baseline") else "predictive"
        lines.append(f"- **{_value(model.get('name'))}** ({role})")
        metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
        for name, value in metrics.items():
            if name in {"confusion_matrix", "class_labels"}:
                continue
            lines.append(f"  - {name}: {_value(value)}")
            if name == "r2" and isinstance(value, (int, float)) and value < 0:
                lines.append(
                    "    - Negative R²: this model performed worse than a simple mean-prediction "
                    "reference on this held-out evaluation."
                )
    associations = result.get("feature_associations") if isinstance(result.get("feature_associations"), list) else []
    if associations:
        lines.extend(["", "#### Top Predictive Feature Associations", ""])
        if not predictive_won:
            lines.extend([
                "These are coefficients from a predictive model that did not outperform the naive baseline; "
                "they are not validated useful predictors.",
                "",
            ])
        for association in associations[:MAX_REPORT_CONTRIBUTORS]:
            if isinstance(association, dict):
                lines.append(
                    f"- **{_value(association.get('feature'))}**: score "
                    f"{_value(association.get('association_score'))}; relative share "
                    f"{_value(association.get('relative_share_percentage'))}%."
                )
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    if warnings:
        lines.extend(["", "#### ML Warnings", "", *[f"- {_value(value)}" for value in warnings[:10]]])
    lines.extend([
        "",
        "Performance is estimated on the held-out test split; one split is not external validation. "
        "Feature associations are predictive and do not imply causation.",
    ])
    return lines


def render_markdown(report: AnalysisReport) -> str:
    """Render a stable, bounded Markdown report."""
    lines = [
        f"# {report.title}",
        "",
        "## Analysis Objective",
        "",
        report.objective,
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Dataset Overview",
        "",
    ]
    if report.dataset_overview:
        for dataset in report.dataset_overview:
            lines.extend([
                f"### {dataset['name']}",
                "",
                f"- Shape: {dataset['rows']} rows × {dataset['columns']} columns",
                f"- Columns: {', '.join(_value(column) for column in dataset['column_names'])}" + (" (truncated)" if dataset["columns_truncated"] else ""),
            ])
            for temporal in dataset["datetime_coverage"]:
                lines.append(
                    f"- Date coverage for `{temporal['column']}`: {temporal['min_date']} → {temporal['max_date']}"
                )
    else:
        lines.append("No dataset profile was supplied.")

    lines.extend(["", "## Key Findings" if report.findings else "## Evidence Availability", ""])
    if report.findings:
        for item in report.findings:
            lines.append(f"- **[{item['evidence_id']}]** {item['summary']}")
    else:
        lines.append("No structured analytical findings were recorded for this response.")

    contribution_items = [item for item in report.findings if item["tool_name"] == "kpi_contribution_analysis"]
    if contribution_items:
        lines.extend(["", "## KPI / Change Evidence", ""])
        driver_lines = []
        for item in contribution_items:
            kpi_lines, drivers = _render_contribution(item)
            lines.extend(kpi_lines + [""])
            driver_lines.extend(drivers)
        if driver_lines:
            lines.extend(["## Drivers and Offsets", "", *driver_lines, ""])

    ml_items = [item for item in report.findings if item["tool_name"] == "train_ml_model"]
    if ml_items:
        lines.extend(["## Machine Learning Results", ""])
        for item in ml_items:
            lines.extend(_render_ml_result(item) + [""])

    quality_lines = []
    for dataset in report.dataset_overview:
        for missing in dataset["missing"]:
            quality_lines.append(
                f"- **{dataset['name']}.{missing['column']}**: {missing.get('missing_count')} missing "
                f"({missing.get('missing_percentage')}%)."
            )
    if quality_lines:
        lines.extend(["## Data-quality Observations", "", *quality_lines, ""])

    if report.visualizations:
        lines.extend(["## Visualizations", ""])
        for visualization in report.visualizations:
            details = [visualization.get("chart_type"), visualization.get("description")]
            details = [str(value) for value in details if value]
            lines.append(f"- **{visualization['title']}**" + (f" — {'; '.join(details)}" if details else ""))
        lines.append("")

    if report.limitations:
        lines.extend(["## Limitations", "", *[f"- {item}" for item in report.limitations], ""])

    lines.extend(["## Methodology", ""])
    lines.extend(f"- `{tool}`" if " " not in tool else f"- {tool}" for tool in report.methodology)

    if report.provenance:
        lines.extend(["", "## Provenance / Evidence Appendix", ""])
        lines.extend(f"- {item}" for item in report.provenance)

    return "\n".join(lines).strip() + "\n"
