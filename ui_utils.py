"""Small, testable helpers for Streamlit upload and user-facing errors."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from tools.date_utils import convert_date_columns


TOOL_DISPLAY_NAMES = {
    "dataset_info": "Dataset Information",
    "missing_values": "Missing Values Analysis",
    "statistics": "Descriptive Statistics",
    "groupby_analysis": "Groupby Analysis",
    "create_visualization": "Visualization",
    "create_multi_dataset_visualization": "Multi-dataset Visualization",
    "time_analysis": "Time Analysis",
    "percentage_change": "Percentage Change",
    "kpi_contribution_analysis": "KPI Contribution Analysis",
    "correlation_analysis": "Correlation Analysis",
    "outlier_analysis": "Outlier Analysis",
    "discover_relationships": "Relationship Discovery",
    "inspect_join_viability": "Join Safety Check",
    "execute_join": "Dataset Join",
    "train_ml_model": "ML Model Training",
}


def build_analysis_details(result: dict[str, Any] | None) -> dict[str, Any]:
    """Build a safe, presentation-only summary from an existing Agent result."""
    result = result or {}
    trace = result.get("trace") if isinstance(result.get("trace"), list) else []
    findings = result.get("findings") if isinstance(result.get("findings"), list) else []
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []

    fallback = any(entry.get("step") == "autonomous_fallback" for entry in trace)
    autonomous = any(
        entry.get("step") == "autonomous_plan"
        or (entry.get("step") == "final_answer" and entry.get("autonomous") is True)
        for entry in trace
    )
    mode = (
        "Autonomous → reactive fallback"
        if fallback else "Autonomous analysis" if autonomous else "Reactive analysis"
    )

    raw_tools: list[str] = []
    if autonomous and not fallback and findings:
        raw_tools.extend(
            str(getattr(finding, "tool_name", "") or (
                finding.get("tool_name", "") if isinstance(finding, dict) else ""
            ))
            for finding in findings
        )
    else:
        raw_tools.extend(
            str(entry.get("tool", ""))
            for entry in trace if entry.get("step") == "tool_call"
        )
        raw_tools.extend(
            str(item.get("tool_name", ""))
            for item in evidence if isinstance(item, dict)
        )
    ordered_tools = list(dict.fromkeys(tool for tool in raw_tools if tool))

    initial_plan_steps = None
    for entry in trace:
        if entry.get("step") == "autonomous_plan" and isinstance(entry.get("plan"), dict):
            steps = entry["plan"].get("steps")
            if isinstance(steps, list):
                initial_plan_steps = len(steps)
            break

    adaptive_review = next(
        (entry for entry in trace if entry.get("step") == "adaptive_review"), None
    )
    adaptive_follow_up = bool(
        adaptive_review and adaptive_review.get("status") == "follow_up"
    )
    adaptive_steps_executed = sum(
        int(entry.get("executed_steps", 0) or 0)
        for entry in trace
        if entry.get("step") == "adaptive_execution"
        and entry.get("status") == "completed"
    )

    limitations: list[str] = []
    if "kpi_contribution_analysis" in ordered_tools:
        limitations.append("KPI contributions are mathematical decomposition, not causation.")
    if "correlation_analysis" in ordered_tools:
        limitations.append("Correlations are statistical associations, not causation.")
    if "train_ml_model" in ordered_tools:
        limitations.append("ML metrics are held-out evaluation estimates, not external validation.")
    if autonomous and adaptive_review is not None:
        limitations.append("Adaptive investigation is bounded to one small follow-up round.")
    if fallback:
        limitations.append("The autonomous attempt stopped safely and the reactive path produced the response.")
    if any(
        entry.get("step") == "tool_call" and entry.get("success") is False
        for entry in trace
    ):
        limitations.append("At least one tool attempt failed; available evidence was preserved.")
    if any(
        entry.get("step") in {"adaptive_review", "adaptive_execution"}
        and entry.get("status") in {"failed", "invalid"}
        for entry in trace
    ):
        limitations.append("An adaptive step did not complete; earlier findings were preserved.")

    return {
        "mode": mode,
        "tools": [TOOL_DISPLAY_NAMES.get(tool, tool.replace("_", " ").title()) for tool in ordered_tools],
        "finding_count": len(findings) if autonomous and not fallback else len(evidence),
        "initial_plan_steps": initial_plan_steps,
        "adaptive_follow_up": adaptive_follow_up,
        "adaptive_steps_executed": adaptive_steps_executed,
        "limitations": limitations,
    }


def load_dataset(uploaded_file: Any) -> tuple[pd.DataFrame, dict]:
    """Load supported uploads and turn parser failures into clear user messages."""
    filename = getattr(uploaded_file, "name", "").lower()

    if filename.endswith(".csv"):
        reader = pd.read_csv
    elif filename.endswith((".xlsx", ".xls")):
        reader = pd.read_excel
    else:
        raise ValueError(
            "Unsupported file type. Please upload a CSV or Excel (.xlsx/.xls) file."
        )

    if getattr(uploaded_file, "size", None) == 0:
        raise ValueError(
            "The uploaded file is empty. Please upload a file with a header and data rows."
        )

    try:
        df = reader(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "The uploaded file is empty. Please upload a file with a header and data rows."
        ) from exc
    except (pd.errors.ParserError, ValueError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            "The uploaded file could not be read. Check that it is a valid CSV or Excel file."
        ) from exc

    if df.shape[1] == 0:
        raise ValueError("The uploaded dataset has no usable columns.")
    if df.empty:
        raise ValueError("The uploaded dataset has no data rows.")

    return convert_date_columns(df)

def dataset_signature(uploaded_file: Any) -> tuple[str, int | None, str]:
    """Identify replacement uploads even when a user reuses the same filename."""
    content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else b""
    return (
        str(getattr(uploaded_file, "name", "")),
        getattr(uploaded_file, "size", None),
        hashlib.sha256(content).hexdigest(),
    )


def user_error_message(error: Exception, action: str = "analysis") -> str:
    """Avoid exposing provider internals while keeping recovery guidance actionable."""
    detail = str(error).lower()
    if "413" in detail or "request too large" in detail or "tpm limit" in detail:
        return "This request is too large to process. Try a shorter question or clear older conversation messages."
    if any(term in detail for term in ("api", "rate limit", "timeout", "connection", "groq")):
        return "The analysis service is temporarily unavailable. Please try again in a moment."
    if action == "upload":
        return "The file could not be loaded. Check its format and contents, then try again."
    return "The analysis could not be completed. Please try a more specific question."
