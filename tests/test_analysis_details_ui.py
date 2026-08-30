import ast
import json
from pathlib import Path

from ui_utils import build_analysis_details


ROOT = Path(__file__).resolve().parents[1]


def test_reactive_details_use_actual_trace_tools_and_deduplicate_them():
    result = {
        "trace": [
            {"step": "question", "question": "Run autonomous correlation analysis"},
            {"step": "routing", "decision": "reactive"},
            {"step": "tool_call", "tool": "statistics", "success": True},
            {"step": "tool_call", "tool": "statistics", "success": True, "reused": True},
        ],
        "evidence": [{"tool_name": "statistics", "result": {"mean": 4}}],
    }

    details = build_analysis_details(result)

    assert details["mode"] == "Reactive analysis"
    assert details["tools"] == ["Descriptive Statistics"]
    assert details["finding_count"] == 1


def test_autonomous_details_report_plan_findings_and_adaptive_execution():
    result = {
        "trace": [
            {"step": "autonomous_plan", "plan": {"steps": [{}, {}]}},
            {"step": "adaptive_review", "status": "follow_up"},
            {"step": "adaptive_execution", "status": "completed", "executed_steps": 2},
            {"step": "final_answer", "autonomous": True},
        ],
        "findings": [
            {"tool_name": "kpi_contribution_analysis"},
            {"tool_name": "time_analysis"},
            {"tool_name": "kpi_contribution_analysis"},
        ],
    }

    details = build_analysis_details(result)

    assert details["mode"] == "Autonomous analysis"
    assert details["tools"] == ["KPI Contribution Analysis", "Time Analysis"]
    assert details["finding_count"] == 3
    assert details["initial_plan_steps"] == 2
    assert details["adaptive_follow_up"] is True
    assert details["adaptive_steps_executed"] == 2
    assert any("bounded" in item for item in details["limitations"])


def test_fallback_is_distinguishable_and_diagnostics_are_not_exposed():
    result = {
        "trace": [
            {
                "step": "autonomous_fallback",
                "stage": "planner_call",
                "message": "private provider payload",
                "raw_prompt": "hidden reasoning",
            },
            {"step": "tool_call", "tool": "groupby_analysis", "success": True},
        ],
        "evidence": [{"tool_name": "groupby_analysis", "result": {}}],
    }

    details = build_analysis_details(result)
    serialized = json.dumps(details)

    assert details["mode"] == "Autonomous → reactive fallback"
    assert details["tools"] == ["Groupby Analysis"]
    assert "private provider payload" not in serialized
    assert "hidden reasoning" not in serialized
    assert any("stopped safely" in item for item in details["limitations"])


def test_failed_tool_and_noncausal_or_ml_safety_are_derived_from_evidence():
    details = build_analysis_details({
        "trace": [
            {"step": "tool_call", "tool": "correlation_analysis", "success": False},
            {"step": "tool_call", "tool": "train_ml_model", "success": True},
        ],
        "evidence": [
            {"tool_name": "correlation_analysis", "result": {"error": "failed"}},
            {"tool_name": "train_ml_model", "result": {}},
        ],
    })

    joined = " ".join(details["limitations"]).lower()
    assert "not causation" in joined
    assert "held-out" in joined
    assert "tool attempt failed" in joined


def test_streamlit_renders_collapsed_details_and_multi_dataset_selector_safely():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    expanders = [
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "expander"
    ]
    selectors = [
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "selectbox"
    ]
    agent_calls = [
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "run"
        and isinstance(call.func.value, ast.Name) and call.func.value.id == "agent"
    ]

    assert any(
        call.args and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "Analysis details"
        and any(keyword.arg == "expanded" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False for keyword in call.keywords)
        for call in expanders
    )
    assert selectors
    assert len(agent_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in agent_calls[0].keywords}
    assert ast.unparse(keywords["datasets"]) == "st.session_state.datasets"
    assert isinstance(keywords["df"], ast.Constant) and keywords["df"].value is None
