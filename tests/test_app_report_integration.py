import ast
from pathlib import Path


def _tree():
    return ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))


def test_streamlit_persists_complete_result_and_builds_report_without_agent_rerun():
    tree = _tree()
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    stored_result = any(
        isinstance(node.value, ast.Name) and node.value.id == "result"
        and any(isinstance(target, ast.Attribute) and target.attr == "last_analysis_result" for target in node.targets)
        for node in assignments
    )
    build_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "build_analysis_report"
    ]
    agent_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run" and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "agent"
    ]

    assert stored_result
    assert len(build_calls) == 1
    assert ast.unparse(build_calls[0].args[1]) == "st.session_state.last_analysis_result"
    assert ast.unparse(build_calls[0].args[2]) == "st.session_state.datasets"
    assert len(agent_calls) == 1


def test_report_state_has_reset_paths_and_markdown_download():
    tree = _tree()
    reset_assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None
        and any(isinstance(target, ast.Attribute) and target.attr == "last_report_markdown" for target in node.targets)
    ]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert len(reset_assignments) >= 4
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "markdown" for call in calls)
    assert any(isinstance(call.func, ast.Attribute) and call.func.attr == "download_button" for call in calls)


def test_word_report_uses_completed_report_state_without_rerunning_agent():
    tree = _tree()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    render_calls = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "render_docx"
    ]
    downloads = [
        call for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "download_button"
    ]

    assert len(render_calls) == 1
    assert ast.unparse(render_calls[0].args[0]) == "report"
    assert any(
        any(
            keyword.arg == "mime"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            for keyword in call.keywords
        )
        for call in downloads
    )
