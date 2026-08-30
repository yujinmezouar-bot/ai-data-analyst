import ast
from pathlib import Path


def test_streamlit_agent_call_uses_auto_mode_and_preserves_dataset_arguments():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "agent"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["autonomous"], ast.Constant)
    assert keywords["autonomous"].value is None
    assert isinstance(keywords["df"], ast.Constant) and keywords["df"].value is None
    assert ast.unparse(keywords["datasets"]) == "st.session_state.datasets"
    assert ast.unparse(keywords["conversation_history"]) == "st.session_state.messages"
