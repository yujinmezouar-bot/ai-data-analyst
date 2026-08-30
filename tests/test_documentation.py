import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_setup_architecture_evaluation_and_privacy():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()

    assert len(readme.strip()) > 2_000
    for heading in (
        "## Architecture",
        "## Installation",
        "## Running the application",
        "## Testing",
        "## Behavioral evaluation",
        "## Privacy and data flow",
        "## Known limitations",
    ):
        assert heading in readme
    assert "streamlit run app.py" in readme
    assert "GROQ_API_KEY" in readme
    assert "not fully offline" in lowered
    assert "not currently hardened" in lowered


def test_streamlit_has_visible_external_provider_privacy_disclosure():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    info_texts = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "info"
        ):
            continue
        if node.args:
            try:
                info_texts.append(str(ast.literal_eval(node.args[0])))
            except (ValueError, TypeError):
                pass

    disclosure = " ".join(info_texts).lower()
    assert "locally" in disclosure
    assert "groq" in disclosure
    assert "external-provider" in disclosure
    assert "confidential or sensitive" in disclosure
