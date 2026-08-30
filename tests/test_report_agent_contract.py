from types import SimpleNamespace
from unittest.mock import patch
import json

from agent.agent import Agent


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_message(name, arguments):
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return _message(tool_calls=[call])


def test_reactive_result_adds_bounded_evidence_without_changing_existing_fields(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.side_effect = [
            _tool_message("statistics", {"column": "Weekly_Sales"}),
            _message("Done."),
            _message("Average sales are 225."),
        ]
        result = Agent().run("Average sales?", sample_df, autonomous=False)

    assert result["answer"] == "Average sales are 225."
    assert result["figure"] is None
    assert isinstance(result["trace"], list)
    assert result["evidence"][0]["tool_name"] == "statistics"
    assert result["evidence"][0]["result"]["mean"] == 225.0


def test_direct_reactive_result_returns_empty_evidence(sample_df):
    with patch("agent.agent.LLMClient") as llm_class:
        llm_class.return_value.chat.return_value = _message("Hello.")
        result = Agent().run("Hello", sample_df, autonomous=False)

    assert result["answer"] == "Hello."
    assert result["evidence"] == []
