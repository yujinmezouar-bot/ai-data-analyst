import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.llm import DEFAULT_MODEL, LLMClient


def test_llm_client_missing_api_key():
    """Test LLMClient raises ValueError if GROQ_API_KEY is not set."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
            LLMClient()


def test_llm_client_successful_init():
    """Test LLMClient initializes properly when GROQ_API_KEY is provided."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "dummy_test_key"}):
        with patch("agent.llm.Groq") as MockGroq:
            client = LLMClient()
            assert client.model == DEFAULT_MODEL
            MockGroq.assert_called_once_with(api_key="dummy_test_key")


def test_llm_client_chat_success():
    """Test LLMClient.chat returns the first message choice."""
    mock_message = SimpleNamespace(content="Hello world", tool_calls=None)
    mock_choice = SimpleNamespace(message=mock_message)
    mock_response = SimpleNamespace(choices=[mock_choice])

    with patch.dict(os.environ, {"GROQ_API_KEY": "dummy_test_key"}):
        with patch("agent.llm.Groq") as MockGroq:
            mock_groq_instance = MockGroq.return_value
            mock_groq_instance.chat.completions.create.return_value = mock_response

            client = LLMClient()
            messages = [{"role": "user", "content": "Hi"}]
            result = client.chat(messages)

            assert result.content == "Hello world"
            mock_groq_instance.chat.completions.create.assert_called_once_with(
                model=DEFAULT_MODEL,
                messages=messages,
            )


def test_llm_client_chat_with_tools_and_tool_choice():
    """Test LLMClient.chat correctly configures tools and tool_choice."""
    mock_message = SimpleNamespace(content=None, tool_calls=[])
    mock_response = SimpleNamespace(choices=[SimpleNamespace(message=mock_message)])

    with patch.dict(os.environ, {"GROQ_API_KEY": "dummy_test_key"}):
        with patch("agent.llm.Groq") as MockGroq:
            mock_groq_instance = MockGroq.return_value
            mock_groq_instance.chat.completions.create.return_value = mock_response

            client = LLMClient()
            tools = [{"type": "function", "function": {"name": "test_tool"}}]

            # Case 1: default tool_choice with tools -> "auto"
            client.chat([{"role": "user", "content": "Run tool"}], tools=tools)
            call_kwargs = mock_groq_instance.chat.completions.create.call_args[1]
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

            # Case 2: explicit tool_choice="none"
            client.chat([{"role": "user", "content": "Explain"}], tools=tools, tool_choice="none")
            call_kwargs2 = mock_groq_instance.chat.completions.create.call_args[1]
            assert call_kwargs2["tool_choice"] == "none"


def test_llm_client_chat_api_failure():
    """Test LLMClient.chat raises RuntimeError on Groq API error."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "dummy_test_key"}):
        with patch("agent.llm.Groq") as MockGroq:
            mock_groq_instance = MockGroq.return_value
            mock_groq_instance.chat.completions.create.side_effect = Exception("Rate limit exceeded")

            client = LLMClient()
            with pytest.raises(RuntimeError, match="Groq API call failed: Rate limit exceeded"):
                client.chat([{"role": "user", "content": "Hi"}])
