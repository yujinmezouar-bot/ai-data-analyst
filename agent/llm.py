import os
from typing import Any, Protocol

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


# ============================================================
# MODEL
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LLM Provider abstraction
# ============================================================

class LLMProvider(Protocol):
    """Protocol for pluggable LLM providers used by the Agent.

    The provider must implement a chat(...) method with the same
    signature and return value expectations as the current LLMClient.
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        ...


# ============================================================
# Groq-backed provider
# ============================================================

class GroqProvider:
    """Concrete provider that uses the Groq API.

    This encapsulates all Groq-specific details so higher-level code
    can depend on the LLMProvider Protocol instead.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure it is set in your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto" if tool_choice is None else tool_choice

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {e}") from e

        return response.choices[0].message


# ============================================================
# Backwards-compatible LLMClient wrapper
# ============================================================

class LLMClient:
    """Compatibility wrapper that preserves the original LLMClient API.

    It delegates to an underlying provider (currently GroqProvider) so
    existing imports and tests that reference agent.llm.LLMClient continue
    to work unchanged.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # Internally use the GroqProvider by default
        self._provider: LLMProvider = GroqProvider(model=model)
        # Backwards-compatible public attributes expected by callers/tests
        # - model: the selected model name
        # - client: the underlying provider client (if available)
        self.model = model
        try:
            self.client = self._provider.client
        except Exception:
            self.client = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        return self._provider.chat(messages, tools=tools, tool_choice=tool_choice)
        """
        Send messages to the Groq LLM.

        Parameters
        ----------
        messages:
            OpenAI-style messages.

        tools:
            Optional list of tools available to the model.

        tool_choice:
            Controls tool calling.

            Examples:
                "auto"  -> model may call tools
                "none"  -> model must NOT call tools

        """

        # ----------------------------------------------------
        # Base request
        # ----------------------------------------------------

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # ----------------------------------------------------
        # TOOL CALLING
        # ----------------------------------------------------

        if tools is not None:

            kwargs["tools"] = tools

            # If the caller didn't specify a choice,
            # allow the model to decide automatically.
            if tool_choice is None:
                kwargs["tool_choice"] = "auto"

            else:
                kwargs["tool_choice"] = tool_choice

        # ----------------------------------------------------
        # API CALL
        # ----------------------------------------------------

        try:

            response = self.client.chat.completions.create(
                **kwargs
            )

        except Exception as e:

            raise RuntimeError(
                f"Groq API call failed: {e}"
            ) from e

        # ----------------------------------------------------
        # Return only the message
        # ----------------------------------------------------

        return response.choices[0].message