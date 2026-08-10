import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


# ============================================================
# MODEL
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LLM CLIENT
# ============================================================

class LLMClient:
    """
    Thin abstraction around the Groq API.

    The rest of the application should only communicate
    with this class, not directly with the Groq package.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ) -> None:

        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Make sure it is set in your .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
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