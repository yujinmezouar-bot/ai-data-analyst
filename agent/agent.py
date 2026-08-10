import json
from typing import Any

import pandas as pd

from agent.llm import LLMClient

from tools.dataset_info import (
    dataset_info,
    DATASET_INFO_SCHEMA,
)

from tools.missing_values import (
    missing_values,
    MISSING_VALUES_SCHEMA,
)

from tools.statistics import (
    statistics,
    STATISTICS_SCHEMA,
)

from tools.groupby import (
    groupby_analysis,
    GROUPBY_ANALYSIS_SCHEMA,
)

from tools.visualization import (
    create_visualization,
    CREATE_VISUALIZATION_SCHEMA,
)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

# Maximum number of messages kept in conversation history.
#
# 20 messages = approximately 10 user/assistant turns.
#
# We keep this limited so the conversation does not grow
# indefinitely and consume too many LLM tokens.
MAX_HISTORY_MESSAGES = 20


# ============================================================
# TOOL FUNCTIONS
# ============================================================

# The LLM can only request tools that exist in this dictionary.

TOOL_FUNCTIONS = {
    "dataset_info": dataset_info,
    "missing_values": missing_values,
    "statistics": statistics,
    "groupby_analysis": groupby_analysis,
    "create_visualization": create_visualization,
}


# ============================================================
# TOOL SCHEMAS
# ============================================================

# These schemas are sent to the LLM during the first call.

TOOL_SCHEMAS = [
    DATASET_INFO_SCHEMA,
    MISSING_VALUES_SCHEMA,
    STATISTICS_SCHEMA,
    GROUPBY_ANALYSIS_SCHEMA,
    CREATE_VISUALIZATION_SCHEMA,
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are a data analyst assistant working with a pandas DataFrame. "

    "IMPORTANT: The actual dataset is available to the Python tools. "
    "You must NEVER claim that the actual data values are unavailable. "

    "Choose the correct tool based on the user's question. "

    "Use dataset_info ONLY when the user asks about the dataset "
    "structure, such as the number of rows, number of columns, "
    "column names, or data types. "

    "Use missing_values when the user asks about missing or null values. "

    "Use statistics when the user asks for descriptive statistics "
    "of numeric columns, such as mean, median, standard deviation, "
    "minimum, maximum, quartiles, range, or distribution statistics. "

    "Use groupby_analysis when the user asks for a calculation "
    "grouped by another column, such as 'average salary by department', "
    "'total sales by store', 'average sales per store', "
    "or 'number of orders by customer'. "

    "For groupby_analysis, identify: "
    "group_column = the column used to define the groups, "
    "value_column = the numeric column being calculated, "
    "agg_function = the requested aggregation such as mean, sum, "
    "count, min, max, median, or std. "

    "Use create_visualization when the user asks for a chart, "
    "graph, plot, or visualization. "

    "Always use a tool when the question requires information "
    "from the dataset. "

    "Never guess or invent numbers. "

    "The Python tools perform all calculations. "
    "Your job is only to select the appropriate tool and later "
    "explain the tool result clearly. "

    "The conversation may contain previous questions and answers. "
    "Use this previous context to understand follow-up questions. "

    "For example, if the user first asks about Weekly_Sales "
    "and then asks 'what about by store?', understand that "
    "the second question refers to Weekly_Sales and Store. "

    "Do not treat previous answers as tool results. "
    "If a new question requires data, use the appropriate tool."
)


# ============================================================
# AGENT
# ============================================================

class Agent:
    """
    Orchestrates the LLM <-> tool-calling process.

    The Agent receives:

        question
        dataframe
        optional conversation history

    The conversation history contains ONLY normal text messages:

        {
            "role": "user",
            "content": "..."
        }

        {
            "role": "assistant",
            "content": "..."
        }

    IMPORTANT:
    Conversation history must NEVER contain:

        - tool_calls
        - role="tool"
        - tool_call_id

    This keeps the conversation memory simple and avoids
    problems with Groq tool-calling across different turns.
    """

    def __init__(self) -> None:
        self.llm = LLMClient()

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(
        self,
        question: str,
        df: pd.DataFrame,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Run one user question through the agent.

        Parameters
        ----------
        question:
            The current user's question.

        df:
            The currently loaded pandas DataFrame.

        conversation_history:
            Previous user/assistant messages.

        Returns
        -------
        dict
            {
                "answer": str,
                "figure": Plotly Figure or None
            }
        """

        # ----------------------------------------------------
        # STEP 1
        # Prepare conversation history
        # ----------------------------------------------------

        history = conversation_history or []

        # Defensive limit.
        # app.py also limits this, but we protect the Agent too.
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]

        # ----------------------------------------------------
        # Build messages for the first LLM call.
        #
        # Important:
        #
        # system prompt
        #       +
        # previous text conversation
        #       +
        # current question
        #
        # No previous tool calls are included.
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        figure = None

        # ----------------------------------------------------
        # STEP 2
        # Ask the LLM which tool to use.
        # ----------------------------------------------------

        response_message = self.llm.chat(
            messages,
            tools=TOOL_SCHEMAS,
        )

        # ----------------------------------------------------
        # If the LLM doesn't request a tool,
        # return its answer directly.
        # ----------------------------------------------------

        if not getattr(response_message, "tool_calls", None):
            return {
                "answer": response_message.content,
                "figure": None,
            }

        # ----------------------------------------------------
        # STEP 3
        # Execute requested tools.
        # ----------------------------------------------------

        tool_results = []

        for tool_call in response_message.tool_calls:

            tool_name = tool_call.function.name

            # -----------------------------------------------
            # Parse tool arguments
            # -----------------------------------------------

            try:
                tool_args = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                tool_args = {}

            # -----------------------------------------------
            # Execute registered tool
            # -----------------------------------------------

            tool_result = self._execute_tool(
                tool_name,
                tool_args,
                df,
            )

            # -----------------------------------------------
            # Extract Plotly figure
            # -----------------------------------------------

            if (
                isinstance(tool_result, dict)
                and "figure" in tool_result
            ):

                figure = tool_result["figure"]

                tool_result = {
                    key: value
                    for key, value in tool_result.items()
                    if key != "figure"
                }

            # -----------------------------------------------
            # Store tool result
            # -----------------------------------------------

            tool_results.append(
                {
                    "tool_name": tool_name,
                    "result": tool_result,
                }
            )

        # ----------------------------------------------------
        # STEP 4
        # Convert tool results to text for final LLM call.
        # ----------------------------------------------------

        tool_results_text = "\n\n".join(
            (
                f"Tool: {item['tool_name']}\n"
                f"Result: "
                f"{json.dumps(item['result'], default=str)}"
            )
            for item in tool_results
        )

        # ----------------------------------------------------
        # STEP 5
        # Final LLM call.
        #
        # IMPORTANT:
        #
        # We deliberately create a completely new conversation.
        #
        # We DO NOT include:
        #
        # - previous conversation history
        # - tool_calls
        # - role="tool"
        # - tool schemas
        #
        # The LLM only explains the Python result.
        # ----------------------------------------------------

        final_messages = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst assistant. "

                    "The Python tools have already performed "
                    "the required calculations. "

                    "Your job now is ONLY to explain the "
                    "provided tool result to the user. "

                    "Do NOT call any tools. "

                    "Do NOT perform another calculation. "

                    "Do NOT invent any numbers. "

                    "Use the exact values contained in the "
                    "tool result. "

                    "Answer clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n"
                    f"{question}\n\n"

                    f"Results calculated by Python:\n"
                    f"{tool_results_text}\n\n"

                    "Now give the final answer to the user."
                ),
            },
        ]

        # ----------------------------------------------------
        # STEP 6
        # Ask LLM to explain the tool result.
        # ----------------------------------------------------

        final_message = self.llm.chat(
            final_messages,
            tool_choice="none",
        )

        # ----------------------------------------------------
        # STEP 7
        # Return final answer and optional visualization.
        # ----------------------------------------------------

        return {
            "answer": final_message.content,
            "figure": figure,
        }

    # ========================================================
    # TOOL EXECUTOR
    # ========================================================

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        df: pd.DataFrame,
    ) -> Any:
        """
        Execute only tools explicitly registered in
        TOOL_FUNCTIONS.

        The LLM cannot execute arbitrary Python code.
        """

        # ----------------------------------------------------
        # Find requested tool
        # ----------------------------------------------------

        tool_function = TOOL_FUNCTIONS.get(tool_name)

        # ----------------------------------------------------
        # Unknown tool
        # ----------------------------------------------------

        if tool_function is None:

            return {
                "error": (
                    f"Unknown tool requested: "
                    f"'{tool_name}'"
                )
            }

        # ----------------------------------------------------
        # Execute tool safely
        # ----------------------------------------------------

        try:

            return tool_function(
                df,
                **tool_args,
            )

        except Exception as e:

            return {
                "error": (
                    f"Tool '{tool_name}' failed: {e}"
                )
            }
