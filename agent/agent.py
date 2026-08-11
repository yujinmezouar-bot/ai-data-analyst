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

MAX_HISTORY_MESSAGES = 20


# ============================================================
# TOOL FUNCTIONS
# ============================================================

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
    "You are a professional data analyst assistant working "
    "with a pandas DataFrame. "

    "IMPORTANT: The actual dataset is available to the "
    "Python tools. You must NEVER claim that the actual "
    "data values are unavailable. "

    "Choose the correct tool based on the user's question. "

    "Use dataset_info ONLY when the user asks about the "
    "dataset structure, such as number of rows, number of "
    "columns, column names, or data types. "

    "Use missing_values when the user asks about missing "
    "or null values. "

    "Use statistics when the user asks for descriptive "
    "statistics of numeric columns, such as mean, median, "
    "standard deviation, minimum, maximum, quartiles, "
    "range, or distribution statistics. "

    "Use groupby_analysis when the user asks for a "
    "calculation BY, PER, FOR EACH, or ACROSS categories "
    "or groups. "

    "For groupby_analysis identify: "
    "group_column = the column defining the groups, "
    "value_column = the numeric column being calculated, "
    "agg_function = mean, sum, count, min, max, median, "
    "or std. "

    "IMPORTANT DATE RULES: "
    "The Python application automatically detects common "
    "date columns and converts them to pandas datetime. "

    "A column whose data type is datetime64 should be "
    "treated as a DATE/TIME column, not as a normal string. "

    "When the user asks about dates, time, months, years, "
    "daily sales, weekly sales, monthly sales, sales over "
    "time, trends over time, or chronological order, "
    "identify the appropriate datetime column. "

    "When the user asks for a chart over time, prefer "
    "a line chart and use the date column as x_column. "

    "When the user asks for a chart BY DATE or BY MONTH "
    "and the data contains a datetime column, use that "
    "datetime column rather than treating it as text. "

    "If a datetime column is present, do not describe it "
    "as merely a string column. "

    "Use create_visualization when the user asks for a "
    "chart, graph, plot, or visualization. "

    "For a time trend, normally use chart_type='line'. "

    "For category comparisons, normally use chart_type='bar'. "

    "For a distribution of one numeric variable, use "
    "chart_type='histogram'. "

    "For comparing distributions across categories, use "
    "chart_type='box'. "

    "Always use a tool when the question requires "
    "information from the dataset. "

    "Never guess or invent numbers. "

    "The Python tools perform all calculations. "

    "Your job is to select the appropriate tool and "
    "later explain the tool result clearly. "

    "The conversation may contain previous questions "
    "and answers. Use previous context to understand "
    "follow-up questions. "

    "For example, if the user first asks about Weekly_Sales "
    "and then asks 'show it by store', understand that "
    "the second question refers to Weekly_Sales and Store. "

    "If the user asks 'show me a chart of that', use the "
    "most recent relevant context to identify the columns "
    "and operation. "

    "Do not treat previous answers as tool results. "

    "If a new question requires data, use the appropriate "
    "tool again."
)


# ============================================================
# AGENT
# ============================================================

class Agent:
    """
    Orchestrates the LLM <-> tool-calling process.
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
        conversation_history: (
            list[dict[str, str]] | None
        ) = None,
    ) -> dict[str, Any]:

        history = conversation_history or []

        if len(history) > MAX_HISTORY_MESSAGES:

            history = history[
                -MAX_HISTORY_MESSAGES:
            ]


        # ----------------------------------------------------
        # First LLM call
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


        response_message = self.llm.chat(
            messages,
            tools=TOOL_SCHEMAS,
        )


        # ----------------------------------------------------
        # No tool requested
        # ----------------------------------------------------

        if not getattr(
            response_message,
            "tool_calls",
            None,
        ):

            return {
                "answer": response_message.content,
                "figure": None,
            }


        # ----------------------------------------------------
        # Execute tools
        # ----------------------------------------------------

        tool_results = []

        figure = None


        for tool_call in response_message.tool_calls:

            tool_name = (
                tool_call.function.name
            )


            try:

                tool_args = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                tool_args = {}


            tool_result = self._execute_tool(
                tool_name,
                tool_args,
                df,
            )


            # Extract Plotly figure

            if (
                isinstance(tool_result, dict)
                and "figure" in tool_result
            ):

                figure = tool_result["figure"]

                tool_result = {
                    key: value
                    for key, value
                    in tool_result.items()
                    if key != "figure"
                }


            tool_results.append(
                {
                    "tool_name": tool_name,
                    "result": tool_result,
                }
            )


        # ----------------------------------------------------
        # Prepare result text
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
        # Final explanation call
        # ----------------------------------------------------

        final_messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional data analyst "
                    "assistant. "

                    "The Python tools have already performed "
                    "the required calculations. "

                    "Your job now is ONLY to explain the "
                    "provided tool result to the user. "

                    "Do NOT call tools. "

                    "Do NOT perform another calculation. "

                    "Do NOT invent numbers. "

                    "Use the exact values contained in the "
                    "tool result. "

                    "If the result contains an error, explain "
                    "the error clearly instead of inventing "
                    "an answer. "

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


        final_message = self.llm.chat(
            final_messages,
            tool_choice="none",
        )


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

        tool_function = TOOL_FUNCTIONS.get(
            tool_name
        )


        if tool_function is None:

            return {
                "error": (
                    f"Unknown tool requested: "
                    f"'{tool_name}'"
                )
            }


        try:

            return tool_function(
                df,
                **tool_args,
            )

        except Exception as e:

            return {
                "error": (
                    f"Tool '{tool_name}' failed: "
                    f"{e}"
                )
            }
