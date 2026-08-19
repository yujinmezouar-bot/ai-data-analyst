import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from agent.agent import Agent, MAX_LLM_REQUEST_CHARS, TOOL_SCHEMAS, _estimate_request_chars
from tools.dataset_info import dataset_info
from tools.groupby import groupby_analysis
from tools.time_analysis import time_analysis


def _tool_message(name: str, arguments: dict):
    return SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))],
    )


def _text_message(content: str):
    return SimpleNamespace(content=content, tool_calls=None)


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([
            "2024-01-15", "2024-01-15", "2024-01-15",
            "2024-02-15", "2024-02-15", "2024-02-15",
            "2024-03-15", "2024-03-15", "2024-03-15",
        ]),
        "Store": [20, 4, 14, 20, 4, 14, 20, 4, 14],
        "Product": ["Laptop", "Phone", "Tablet"] * 3,
        "Sales": [500, 300, 200, 450, 280, 180, 400, 260, 150],
        "Quantity": [5, 6, 4, 4, 6, 3, 4, 5, 3],
        "Price": [100, 50, 50, 112.5, 46.67, 60, 100, 52, 50],
        "Customer_ID": [f"C{i:03d}" for i in range(9)],
        "Region": ["North", "South", "North"] * 3,
        "Constant_Flag": [1] * 9,
        "Discount": [0.0, None, 0.1, 0.0, None, 0.15, 0.05, 0.0, None],
    })


def test_realistic_sales_tools_cover_ranking_trend_comparison_and_contributors():
    """Realistic sales data produces deterministic inputs for common analyst questions."""
    df = _sales_df()

    stores = groupby_analysis(df, "Store", "Sales", agg_function="sum", top_n=3)
    products = groupby_analysis(df, "Product", "Sales", agg_function="sum", top_n=2)
    comparison = groupby_analysis(df, "Store", "Sales", agg_function="sum", filter_values=[20, 4])
    trend = time_analysis(df, "Date", "Sales", period="month", agg_function="sum")

    assert stores["ranking"] == ["20", "4", "14"]
    assert products["best_group"] == "Laptop"
    assert comparison["comparison"]["absolute_difference"] == 510.0
    assert trend["trend_direction"] == "strictly_decreasing"
    # These are observable sales contributors, not an assertion of causation.
    assert products["result"]["Laptop"] == 1350.0


def test_realistic_profile_detects_dates_missing_constants_and_identifiers():
    df = _sales_df()
    profile = dataset_info(df)

    assert "Date" in profile["date_columns"]
    assert profile["missing_summary"]["Discount"]["missing_count"] == 3
    assert "Constant_Flag" in profile["constant_columns"]
    assert any(item["column"] == "Customer_ID" for item in profile["potential_identifiers"])


def test_realistic_top_store_to_filtered_trend_workflow_uses_numeric_filters():
    """Mocked LLM orchestration passes exact ranked numeric IDs to time analysis."""
    df = _sales_df()
    with patch("agent.agent.LLMClient") as MockLLM:
        llm = MockLLM.return_value
        llm.chat.side_effect = [
            _tool_message("groupby_analysis", {
                "group_column": "Store", "value_column": "Sales", "agg_function": "sum", "top_n": 3,
            }),
            _tool_message("time_analysis", {
                "date_column": "Date", "value_column": "Sales", "period": "month", "agg_function": "sum",
                "group_column": "Store", "filter_values": [20, 4, 14],
            }),
            _text_message("Enough information."),
            _text_message("Among stores 20, 4, and 14, sales declined over the available months."),
        ]

        result = Agent().run("Show the monthly trend for the top 3 stores", df)

    calls = [entry for entry in result["trace"] if entry["step"] == "tool_call"]
    assert [entry["tool"] for entry in calls] == ["groupby_analysis", "time_analysis"]
    assert calls[1]["arguments"]["filter_values"] == [20, 4, 14]
    assert "Among stores" in result["answer"]


def test_realistic_trend_to_visualization_keeps_figure_out_of_llm_payload():
    df = _sales_df()
    with patch("agent.agent.LLMClient") as MockLLM:
        llm = MockLLM.return_value
        llm.chat.side_effect = [
            _tool_message("time_analysis", {
                "date_column": "Date", "value_column": "Sales", "period": "month", "agg_function": "sum",
            }),
            _tool_message("create_visualization", {
                "chart_type": "line", "x_column": "Date", "y_column": "Sales",
                "agg_function": "sum", "period": "month",
            }),
            _text_message("Enough information."),
            _text_message("Monthly sales declined, as shown in the chart."),
        ]

        result = Agent().run("Show the monthly sales trend", df)

    final_messages = llm.chat.call_args_list[-1].args[0]
    assert result["figure"] is not None
    assert '"figure"' not in final_messages[1]["content"]
    assert _estimate_request_chars(final_messages) <= MAX_LLM_REQUEST_CHARS


def test_realistic_follow_up_preserves_recent_context_within_budget():
    df = _sales_df()
    history = [
        {"role": "user", "content": "What are the top stores by sales?"},
        {"role": "assistant", "content": "The top stores are 20, 4, and 14."},
    ] + [
        {"role": "user" if i % 2 else "assistant", "content": f"old context {i}: " + "x" * 1800}
        for i in range(18)
    ]
    question = "Now show me the monthly trend for the top 3. Which one performed worst?"
    with patch("agent.agent.LLMClient") as MockLLM:
        llm = MockLLM.return_value
        llm.chat.return_value = _text_message("I will use stores 20, 4, and 14.")
        Agent().run(question, df, conversation_history=history)

    messages = llm.chat.call_args.args[0]
    assert _estimate_request_chars(messages, TOOL_SCHEMAS) <= MAX_LLM_REQUEST_CHARS
    assert any(message["content"] == question for message in messages)
    assert any("old context 17" in message["content"] for message in messages)
    assert not any("old context 0" in message["content"] for message in messages)
