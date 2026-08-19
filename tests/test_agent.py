import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from agent.agent import (
    FINAL_EXPLANATION_SYSTEM_PROMPT,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_ITERATIONS,
    SYSTEM_PROMPT,
    Agent,
    _compact_tool_result,
    _summarise_tool_results,
)


def _make_tool_msg(tool_name: str, args: dict | str):
    """Helper to mock a message containing a tool call."""
    raw_args = args if isinstance(args, str) else json.dumps(args)
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name=tool_name,
            arguments=raw_args,
        )
    )
    return SimpleNamespace(content=None, tool_calls=[tool_call])


def _make_text_msg(content: str):
    """Helper to mock a plain text message without tool calls."""
    return SimpleNamespace(content=content, tool_calls=None)


def test_agent_direct_answer_no_tool(sample_df: pd.DataFrame):
    """Test Agent when LLM answers directly without calling any tools."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.return_value = _make_text_msg("Hello! How can I help you?")

        agent = Agent()
        res = agent.run("Hello", sample_df)

        assert res["answer"] == "Hello! How can I help you?"
        assert res["figure"] is None
        assert len(res["trace"]) == 2
        assert res["trace"][0]["step"] == "question"
        assert res["trace"][1]["step"] == "final_answer"
        assert res["trace"][1]["tool_used"] is False


def test_agent_single_tool_execution(sample_df: pd.DataFrame):
    """Test Agent executing one tool call followed by final explanation."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("statistics", {"column": "Weekly_Sales"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("The average weekly sales is 225.0."),
        ]

        agent = Agent()
        res = agent.run("What is the average sales?", sample_df)

        assert res["answer"] == "The average weekly sales is 225.0."
        assert res["figure"] is None
        assert any(t.get("step") == "tool_call" and t.get("tool") == "statistics" for t in res["trace"])
        assert any(t.get("step") == "final_answer" and t.get("tool_used") is True for t in res["trace"])


def test_agent_multi_step_tool_execution(sample_df: pd.DataFrame):
    """Test Agent performing multi-step tool calls (iteration 0 -> tool A, iteration 1 -> tool B)."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("groupby_analysis", {"group_column": "Store", "value_column": "Weekly_Sales"}),
            _make_tool_msg("statistics", {"column": "Weekly_Sales"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("Here is the store breakdown and overall stats."),
        ]

        agent = Agent()
        res = agent.run("Break down sales by store and give overall stats", sample_df)

        assert res["answer"] == "Here is the store breakdown and overall stats."
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 2
        assert tool_steps[0]["tool"] == "groupby_analysis"
        assert tool_steps[1]["tool"] == "statistics"


def test_agent_malformed_json_arguments(sample_df: pd.DataFrame):
    """Test Agent recovery when LLM returns invalid JSON in tool arguments."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("groupby_analysis", "{invalid json: true"),
            _make_text_msg("Done with tools."),
            _make_text_msg("I encountered a JSON format issue and could not proceed."),
        ]

        agent = Agent()
        res = agent.run("Group by store", sample_df)

        assert "JSON" in res["answer"] or "could not proceed" in res["answer"]
        tool_trace = next(t for t in res["trace"] if t.get("step") == "tool_call")
        assert tool_trace["success"] is False
        assert "not valid JSON" in tool_trace["error"]


def test_agent_missing_required_arguments(sample_df: pd.DataFrame):
    """Test Agent handling TypeError with targeted required parameters hint."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        # groupby_analysis requires both group_column and value_column
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("groupby_analysis", {"group_column": "Store"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("Missing required parameter value_column."),
        ]

        agent = Agent()
        res = agent.run("Group by store", sample_df)

        tool_trace = next(t for t in res["trace"] if t.get("step") == "tool_call")
        assert tool_trace["success"] is False
        assert "invalid or missing arguments" in tool_trace["error"]


def test_agent_unknown_tool(sample_df: pd.DataFrame):
    """Test Agent handling unknown tool request."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("non_existent_tool", {}),
            _make_text_msg("Done with tools."),
            _make_text_msg("That tool is not available."),
        ]

        agent = Agent()
        res = agent.run("Run mystery tool", sample_df)

        tool_trace = next(t for t in res["trace"] if t.get("step") == "tool_call")
        assert tool_trace["success"] is False
        assert "Unknown tool requested" in tool_trace["error"]


def test_agent_max_tool_iterations_limit(sample_df: pd.DataFrame):
    """Test Agent stops looping and forces final explanation after MAX_TOOL_ITERATIONS."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        # Return tool calls indefinitely (4 iterations)
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("statistics", {"column": "Weekly_Sales"})
            for _ in range(MAX_TOOL_ITERATIONS)
        ] + [_make_text_msg("Final summarized answer after max iterations.")]

        agent = Agent()
        res = agent.run("Endless loop test", sample_df)

        assert res["answer"] == "Final summarized answer after max iterations."
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == MAX_TOOL_ITERATIONS


def test_agent_figure_extraction_and_persistence(sample_df: pd.DataFrame):
    """Test that visualization figure is extracted and returned in final response."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("create_visualization", {
                "chart_type": "bar",
                "x_column": "Store",
                "y_column": "Weekly_Sales",
                "agg_function": "mean",
            }),
            _make_text_msg("Done with tools."),
            _make_text_msg("Here is the bar chart of average sales by store."),
        ]

        agent = Agent()
        res = agent.run("Show me a chart of sales by store", sample_df)

        assert res["figure"] is not None
        assert isinstance(res["figure"], go.Figure)
        assert res["answer"] == "Here is the bar chart of average sales by store."


def test_agent_history_truncation(sample_df: pd.DataFrame):
    """Test that conversation history exceeding MAX_HISTORY_MESSAGES is truncated."""
    long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(30)]

    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.return_value = _make_text_msg("Done.")

        agent = Agent()
        res = agent.run("New question", sample_df, conversation_history=long_history)

        # Inspect the messages passed to LLMClient.chat
        called_messages = mock_llm_instance.chat.call_args[0][0]
        # system (1) + truncated history (20) + user question (1) = 22
        assert len(called_messages) == 1 + MAX_HISTORY_MESSAGES + 1


def test_compact_tool_result():
    """Test _compact_tool_result safely truncates oversized payloads."""
    # 1. Normal small payload is unchanged
    small = {"status": "ok", "value": 42}
    assert _compact_tool_result(small) == small

    # 2. Large dict exceeding 4000 characters with "result" key truncated to 25 items
    large_dict = {
        "result": {
            f"a_very_long_descriptive_key_name_number_{i}": i * 100
            for i in range(200)
        }
    }
    assert len(json.dumps(large_dict)) > 4000
    compact_dict = _compact_tool_result(large_dict)
    assert len(compact_dict["result"]) == 25
    assert "note" in compact_dict

    # 3. Large dict exceeding 4000 characters with "top_correlations" list truncated to 25 items
    large_list = {
        "top_correlations": [
            {"column_1": f"long_column_name_alpha_{i}", "column_2": f"long_column_name_beta_{i}", "correlation": 0.5}
            for i in range(200)
        ]
    }
    assert len(json.dumps(large_list)) > 4000
    compact_list = _compact_tool_result(large_list)
    assert len(compact_list["top_correlations"]) == 25
    assert "note" in compact_list


def test_agent_repeated_tool_call_cached(sample_df: pd.DataFrame):
    """Test that duplicate tool calls with identical arguments in the same run reuse cached results."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        # LLM calls statistics(column='Weekly_Sales') twice consecutively
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("statistics", {"column": "Weekly_Sales"}),
            _make_tool_msg("statistics", {"column": "Weekly_Sales"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("The mean sales is 225.0."),
        ]

        agent = Agent()
        res = agent.run("What is average sales?", sample_df)

        assert res["answer"] == "The mean sales is 225.0."
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 2
        assert tool_steps[0]["success"] is True
        assert tool_steps[1]["success"] is True


def test_agent_tool_failure_and_recovery(sample_df: pd.DataFrame):
    """Test Agent recovering from an initial tool failure by calling tool with corrected args."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        # Iteration 0: invalid column 'BadColumn' -> returns error
        # Iteration 1: model corrects to 'Weekly_Sales' -> succeeds
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("statistics", {"column": "BadColumn"}),
            _make_tool_msg("statistics", {"column": "Weekly_Sales"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("After correcting the column name, the mean sales is 225.0."),
        ]

        agent = Agent()
        res = agent.run("What is average sales?", sample_df)

        assert "225.0" in res["answer"]
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 2
        assert tool_steps[0]["success"] is False
        assert "not found" in tool_steps[0]["error"]
        assert tool_steps[1]["success"] is True


def test_agent_multiple_tool_failures(sample_df: pd.DataFrame):
    """Test Agent when all tool attempts fail and final explanation conveys the error."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("statistics", {"column": "NonExistent1"}),
            _make_tool_msg("statistics", {"column": "NonExistent2"}),
            _make_text_msg("Done with tools."),
            _make_text_msg("I could not calculate statistics because neither column exists in the dataset."),
        ]

        agent = Agent()
        res = agent.run("Calculate stats on missing columns", sample_df)

        assert "could not calculate" in res["answer"]
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 2
        assert all(t["success"] is False for t in tool_steps)


def test_agent_execution_trace_structure(sample_df: pd.DataFrame):
    """Test that the execution trace records complete metadata for all steps."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("dataset_info", {}),
            _make_text_msg("Done."),
            _make_text_msg("The dataset has 6 rows and 6 columns."),
        ]

        agent = Agent()
        res = agent.run("Describe dataset", sample_df)

        trace = res["trace"]
        assert trace[0]["step"] == "question"
        assert trace[0]["question"] == "Describe dataset"
        assert trace[1]["step"] == "tool_call"
        assert trace[1]["tool"] == "dataset_info"
        assert trace[1]["success"] is True
        assert trace[1]["error"] is None
        assert trace[2]["step"] == "final_answer"
        assert trace[2]["tool_used"] is True
        assert trace[2]["iterations"] == 1


def test_agent_receives_dataset_context(sample_df: pd.DataFrame):
    """Test Agent receives compact dataset profile context in the initial system prompt."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.return_value = _make_text_msg("I understand the dataset structure.")

        agent = Agent()
        agent.run("What columns do we have?", sample_df)

        # Inspect the system message passed to LLMClient.chat
        system_msg = mock_llm_instance.chat.call_args[0][0][0]
        assert system_msg["role"] == "system"
        assert "[Active Dataset Context]" in system_msg["content"]
        assert "Weekly_Sales" in system_msg["content"]
        assert "Store" in system_msg["content"]
        assert "Date" in system_msg["content"]


def test_agent_dataset_context_compact_and_no_raw_dump(sample_df: pd.DataFrame):
    """Test dataset context is compact and does not dump full raw tabular rows.

    The system prompt grew intentionally in V5.3 (analytical reasoning guidelines
    were added).  The meaningful constraints remain: no raw row-level data is
    serialised, and the combined prompt stays well below the model context limit.
    """
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.return_value = _make_text_msg("Answer.")

        agent = Agent()
        agent.run("Analyze data", sample_df)

        system_msg = mock_llm_instance.chat.call_args[0][0][0]["content"]
        # Must remain compact — well under the model's 32 K context window
        assert len(system_msg) < 8000
        # Must not contain raw row-by-row data dump
        assert "2024-01-15 00:00:00" not in system_msg


def test_agent_run_with_empty_or_none_df(empty_df: pd.DataFrame):
    """Test Agent handles df=None or empty DataFrame without crashing."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.return_value = _make_text_msg("No data is loaded.")

        agent = Agent()
        res_none = agent.run("Hello", None)
        assert res_none["answer"] == "No data is loaded."

        res_empty = agent.run("Hello", empty_df)
        assert res_empty["answer"] == "No data is loaded."


def test_agent_analytical_comparison_multi_step(sample_df: pd.DataFrame):
    """Test Agent performing a multi-step analytical comparison workflow."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("groupby_analysis", {"group_column": "Store", "value_column": "Weekly_Sales", "top_n": 2}),
            _make_tool_msg("time_analysis", {"date_column": "Date", "value_column": "Weekly_Sales", "group_column": "Store", "filter_values": ["C", "B"]}),
            _make_text_msg("Done."),
            _make_text_msg("Store C had highest sales followed by Store B, with both showing monthly growth."),
        ]

        agent = Agent()
        res = agent.run("Compare top 2 stores over time", sample_df)

        assert "Store C" in res["answer"]
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 2
        assert tool_steps[0]["tool"] == "groupby_analysis"
        assert tool_steps[1]["tool"] == "time_analysis"
        assert res["trace"][-1]["tool_used"] is True


def test_agent_analytical_trend_workflow(sample_df: pd.DataFrame):
    """Test Agent analytical trend workflow using time_analysis."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.chat.side_effect = [
            _make_tool_msg("time_analysis", {"date_column": "Date", "value_column": "Weekly_Sales", "period": "month"}),
            _make_text_msg("Done."),
            _make_text_msg("Sales showed a strictly increasing trend from 100 to 350 across the 6 months."),
        ]

        agent = Agent()
        res = agent.run("What is the sales trend?", sample_df)

        assert "strictly increasing" in res["answer"]
        tool_steps = [t for t in res["trace"] if t.get("step") == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0]["tool"] == "time_analysis"


def test_v54_direct_answers_remain_concise(sample_df: pd.DataFrame):
    """V5.4 keeps non-analytical replies on the existing direct-answer path."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.return_value = _make_text_msg("Hello! What would you like to analyze?")

        result = Agent().run("Hello", sample_df)

        assert result["answer"] == "Hello! What would you like to analyze?"
        assert result["trace"][-1] == {"step": "final_answer", "tool_used": False}
        assert mock_llm.chat.call_count == 1


def test_v54_final_answer_prompt_is_grounded_and_structured(sample_df: pd.DataFrame):
    """The isolated final call receives deterministic findings and grounding rules."""
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.side_effect = [
            _make_tool_msg("time_analysis", {
                "date_column": "Date", "value_column": "Weekly_Sales", "period": "month",
            }),
            _make_text_msg("Tools complete."),
            _make_text_msg("Conclusion: sales increased.\n\nKey findings\n- The trend was strictly_increasing."),
        ]

        result = Agent().run("Analyze the sales trend", sample_df)

        assert result["answer"].startswith("Conclusion:")
        final_messages = mock_llm.chat.call_args_list[-1].args[0]
        assert mock_llm.chat.call_args_list[-1].kwargs["tool_choice"] == "none"
        assert final_messages[0]["content"] == FINAL_EXPLANATION_SYSTEM_PROMPT
        assert "Trend direction: strictly_increasing" in final_messages[1]["content"]
        assert "use ONLY the values returned by the tools" in FINAL_EXPLANATION_SYSTEM_PROMPT


def test_v54_insight_summary_covers_comparison_correlation_and_limitations():
    """V5.4 surfaces existing tool metrics without recalculating them."""
    tool_results = [
        {"tool_name": "percentage_change", "result": {
            "comparison_summary": "increased", "absolute_change": 25.0,
            "percentage_change": 12.5,
        }},
        {"tool_name": "correlation_analysis", "result": {
            "strongest_positive": ("Advertising", 0.8),
            "strongest_negative": ("Price", -0.4),
        }},
        {"tool_name": "groupby_analysis", "result": {
            "result": {"North": 120.0, "South": 95.0},
            "ranking": ["North", "South"], "best_group": "North", "worst_group": "South",
        }},
        {"tool_name": "statistics", "result": {"column": "Sales", "skewness": 1.2, "coefficient_of_variation": 1.1}},
        {"tool_name": "time_analysis", "result": {"note": "Result truncated to the available subset."}},
    ]

    summary = _summarise_tool_results(tool_results, has_figure=True)

    assert "absolute change +25 (+12.5%)" in summary
    assert "Advertising (r=0.800)" in summary
    assert "Price (r=-0.400)" in summary
    assert "Leading ranking: North (120), South (95)" in summary
    assert "notably right (positively) skewed" in summary
    assert "chart was generated" in summary
    assert "truncated" in summary.lower()


def test_v54_insight_summary_reports_missing_values_and_outliers():
    """Missing-data and both outlier result shapes receive deterministic hints."""
    summary = _summarise_tool_results([
        {"tool_name": "missing_values", "result": {
            "total_missing_values": 4,
            "columns_with_missing": {"Discount": {"missing_count": 3, "missing_percentage": 50.0}},
        }},
        {"tool_name": "outlier_analysis", "result": {
            "column": "Sales", "outlier_count": 2, "outlier_percentage": 10.0,
            "example_outlier_values": [900.0],
        }},
        {"tool_name": "outlier_analysis", "result": {
            "results": {"Profit": {"outlier_count": 1}},
        }},
    ], has_figure=False)

    assert "Discount (3 missing; 50.0%)" in summary
    assert "Sales has 2 outlier(s) (10.0%)" in summary
    assert "Profit (1 outlier(s))" in summary


def test_v54_why_and_visualization_guidance_is_explicit():
    """Tool selection guidance preserves association language and chart restraint."""
    assert "after identifying the change with time_analysis or percentage_change" in SYSTEM_PROMPT
    assert "associations only" in SYSTEM_PROMPT
    assert "create_visualization" in SYSTEM_PROMPT
    assert "do not generate charts for simple scalar lookups" in SYSTEM_PROMPT



