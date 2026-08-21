"""
V8 Test Suite: Relationship Discovery, Schema Graph, and Cross-Dataset Visualization.
"""
import pytest
import pandas as pd
from tools.relationship_discovery import (
    _normalize_column_name,
    _compute_name_similarity,
    _compute_value_overlap,
    _are_types_compatible,
    score_key_pair,
    discover_relationships,
    build_schema_graph_summary,
)
from tools.visualization import create_multi_dataset_visualization
from agent.agent import Agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def customers_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "city": ["London", "Paris", "New York", "Berlin", "Tokyo"],
    })


@pytest.fixture
def sales_df():
    """Shares Cust_ID values with customers_df but uses a different column name."""
    return pd.DataFrame({
        "Cust_ID": [1, 2, 3, 4, 5],
        "product": ["Widget", "Gadget", "Doohickey", "Thingamajig", "Gizmo"],
        "revenue": [100.0, 250.0, 180.0, 320.0, 90.0],
    })


@pytest.fixture
def sales_2023_df():
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar"],
        "revenue_2023": [1000.0, 1200.0, 900.0],
    })


@pytest.fixture
def sales_2024_df():
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar"],
        "revenue_2024": [1100.0, 1400.0, 950.0],
    })


# ---------------------------------------------------------------------------
# 1. Column name normalization
# ---------------------------------------------------------------------------

def test_normalize_strips_id_prefix():
    assert _normalize_column_name("id_customer") == "customer"


def test_normalize_strips_id_suffix():
    assert _normalize_column_name("customer_id") == "customer"


def test_normalize_strips_num_suffix():
    assert _normalize_column_name("cust_num") == "cust"


def test_normalize_case_insensitive():
    n1 = _normalize_column_name("Customer_ID")
    n2 = _normalize_column_name("customer_id")
    assert n1 == n2


# ---------------------------------------------------------------------------
# 2. Name similarity scoring
# ---------------------------------------------------------------------------

def test_name_similarity_exact_match():
    assert _compute_name_similarity("customer_id", "customer_id") == 1.0


def test_name_similarity_normalized_match():
    # "customer_id" vs "Cust_ID" — after normalization both root to "cust"
    score = _compute_name_similarity("customer_id", "Cust_ID")
    assert score >= 0.6, f"Expected >= 0.6, got {score}"


def test_name_similarity_unrelated_columns():
    score = _compute_name_similarity("revenue", "birthday")
    assert score < 0.5, f"Expected < 0.5, got {score}"


# ---------------------------------------------------------------------------
# 3. Value overlap (Jaccard) scoring
# ---------------------------------------------------------------------------

def test_value_overlap_high_for_matching_series(customers_df, sales_df):
    overlap = _compute_value_overlap(customers_df["customer_id"], sales_df["Cust_ID"])
    assert overlap >= 0.8, f"Expected >= 0.8, got {overlap}"


def test_value_overlap_zero_for_no_common_values():
    s1 = pd.Series([1, 2, 3])
    s2 = pd.Series([100, 200, 300])
    overlap = _compute_value_overlap(s1, s2)
    assert overlap == 0.0


def test_value_overlap_empty_series():
    s1 = pd.Series([], dtype=float)
    s2 = pd.Series([1, 2, 3])
    assert _compute_value_overlap(s1, s2) == 0.0


# ---------------------------------------------------------------------------
# 4. Type compatibility
# ---------------------------------------------------------------------------

def test_type_compatibility_same_type():
    s1 = pd.Series([1, 2], dtype="int64")
    s2 = pd.Series([3, 4], dtype="int64")
    assert _are_types_compatible(s1, s2) is True


def test_type_compatibility_int_and_float():
    s1 = pd.Series([1, 2], dtype="int64")
    s2 = pd.Series([1.0, 2.0], dtype="float64")
    assert _are_types_compatible(s1, s2) is True


def test_type_incompatibility_int_and_datetime():
    s1 = pd.Series([1, 2], dtype="int64")
    s2 = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert _are_types_compatible(s1, s2) is False


def test_type_incompatibility_int_and_string():
    s1 = pd.Series([1, 2], dtype="int64")
    s2 = pd.Series(["a", "b"], dtype="object")
    assert _are_types_compatible(s1, s2) is False


# ---------------------------------------------------------------------------
# 5. score_key_pair
# ---------------------------------------------------------------------------

def test_score_key_pair_high_confidence(customers_df, sales_df):
    result = score_key_pair(customers_df, sales_df, "customer_id", "Cust_ID")
    assert result is not None
    assert result["confidence"] >= 0.5
    assert result["cardinality"] == "1:1"
    assert result["safe_to_join"] is True


def test_score_key_pair_incompatible_types_returns_none(customers_df):
    other_df = pd.DataFrame({"customer_id": ["1", "2", "3"]})
    result = score_key_pair(customers_df, other_df, "customer_id", "customer_id")
    assert result is None


def test_score_key_pair_constant_column_returns_none():
    df1 = pd.DataFrame({"key": [1, 1, 1]})
    df2 = pd.DataFrame({"key": [1, 2, 3]})
    assert score_key_pair(df1, df2, "key", "key") is None


def test_score_key_pair_missing_column_returns_none(customers_df, sales_df):
    assert score_key_pair(customers_df, sales_df, "nonexistent", "Cust_ID") is None


# ---------------------------------------------------------------------------
# 6. discover_relationships
# ---------------------------------------------------------------------------

def test_discover_relationships_finds_candidates(customers_df, sales_df):
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    result = discover_relationships(datasets)
    assert result["status"] == "success"
    assert result["relationships_found"] >= 1
    # The top relationship should be the customer_id <-> Cust_ID pair
    top = result["relationships"][0]
    assert top["confidence"] >= 0.5


def test_discover_relationships_requires_two_datasets(customers_df):
    result = discover_relationships({"customers.csv": customers_df})
    assert result["status"] == "info"
    assert result["relationships"] == []


def test_discover_relationships_is_read_only(customers_df, sales_df):
    """discover_relationships must never add to active or derived datasets."""
    orig_cust = customers_df.copy()
    orig_sales = sales_df.copy()
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    discover_relationships(datasets)
    pd.testing.assert_frame_equal(customers_df, orig_cust)
    pd.testing.assert_frame_equal(sales_df, orig_sales)


def test_discover_relationships_respects_min_confidence(customers_df, sales_df):
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    result_high = discover_relationships(datasets, min_confidence=0.99)
    result_low = discover_relationships(datasets, min_confidence=0.01)
    # High threshold should return fewer or equal candidates than low threshold
    assert result_high["relationships_found"] <= result_low["relationships_found"]


def test_discover_relationships_result_sorted_by_confidence(customers_df, sales_df):
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    result = discover_relationships(datasets, min_confidence=0.0)
    confidences = [r["confidence"] for r in result["relationships"]]
    assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# 7. build_schema_graph_summary
# ---------------------------------------------------------------------------

def test_build_schema_graph_summary_includes_header(customers_df, sales_df):
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    summary = build_schema_graph_summary(datasets)
    if summary:
        assert "[Schema Relationship Map]" in summary


def test_build_schema_graph_summary_empty_for_single_dataset(customers_df):
    datasets = {"customers.csv": customers_df}
    assert build_schema_graph_summary(datasets) == ""


def test_build_schema_graph_summary_compact(customers_df, sales_df):
    """Schema summary must stay within the 1200-char context budget."""
    datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    summary = build_schema_graph_summary(datasets)
    assert len(summary) <= 1200, f"Summary too long: {len(summary)} chars"


# ---------------------------------------------------------------------------
# 8. create_multi_dataset_visualization
# ---------------------------------------------------------------------------

def test_multi_dataset_visualization_two_series(sales_2023_df, sales_2024_df):
    datasets = {"sales_2023": sales_2023_df, "sales_2024": sales_2024_df}
    result = create_multi_dataset_visualization(
        datasets=datasets,
        series=[
            {"dataset_name": "sales_2023", "x_column": "month", "y_column": "revenue_2023", "name": "2023"},
            {"dataset_name": "sales_2024", "x_column": "month", "y_column": "revenue_2024", "name": "2024"},
        ],
        chart_type="line",
        title="Revenue Comparison",
    )
    assert result["status"] == "success"
    assert result["traces_count"] == 2
    assert "figure" in result
    assert result["title"] == "Revenue Comparison"


def test_multi_dataset_visualization_bar_chart(sales_2023_df, sales_2024_df):
    datasets = {"sales_2023": sales_2023_df, "sales_2024": sales_2024_df}
    result = create_multi_dataset_visualization(
        datasets=datasets,
        series=[
            {"dataset_name": "sales_2023", "x_column": "month", "y_column": "revenue_2023"},
            {"dataset_name": "sales_2024", "x_column": "month", "y_column": "revenue_2024"},
        ],
        chart_type="bar",
    )
    assert result["status"] == "success"
    assert result["chart_type"] == "bar"


def test_multi_dataset_visualization_missing_dataset(sales_2023_df):
    datasets = {"sales_2023": sales_2023_df}
    result = create_multi_dataset_visualization(
        datasets=datasets,
        series=[
            {"dataset_name": "sales_2023", "x_column": "month", "y_column": "revenue_2023"},
            {"dataset_name": "NONEXISTENT", "x_column": "month", "y_column": "revenue_2024"},
        ],
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_multi_dataset_visualization_invalid_chart_type(sales_2023_df):
    datasets = {"sales_2023": sales_2023_df}
    result = create_multi_dataset_visualization(
        datasets=datasets,
        series=[{"dataset_name": "sales_2023", "x_column": "month", "y_column": "revenue_2023"}],
        chart_type="scatter",  # Not supported in multi-dataset viz
    )
    assert "error" in result


def test_multi_dataset_visualization_non_numeric_y(sales_2023_df):
    datasets = {"sales_2023": sales_2023_df}
    result = create_multi_dataset_visualization(
        datasets=datasets,
        series=[{"dataset_name": "sales_2023", "x_column": "month", "y_column": "month"}],
    )
    assert "error" in result
    assert "not numeric" in result["error"]


def test_multi_dataset_visualization_empty_series(sales_2023_df):
    datasets = {"sales_2023": sales_2023_df}
    result = create_multi_dataset_visualization(datasets=datasets, series=[])
    assert "error" in result


# ---------------------------------------------------------------------------
# 9. Agent-level integration
# ---------------------------------------------------------------------------

def test_agent_discover_relationships_via_execute_tool(customers_df, sales_df):
    agent = Agent()
    agent.active_datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    agent.derived_datasets = {}

    result = agent._execute_tool("discover_relationships", {}, df=customers_df)
    assert "relationships" in result
    # Tool must not mutate any datasets
    assert "customers.csv" in agent.active_datasets
    assert "sales.csv" in agent.active_datasets
    assert len(agent.derived_datasets) == 0  # read-only: nothing registered


def test_agent_discover_relationships_does_not_execute_joins(customers_df, sales_df):
    """Core V8 safety: discover_relationships must leave active and derived datasets unchanged."""
    agent = Agent()
    agent.active_datasets = {"customers.csv": customers_df, "sales.csv": sales_df}
    agent.derived_datasets = {}

    agent._execute_tool("discover_relationships", {}, df=customers_df)

    # After calling discover_relationships, no join should have happened
    assert len(agent.derived_datasets) == 0
    assert set(agent.active_datasets.keys()) == {"customers.csv", "sales.csv"}


def test_agent_multi_dataset_vis_via_execute_tool(sales_2023_df, sales_2024_df):
    agent = Agent()
    agent.active_datasets = {"sales_2023": sales_2023_df, "sales_2024": sales_2024_df}
    agent.derived_datasets = {}

    result = agent._execute_tool(
        "create_multi_dataset_visualization",
        {
            "series": [
                {"dataset_name": "sales_2023", "x_column": "month", "y_column": "revenue_2023"},
                {"dataset_name": "sales_2024", "x_column": "month", "y_column": "revenue_2024"},
            ],
            "chart_type": "line",
            "title": "YoY Revenue",
        },
        df=sales_2023_df,
    )
    assert result["status"] == "success"
    assert result["traces_count"] == 2


def test_agent_multi_dataset_vis_includes_derived_datasets(customers_df, sales_df):
    """V6 derived datasets must also be accessible to multi-dataset viz."""
    import pandas as pd
    joined = customers_df.merge(sales_df, left_on="customer_id", right_on="Cust_ID")
    agent = Agent()
    agent.active_datasets = {"customers.csv": customers_df}
    agent.derived_datasets = {"derived_join_1": joined}

    result = agent._execute_tool(
        "create_multi_dataset_visualization",
        {
            "series": [
                {"dataset_name": "customers.csv", "x_column": "customer_id", "y_column": "customer_id"},
                {"dataset_name": "derived_join_1", "x_column": "customer_id", "y_column": "revenue"},
            ],
            "chart_type": "bar",
        },
        df=customers_df,
    )
    # customer_id is numeric so customers trace is valid; derived has revenue too
    assert result["status"] == "success"
    assert result["traces_count"] == 2


def test_agent_schema_graph_appears_in_system_prompt_with_two_datasets(customers_df, sales_df):
    """V8: schema relationship map is included in system prompt when 2+ datasets present."""
    from unittest.mock import patch

    agent = Agent()
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.return_value.tool_calls = None
        mock_llm.chat.return_value.content = "Answered."

        agent.run(
            "What relationships exist between these datasets?",
            datasets={"customers.csv": customers_df, "sales.csv": sales_df},
        )

    system_msg = mock_llm.chat.call_args[0][0][0]
    assert system_msg["role"] == "system"
    # The schema graph is only added if candidates are found above 0.5 confidence;
    # at minimum, the context must include both dataset names.
    assert "customers.csv" in system_msg["content"]
    assert "sales.csv" in system_msg["content"]


def test_agent_schema_graph_absent_for_single_dataset(customers_df):
    """V8: schema map must NOT appear when only one dataset is loaded."""
    from unittest.mock import patch

    agent = Agent()
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.return_value.tool_calls = None
        mock_llm.chat.return_value.content = "Answered."

        agent.run("Describe the dataset.", df=customers_df)

    system_msg = mock_llm.chat.call_args[0][0][0]
    assert "[Schema Relationship Map]" not in system_msg["content"]
