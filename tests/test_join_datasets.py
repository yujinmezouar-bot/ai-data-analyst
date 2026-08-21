import pandas as pd
import pytest
from unittest.mock import patch

from agent.agent import Agent
from tools.join_datasets import inspect_join_viability, execute_join, MAX_ROW_COUNT_LIMIT


@pytest.fixture
def sample_customers_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "city": ["New York", "London", "Paris"],
    })


@pytest.fixture
def sample_profiles_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "loyalty_tier": ["Gold", "Silver", "Bronze"],
    })


@pytest.fixture
def sample_orders_df():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "customer_id": [1, 1, 2, 3],
        "amount": [50.0, 75.0, 120.0, 200.0],
    })


@pytest.fixture
def sample_items_df():
    return pd.DataFrame({
        "customer_id": [1, 1, 2, 2],
        "item_name": ["Widget", "Gadget", "Doohickey", "Thingamajig"],
    })


# 1. 1:1 allowed
def test_join_cardinality_1_to_1_allowed(sample_customers_df, sample_profiles_df):
    viability = inspect_join_viability(sample_customers_df, sample_profiles_df, "customer_id", "customer_id")
    assert viability["safe_to_join"] is True
    assert viability["cardinality"] == "1:1"

    result = execute_join(sample_customers_df, sample_profiles_df, "customer_id", "customer_id", how="inner")
    assert result["status"] == "success"
    assert result["cardinality"] == "1:1"
    assert len(result["dataframe"]) == 3
    assert "loyalty_tier" in result["columns"]


# 2. 1:N allowed
def test_join_cardinality_1_to_n_allowed(sample_customers_df, sample_orders_df):
    viability = inspect_join_viability(sample_customers_df, sample_orders_df, "customer_id", "customer_id")
    assert viability["safe_to_join"] is True
    assert viability["cardinality"] == "1:N"

    result = execute_join(sample_customers_df, sample_orders_df, "customer_id", "customer_id", how="inner")
    assert result["status"] == "success"
    assert result["cardinality"] == "1:N"
    assert len(result["dataframe"]) == 4


# 3. N:1 allowed
def test_join_cardinality_n_to_1_allowed(sample_orders_df, sample_customers_df):
    viability = inspect_join_viability(sample_orders_df, sample_customers_df, "customer_id", "customer_id")
    assert viability["safe_to_join"] is True
    assert viability["cardinality"] == "N:1"

    result = execute_join(sample_orders_df, sample_customers_df, "customer_id", "customer_id", how="inner")
    assert result["status"] == "success"
    assert result["cardinality"] == "N:1"
    assert len(result["dataframe"]) == 4


# 4. N:N blocked
def test_join_cardinality_n_to_n_blocked(sample_orders_df, sample_items_df):
    viability = inspect_join_viability(sample_orders_df, sample_items_df, "customer_id", "customer_id")
    assert viability["safe_to_join"] is False
    assert viability["cardinality"] == "N:N"

    result = execute_join(sample_orders_df, sample_items_df, "customer_id", "customer_id", how="inner")
    assert "error" in result
    assert "N:N joins are blocked" in result["error"]


# 5. Output row safety limit
def test_join_output_row_safety_limit(sample_customers_df, sample_orders_df):
    with patch("tools.join_datasets.MAX_ROW_COUNT_LIMIT", 2):
        result = execute_join(sample_customers_df, sample_orders_df, "customer_id", "customer_id", how="inner")
        assert "error" in result
        assert "exceeds maximum row limit" in result["error"]


# 6. Invalid join type
def test_join_invalid_join_type(sample_customers_df, sample_orders_df):
    result = execute_join(sample_customers_df, sample_orders_df, "customer_id", "customer_id", how="diagonal")
    assert "error" in result
    assert "Invalid join type" in result["error"]


# 7. Incompatible dtypes
def test_join_incompatible_dtypes(sample_customers_df):
    other_df = pd.DataFrame({
        "customer_id": ["1", "2", "3"],
        "notes": ["A", "B", "C"],
    })
    viability = inspect_join_viability(sample_customers_df, other_df, "customer_id", "customer_id")
    assert viability["safe_to_join"] is False
    assert "Incompatible data types" in viability["error"]

    result = execute_join(sample_customers_df, other_df, "customer_id", "customer_id")
    assert "error" in result
    assert "Incompatible data types" in result["error"]


# 8. Missing join columns
def test_join_missing_join_columns(sample_customers_df, sample_orders_df):
    viability = inspect_join_viability(sample_customers_df, sample_orders_df, "non_existent_col", "customer_id")
    assert viability["safe_to_join"] is False
    assert "not found in left dataset" in viability["error"]

    result = execute_join(sample_customers_df, sample_orders_df, "customer_id", "non_existent_col")
    assert "error" in result
    assert "not found in right dataset" in result["error"]


# 9. Derived dataset registration in Agent
def test_agent_derived_dataset_registration(sample_customers_df, sample_orders_df):
    agent = Agent()
    agent.active_datasets = {
        "customers.csv": sample_customers_df,
        "orders.csv": sample_orders_df,
    }

    result = agent._execute_tool(
        "execute_join",
        {
            "left_dataset": "customers.csv",
            "right_dataset": "orders.csv",
            "left_on": "customer_id",
            "right_on": "customer_id",
            "how": "inner",
        },
        df=sample_customers_df,
    )

    assert result["status"] == "success"
    assert result["dataset_name"] == "derived_join_1"
    assert "derived_join_1" in agent.derived_datasets
    assert isinstance(agent.derived_datasets["derived_join_1"], pd.DataFrame)
    # Ensure raw dataframe is NOT returned to LLM
    assert "dataframe" not in result


# 10. Original datasets remain unchanged
def test_original_datasets_remain_unchanged(sample_customers_df, sample_orders_df):
    agent = Agent()
    orig_customers = sample_customers_df.copy()
    orig_orders = sample_orders_df.copy()

    agent.active_datasets = {
        "customers.csv": sample_customers_df,
        "orders.csv": sample_orders_df,
    }

    agent._execute_tool(
        "execute_join",
        {
            "left_dataset": "customers.csv",
            "right_dataset": "orders.csv",
            "left_on": "customer_id",
            "right_on": "customer_id",
        },
        df=sample_customers_df,
    )

    # Verify original DataFrames were not mutated
    pd.testing.assert_frame_equal(agent.active_datasets["customers.csv"], orig_customers)
    pd.testing.assert_frame_equal(agent.active_datasets["orders.csv"], orig_orders)
    # Verify active_datasets does not contain derived data
    assert "derived_join_1" not in agent.active_datasets


# 11. Repeated joins stay bounded at 3 derived datasets (LRU eviction)
def test_repeated_joins_bounded_at_3_lru(sample_customers_df, sample_orders_df):
    agent = Agent()
    agent.active_datasets = {
        "customers.csv": sample_customers_df,
        "orders.csv": sample_orders_df,
    }

    # Execute 4 joins
    for i in range(4):
        agent._execute_tool(
            "execute_join",
            {
                "left_dataset": "customers.csv",
                "right_dataset": "orders.csv",
                "left_on": "customer_id",
                "right_on": "customer_id",
            },
            df=sample_customers_df,
        )

    # Capped at 3
    assert len(agent.derived_datasets) == 3
    # derived_join_1 should have been evicted (oldest)
    assert "derived_join_1" not in agent.derived_datasets
    assert "derived_join_2" in agent.derived_datasets
    assert "derived_join_3" in agent.derived_datasets
    assert "derived_join_4" in agent.derived_datasets


# 12. Derived dataset can be routed to existing V6 tools
def test_derived_dataset_routed_to_v6_tools(sample_customers_df, sample_orders_df):
    agent = Agent()
    agent.active_datasets = {
        "customers.csv": sample_customers_df,
        "orders.csv": sample_orders_df,
    }

    # Join
    join_res = agent._execute_tool(
        "execute_join",
        {
            "left_dataset": "customers.csv",
            "right_dataset": "orders.csv",
            "left_on": "customer_id",
            "right_on": "customer_id",
        },
        df=sample_customers_df,
    )
    derived_name = join_res["dataset_name"]

    # Query the derived dataset with statistics tool
    stats_res = agent._execute_tool(
        "statistics",
        {
            "dataset_name": derived_name,
            "column": "amount",
        },
        df=sample_customers_df,
    )

    assert "mean" in stats_res
    assert stats_res["count"] == 4
    assert stats_res["min"] == 50.0
    assert stats_res["max"] == 200.0

    # Query with groupby tool on a column from each of the original tables (city from customers, amount from orders)
    groupby_res = agent._execute_tool(
        "groupby_analysis",
        {
            "dataset_name": derived_name,
            "group_column": "city",
            "value_column": "amount",
            "agg_function": "sum",
        },
        df=sample_customers_df,
    )

    assert "result" in groupby_res
    assert len(groupby_res["result"]) == 3


# 13. Deterministic derived_join_N naming
def test_deterministic_derived_join_naming(sample_customers_df, sample_orders_df):
    agent = Agent()
    agent.active_datasets = {
        "customers.csv": sample_customers_df,
        "orders.csv": sample_orders_df,
    }

    res1 = agent._execute_tool(
        "execute_join",
        {
            "left_dataset": "customers.csv",
            "right_dataset": "orders.csv",
            "left_on": "customer_id",
            "right_on": "customer_id",
        },
        df=sample_customers_df,
    )
    assert res1["dataset_name"] == "derived_join_1"

    res2 = agent._execute_tool(
        "execute_join",
        {
            "left_dataset": "customers.csv",
            "right_dataset": "orders.csv",
            "left_on": "customer_id",
            "right_on": "customer_id",
        },
        df=sample_customers_df,
    )
    assert res2["dataset_name"] == "derived_join_2"
