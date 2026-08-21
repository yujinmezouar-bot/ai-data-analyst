import pandas as pd
import pytest

from agent.agent import Agent, _estimate_request_chars, MAX_LLM_REQUEST_CHARS, TOOL_SCHEMAS
from tools.dataset_info import format_datasets_context


@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "Sales": [100.0, 150.0, 200.0]
    })


@pytest.fixture
def sample_products_df():
    return pd.DataFrame({
        "Product_ID": ["A", "B", "C"],
        "Category": ["Electronics", "Clothing", "Food"]
    })


def test_agent_backward_compatibility(sample_sales_df):
    agent = Agent()
    # We patch LLM to avoid real calls
    from unittest.mock import patch
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.return_value.tool_calls = None
        mock_llm.chat.return_value.content = "No tools needed."
        
        result = agent.run("What is the data shape?", df=sample_sales_df)
    
    assert agent.primary_df is not None
    assert agent.primary_df.shape == (3, 2)
    assert agent.active_datasets["default"] is not None


def test_format_datasets_context_multiple(sample_sales_df, sample_products_df):
    datasets = {
        "sales.csv": sample_sales_df,
        "products.csv": sample_products_df
    }
    context = format_datasets_context(datasets)
    
    assert "[Dataset: sales.csv]" in context
    assert "[Dataset: products.csv]" in context
    assert "Shape: 3 rows, 2 columns" in context
    assert "- Numeric columns: Store, Sales" in context
    assert "Product_ID" in context
    assert "Category" in context


def test_format_datasets_context_empty():
    assert format_datasets_context({}) == ""
    assert format_datasets_context(None) == ""


def test_multi_dataset_context_in_budget(sample_sales_df, sample_products_df):
    datasets = {
        "sales.csv": sample_sales_df,
        "products.csv": sample_products_df
    }
    
    from unittest.mock import patch
    with patch("agent.agent.LLMClient") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.chat.return_value.tool_calls = None
        mock_llm.chat.return_value.content = "Multi data answered."
        
        agent = Agent()
        agent.run("What are these datasets?", df=None, datasets=datasets)
    
    # Check what was actually sent
    sent_messages = mock_llm.chat.call_args_list[0].args[0]
    sys_prompt = sent_messages[0]["content"]
    assert "[Dataset: sales.csv]" in sys_prompt
    assert _estimate_request_chars(sent_messages, TOOL_SCHEMAS) <= MAX_LLM_REQUEST_CHARS


def test_agent_resolves_dataset_name(sample_sales_df, sample_products_df):
    agent = Agent()
    agent.active_datasets = {
        "sales.csv": sample_sales_df,
        "products.csv": sample_products_df
    }
    
    # Using dataset_info tool which just returns the profile
    result = agent._execute_tool("dataset_info", {"dataset_name": "products.csv"}, df=sample_sales_df)
    
    # The tool should have processed products.csv
    assert result["num_columns"] == 2
    assert result["categorical_columns"] == ["Product_ID", "Category"]


def test_agent_handles_missing_dataset_name(sample_sales_df, sample_products_df):
    agent = Agent()
    agent.active_datasets = {
        "sales.csv": sample_sales_df,
        "products.csv": sample_products_df
    }
    
    # No dataset_name provided, df=sample_sales_df
    result = agent._execute_tool("dataset_info", {}, df=sample_sales_df)
    
    assert result["num_columns"] == 2
    assert "Store" in result["numeric_columns"]


def test_agent_handles_invalid_dataset_name(sample_sales_df):
    agent = Agent()
    agent.active_datasets = {"sales.csv": sample_sales_df}
    
    result = agent._execute_tool("dataset_info", {"dataset_name": "unknown.csv"}, df=sample_sales_df)
    
    assert "error" in result
    assert "Dataset 'unknown.csv' not found" in result["error"]


def test_agent_handles_null_dataset_name(sample_sales_df, sample_products_df):
    agent = Agent()
    agent.active_datasets = {
        "sales.csv": sample_sales_df,
        "products.csv": sample_products_df
    }
    
    # explicit None/null
    result = agent._execute_tool("dataset_info", {"dataset_name": None}, df=sample_sales_df)
    
    assert result["num_columns"] == 2
    assert "Store" in result["numeric_columns"]


def test_dataset_name_is_stripped_from_tool_args():
    # Verify that dataset_name is not passed to the underlying python tool function
    # by testing a tool that does not accept dataset_name as a kwarg in python.
    # missing_values takes (df)
    agent = Agent()
    df = pd.DataFrame({"A": [1, None, 3]})
    agent.active_datasets = {"primary.csv": df}
    
    # If dataset_name was not stripped, this would raise TypeError: missing_values() got an unexpected keyword argument 'dataset_name'
    result = agent._execute_tool("missing_values", {"dataset_name": "primary.csv"}, df=df)
    
    # We should get a valid result back without TypeError
    assert isinstance(result, dict)
    assert "A" in result["columns_with_missing"]

