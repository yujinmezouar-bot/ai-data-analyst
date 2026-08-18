import pandas as pd
from tools.correlation import correlation_analysis, CORRELATION_ANALYSIS_SCHEMA


def test_correlation_regression_null_args(sample_df: pd.DataFrame):
    """Regression test: verify Groq null args for optional parameters don't break."""
    # 1. column=None
    res1 = correlation_analysis(sample_df, column=None)
    assert "error" not in res1
    assert "top_correlations" in res1

    # 2. columns=None
    res2 = correlation_analysis(sample_df, columns=None)
    assert "error" not in res2
    assert "top_correlations" in res2

    # 3. top_n=None
    res3 = correlation_analysis(sample_df, top_n=None)
    assert "error" not in res3
    assert "top_correlations" in res3

    # 4. All optional arguments explicitly None
    res4 = correlation_analysis(sample_df, column=None, columns=None, top_n=None)
    assert "error" not in res4
    assert "top_correlations" in res4


def test_correlation_single_target_column(sample_df: pd.DataFrame):
    """Test correlation against a single target column."""
    res = correlation_analysis(sample_df, column="Weekly_Sales")
    assert "error" not in res
    assert res["target_column"] == "Weekly_Sales"
    assert "correlations" in res
    assert "Temperature" in res["correlations"]
    assert "Fuel_Price" in res["correlations"]
    # Weekly_Sales and Temperature are perfectly linearly correlated in sample_df (corr = 1.0)
    assert res["correlations"]["Temperature"] == 1.0


def test_correlation_subset_columns(sample_df: pd.DataFrame):
    """Test correlation restricted to specific subset of numeric columns."""
    res = correlation_analysis(sample_df, columns=["Weekly_Sales", "Temperature"])
    assert "error" not in res
    assert "top_correlations" in res
    assert len(res["top_correlations"]) == 1


def test_correlation_constant_column_dropped(df_constants: pd.DataFrame):
    """Test constant columns are excluded and noted."""
    res = correlation_analysis(df_constants)
    assert "error" not in res
    assert "Constant" not in res["numeric_columns_analyzed"]
    assert "note" in res
    assert "Excluded constant column" in res["note"]


def test_correlation_insufficient_numeric_columns():
    """Test when dataset has fewer than 2 numeric columns with variance."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    res = correlation_analysis(df)
    assert "error" in res
    assert "Not enough numeric columns" in res["error"]


def test_correlation_invalid_target_column(sample_df: pd.DataFrame):
    """Test correlation with non-existent target column."""
    res = correlation_analysis(sample_df, column="MissingCol")
    assert "error" in res
    assert "not found" in res["error"]


def test_correlation_non_numeric_target_column(sample_df: pd.DataFrame):
    """Test correlation with categorical target column."""
    res = correlation_analysis(sample_df, column="Store")
    assert "error" in res
    assert "not numeric" in res["error"]


def test_correlation_invalid_columns_subset(sample_df: pd.DataFrame):
    """Test subset of columns containing non-existent or non-numeric column."""
    res_miss = correlation_analysis(sample_df, columns=["Weekly_Sales", "MissingCol"])
    assert "error" in res_miss
    assert "not found" in res_miss["error"]

    res_cat = correlation_analysis(sample_df, columns=["Weekly_Sales", "Store"])
    assert "error" in res_cat
    assert "not numeric" in res_cat["error"]


def test_correlation_none_df():
    """Test correlation with df=None."""
    res = correlation_analysis(None)
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_correlation_schema():
    """Test CORRELATION_ANALYSIS_SCHEMA properties and null type declarations."""
    schema = CORRELATION_ANALYSIS_SCHEMA["function"]
    assert schema["name"] == "correlation_analysis"
    assert "null" in schema["parameters"]["properties"]["column"]["type"]
    assert "null" in schema["parameters"]["properties"]["columns"]["type"]
    assert "null" in schema["parameters"]["properties"]["top_n"]["type"]


def test_correlation_strongest_positive_and_negative_pairs(sample_df: pd.DataFrame):
    """Test strongest_positive_pair and analytical note in overall correlation mode."""
    res = correlation_analysis(sample_df)
    assert "strongest_positive_pair" in res
    assert res["strongest_positive_pair"] is not None
    assert "analytical_note" in res
    assert "causation" in res["analytical_note"]


def test_correlation_single_column_positive_and_negative(sample_df: pd.DataFrame):
    """Test strongest_positive and strongest_negative for a single target column."""
    res = correlation_analysis(sample_df, column="Weekly_Sales")
    assert "strongest_positive" in res
    assert "analytical_note" in res

