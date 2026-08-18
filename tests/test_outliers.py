import pandas as pd
from tools.outliers import outlier_analysis, OUTLIER_ANALYSIS_SCHEMA


def test_outlier_analysis_regression_null_column(sample_df: pd.DataFrame):
    """Regression test: verify Groq null args for optional parameters don't break."""
    # 1. column=None
    res1 = outlier_analysis(sample_df, column=None)
    assert "error" not in res1
    assert "results" in res1

    # 2. column=None and multiplier=None
    res2 = outlier_analysis(sample_df, column=None, multiplier=None)
    assert "error" not in res2
    assert "results" in res2


def test_outlier_analysis_single_column_with_outlier(df_outliers: pd.DataFrame):
    """Test outlier detection on a column containing an extreme outlier."""
    res = outlier_analysis(df_outliers, column="With_Outlier")
    assert "error" not in res
    assert res["column"] == "With_Outlier"
    assert res["outlier_count"] >= 1
    assert res["outlier_percentage"] > 0
    assert "example_outlier_values" in res
    assert 500.0 in res["example_outlier_values"]


def test_outlier_analysis_single_column_normal(df_outliers: pd.DataFrame):
    """Test outlier detection on a normal column with no outliers."""
    res = outlier_analysis(df_outliers, column="Normal")
    assert "error" not in res
    assert res["outlier_count"] == 0
    assert res["outlier_percentage"] == 0.0


def test_outlier_analysis_all_columns_ranked(df_outliers: pd.DataFrame):
    """Test analyzing all numeric columns and ranking by outlier percentage."""
    res = outlier_analysis(df_outliers)
    assert "error" not in res
    assert "columns_analyzed" in res
    assert res["columns_analyzed"][0] == "With_Outlier"


def test_outlier_analysis_constant_column(df_constants: pd.DataFrame):
    """Test constant column detection within outlier analysis."""
    res = outlier_analysis(df_constants, column="Constant")
    assert "error" not in res
    assert "constant" in res["note"].lower()


def test_outlier_analysis_invalid_column(sample_df: pd.DataFrame):
    """Test outlier analysis on non-existent column."""
    res = outlier_analysis(sample_df, column="NonExistent")
    assert "error" in res
    assert "not found" in res["error"]


def test_outlier_analysis_non_numeric_column(sample_df: pd.DataFrame):
    """Test outlier analysis on categorical column."""
    res = outlier_analysis(sample_df, column="Store")
    assert "error" in res
    assert "not numeric" in res["error"]


def test_outlier_analysis_invalid_multiplier(sample_df: pd.DataFrame):
    """Test negative or zero multiplier."""
    res_zero = outlier_analysis(sample_df, multiplier=0)
    assert "error" in res_zero
    assert "must be a positive number" in res_zero["error"]

    res_neg = outlier_analysis(sample_df, multiplier=-1.5)
    assert "error" in res_neg
    assert "must be a positive number" in res_neg["error"]


def test_outlier_analysis_none_df():
    """Test outlier analysis with df=None."""
    res = outlier_analysis(None)
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_outlier_analysis_schema():
    """Test OUTLIER_ANALYSIS_SCHEMA properties and null type declarations."""
    schema = OUTLIER_ANALYSIS_SCHEMA["function"]
    assert schema["name"] == "outlier_analysis"
    assert "null" in schema["parameters"]["properties"]["column"]["type"]
