import pandas as pd
from tools.statistics import statistics, STATISTICS_SCHEMA


def test_statistics_single_column(sample_df: pd.DataFrame):
    """Test statistics for a single numeric column."""
    res = statistics(sample_df, column="Weekly_Sales")
    assert "error" not in res
    assert res["column"] == "Weekly_Sales"
    assert res["count"] == 6
    assert res["mean"] == 225.0
    assert res["median"] == 225.0
    assert res["min"] == 100.0
    assert res["max"] == 350.0
    assert res["q25"] == 162.5
    assert res["q75"] == 287.5


def test_statistics_all_columns(sample_df: pd.DataFrame):
    """Test statistics when column is None (analyzes all numeric columns)."""
    res = statistics(sample_df, column=None)
    assert "error" not in res
    assert "Weekly_Sales" in res["statistics"]
    assert "Temperature" in res["statistics"]
    assert "Fuel_Price" in res["statistics"]
    assert "Store" in res["non_numeric_columns"]


def test_statistics_invalid_column(sample_df: pd.DataFrame):
    """Test statistics with a non-existent column name."""
    res = statistics(sample_df, column="NonExistent")
    assert "error" in res
    assert "not found" in res["error"]
    assert "available_columns" in res


def test_statistics_non_numeric_column(sample_df: pd.DataFrame):
    """Test statistics on a categorical/non-numeric column."""
    res = statistics(sample_df, column="Store")
    assert "error" in res
    assert "not numeric" in res["error"]


def test_statistics_all_null_numeric_column():
    """Test statistics on a numeric column where all values are NaN."""
    df = pd.DataFrame({"empty_col": [None, None, None]}, dtype="float64")
    res = statistics(df, column="empty_col")
    assert "error" in res
    assert "no valid numeric values" in res["error"]


def test_statistics_no_numeric_columns():
    """Test statistics on a dataset with only string columns."""
    df = pd.DataFrame({"A": ["a", "b"], "B": ["c", "d"]})
    res = statistics(df)
    assert "message" in res
    assert "No numeric columns found" in res["message"]


def test_statistics_none_df():
    """Test statistics with df=None."""
    res = statistics(None)
    assert "error" in res
    assert "No dataset is loaded" in res["error"]


def test_statistics_schema():
    """Test STATISTICS_SCHEMA structure."""
    assert STATISTICS_SCHEMA["type"] == "function"
    assert STATISTICS_SCHEMA["function"]["name"] == "statistics"


def test_statistics_distribution_metrics(sample_df: pd.DataFrame):
    """Test IQR, skewness, and coefficient_of_variation calculations."""
    res = statistics(sample_df, column="Weekly_Sales")
    assert res["iqr"] == 125.0  # 287.5 - 162.5
    assert isinstance(res["skewness"], float)
    assert res["coefficient_of_variation"] is not None
    assert res["coefficient_of_variation"] > 0


def test_statistics_constant_column(df_constants: pd.DataFrame):
    """Test distribution metrics on a constant column."""
    res = statistics(df_constants, column="Constant")
    assert res["std"] == 0.0
    assert res["iqr"] == 0.0
    assert res["skewness"] == 0.0
    assert res["coefficient_of_variation"] is None


def test_statistics_zero_mean():
    """Test coefficient of variation returns None when mean is 0."""
    df = pd.DataFrame({"balanced": [-10.0, 0.0, 10.0]})
    res = statistics(df, column="balanced")
    assert res["mean"] == 0.0
    assert res["coefficient_of_variation"] is None

