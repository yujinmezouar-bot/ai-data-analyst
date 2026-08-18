import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Standard dataset with numeric, categorical, and datetime columns."""
    return pd.DataFrame({
        "Store": ["A", "B", "C", "A", "B", "C"],
        "Date": pd.to_datetime([
            "2024-01-15",
            "2024-02-15",
            "2024-03-15",
            "2024-04-15",
            "2024-05-15",
            "2024-06-15",
        ]),
        "Weekly_Sales": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
        "Temperature": [45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        "Fuel_Price": [3.10, 3.20, 3.30, 3.40, 3.50, 3.60],
        "IsHoliday": [False, False, True, False, False, True],
    })


@pytest.fixture
def df_missing() -> pd.DataFrame:
    """Dataset with missing values and an all-null column."""
    return pd.DataFrame({
        "A": [1.0, np.nan, 3.0, 4.0, 5.0],
        "B": ["x", "y", None, "w", "z"],
        "C": [10.0, 20.0, 30.0, 40.0, 50.0],
        "All_Null": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })


@pytest.fixture
def df_outliers() -> pd.DataFrame:
    """Dataset with known IQR outliers."""
    return pd.DataFrame({
        "Normal": [10.0, 11.0, 10.5, 12.0, 9.5, 10.2, 11.1, 10.8, 10.0, 11.5],
        "With_Outlier": [10.0, 11.0, 10.5, 12.0, 9.5, 10.2, 11.1, 10.8, 10.0, 500.0],
        "Category": ["cat"] * 10,
    })


@pytest.fixture
def df_constants() -> pd.DataFrame:
    """Dataset with constant / zero-variance column."""
    return pd.DataFrame({
        "Constant": [5.0, 5.0, 5.0, 5.0, 5.0],
        "Varying_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Varying_2": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def df_multi_year() -> pd.DataFrame:
    """Dataset spanning 3 years (2023, 2024, 2025) for period comparisons."""
    dates = pd.to_datetime([
        "2023-01-01", "2023-06-01", "2023-12-01",
        "2024-01-01", "2024-06-01", "2024-12-01",
        "2025-01-01", "2025-06-01", "2025-12-01",
    ])
    return pd.DataFrame({
        "Date": dates,
        "Weekly_Sales": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0, 180.0, 280.0, 400.0],
        "Store": ["Store1", "Store2", "Store1", "Store1", "Store2", "Store1", "Store1", "Store2", "Store1"],
    })


@pytest.fixture
def df_zero_sales() -> pd.DataFrame:
    """Dataset with 0 value in the base period to test division-by-zero protection."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
        "Weekly_Sales": [0.0, 150.0],
    })


@pytest.fixture
def df_date_strings() -> pd.DataFrame:
    """Dataset containing raw date strings in multiple formats."""
    return pd.DataFrame({
        "Date_ISO": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "Date_US": ["01/15/2024", "02/15/2024", "03/15/2024", "04/15/2024"],
        "Date_Slash": ["2024/01/01", "2024/02/01", "2024/03/01", "2024/04/01"],
        "Not_A_Date": ["apple", "banana", "cherry", "date_fruit"],
        "Numeric_ID": [101, 102, 103, 104],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def df_identifiers() -> pd.DataFrame:
    """Dataset with clear identifier columns and normal categorical columns."""
    return pd.DataFrame({
        "user_id": [f"USR_{i:04d}" for i in range(20)],
        "order_code": [f"ORD_{i:04d}" for i in range(20)],
        "City": ["London", "Paris", "Tokyo", "London", "Paris"] * 4,
        "Revenue": [10.0 * i for i in range(20)],
    })


@pytest.fixture
def df_no_numeric() -> pd.DataFrame:
    """Dataset with only string/categorical and date columns."""
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Department": ["Sales", "HR", "Engineering"],
        "JoinDate": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    })


@pytest.fixture
def df_no_categorical() -> pd.DataFrame:
    """Dataset with only numeric and date columns."""
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "Sales": [100.0, 200.0, 300.0],
        "Units": [10, 20, 30],
    })


@pytest.fixture
def df_no_date() -> pd.DataFrame:
    """Dataset with only numeric and categorical columns (no dates)."""
    return pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Home"],
        "Price": [299.99, 49.99, 89.99],
        "Stock": [15, 50, 30],
    })

