from typing import Any

import pandas as pd


def missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return the count and percentage of missing values per column.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    total_rows = len(df)
    null_counts = df.isnull().sum()

    columns_with_missing = {
        col: {
            "missing_count": int(count),
            "missing_percentage": round(
                float(count) / total_rows * 100, 2
            ) if total_rows > 0 else 0.0,
        }
        for col, count in null_counts.items()
        if count > 0
    }

    return {
        "total_rows": int(total_rows),
        "total_missing_values": int(null_counts.sum()),
        "columns_with_missing": columns_with_missing,
        "has_missing_values": bool(null_counts.sum() > 0),
    }


MISSING_VALUES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "missing_values",
        "description": (
            "Check for missing (null/NaN) values in the dataset. Returns the count "
            "and percentage of missing values for each column that has at least one. "
            "Use this when the user asks about missing data, null values, data quality, "
            "or completeness."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}