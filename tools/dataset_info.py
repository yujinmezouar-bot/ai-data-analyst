from typing import Any

import pandas as pd


def dataset_info(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return basic structural information about the dataset.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "column_types": dtypes,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


DATASET_INFO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dataset_info",
        "description": (
            "Get basic structural information about the currently loaded dataset: "
            "number of rows, number of columns, column names, column data types, "
            "and memory usage. Use this when the user asks general questions about "
            "the dataset's shape or structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}