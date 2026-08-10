from typing import Any

import pandas as pd

# Aggregation functions we allow the LLM to request.
# Keeping this as an explicit whitelist (not e.g. getattr(df, agg_function))
# means the LLM can never trigger an arbitrary pandas method by name.
ALLOWED_AGG_FUNCTIONS = {"mean", "sum", "count", "min", "max", "median", "std"}


def groupby_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    agg_function: str = "mean",
) -> dict[str, Any]:
    """
    Group the dataset by `group_column` and aggregate `value_column` using
    `agg_function` (e.g. mean, sum, count).

    Example: group_column="department", value_column="salary", agg_function="mean"
    is equivalent to: df.groupby("department")["salary"].mean()
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    if group_column not in df.columns:
        return {
            "error": f"Column '{group_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    if value_column not in df.columns:
        return {
            "error": f"Column '{value_column}' not found in the dataset.",
            "available_columns": list(df.columns),
        }

    if agg_function not in ALLOWED_AGG_FUNCTIONS:
        return {
            "error": f"Aggregation function '{agg_function}' is not supported.",
            "allowed_functions": sorted(ALLOWED_AGG_FUNCTIONS),
        }

    if agg_function != "count" and not pd.api.types.is_numeric_dtype(df[value_column]):
        return {
            "error": (
                f"Column '{value_column}' is not numeric, so '{agg_function}' "
                f"cannot be computed on it. Try 'count' instead, or pick a numeric column."
            ),
        }

    try:
        grouped = df.groupby(group_column)[value_column].agg(agg_function)
    except Exception as e:
        return {"error": f"Groupby operation failed: {e}"}

    result = {str(k): round(float(v), 2) for k, v in grouped.items()}

    return {
        "group_column": group_column,
        "value_column": value_column,
        "agg_function": agg_function,
        "result": result,
    }


GROUPBY_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "groupby_analysis",
        "description": (
            "Use this tool whenever the user asks for a calculation "
            "BY, PER, FOR EACH, or ACROSS categories or groups. "
            "Examples include: "
            "'average salary by department', "
            "'average weekly sales by store', "
            "'total sales per region', "
            "'median salary by department', "
            "'minimum sales for each store', "
            "'maximum sales per region', "
            "or 'number of orders per customer'. "
            "group_column is the column defining the groups. "
            "value_column is the column to aggregate. "
            "agg_function specifies the calculation such as mean, sum, "
            "count, min, max, median, or std."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_column": {
                    "type": "string",
                    "description": "The categorical column to group rows by, e.g. 'department' or 'region'.",
                },
                "value_column": {
                    "type": "string",
                    "description": "The numeric column to aggregate, e.g. 'salary' or 'amount'.",
                },
                "agg_function": {
                    "type": "string",
                    "enum": sorted(ALLOWED_AGG_FUNCTIONS),
                    "description": "The aggregation function to apply. Defaults to 'mean' if unsure.",
                },
            },
            "required": ["group_column", "value_column"],
        },
    },
}