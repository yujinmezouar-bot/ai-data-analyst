from typing import Any

import pandas as pd


def statistics(
    df: pd.DataFrame,
    column: str | None = None,
) -> dict[str, Any]:
    """
    Calculate descriptive statistics for numeric columns.

    If `column` is provided, statistics are calculated only for that column.
    If `column` is None, statistics are calculated for all numeric columns.
    """

    if df is None:
        return {
            "error": "No dataset is loaded."
        }

    # ---------------------------------------------------------
    # If a specific column was requested
    # ---------------------------------------------------------

    if column is not None:

        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found in the dataset.",
                "available_columns": list(df.columns),
            }

        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "error": (
                    f"Column '{column}' is not numeric, "
                    "so descriptive statistics cannot be calculated."
                )
            }

        series = df[column].dropna()

        if series.empty:
            return {
                "error": f"Column '{column}' contains no valid numeric values."
            }

        return {
            "column": column,
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2)
            if len(series) > 1
            else 0.0,
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "q75": round(float(series.quantile(0.75)), 2),
        }

    # ---------------------------------------------------------
    # If no specific column was requested
    # ---------------------------------------------------------

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "message": "No numeric columns found in the dataset.",
            "non_numeric_columns": list(df.columns),
        }

    stats: dict[str, dict[str, float]] = {}

    for col in numeric_df.columns:

        series = numeric_df[col].dropna()

        if series.empty:
            continue

        stats[col] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2)
            if len(series) > 1
            else 0.0,
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "q75": round(float(series.quantile(0.75)), 2),
        }

    non_numeric_columns = [
        col for col in df.columns
        if col not in numeric_df.columns
    ]

    return {
        "numeric_columns_analyzed": list(stats.keys()),
        "statistics": stats,
        "non_numeric_columns": non_numeric_columns,
    }


STATISTICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "statistics",
        "description": (
            "Calculate descriptive statistics for numeric columns. "
            "Use this tool when the user asks for statistics about "
            "a specific numeric column, such as median, mean, minimum, "
            "maximum, standard deviation, quartiles, or descriptive "
            "statistics. For example: "
            "'what is the median Weekly_Sales?', "
            "'what is the average Fuel_Price?', "
            "'what is the minimum sales?', "
            "'what is the maximum Temperature?'. "
            "If the user asks for general descriptive statistics "
            "without specifying a column, analyze all numeric columns. "
            "If the question asks for a calculation BY or PER another "
            "column, use groupby_analysis instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": (
                        "The numeric column to analyze. "
                        "For example: 'Weekly_Sales', "
                        "'Fuel_Price', 'Temperature', or 'CPI'. "
                        "Omit this parameter when the user asks "
                        "for general descriptive statistics."
                    ),
                },
            },
            "required": [],
        },
    },
}
