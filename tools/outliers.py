from typing import Any

import pandas as pd


DEFAULT_IQR_MULTIPLIER = 1.5
MAX_COLUMNS_RETURNED = 20  # consistent cap philosophy with MAX_GROUPS_RETURNED
MAX_EXAMPLE_VALUES = 5


def _iqr_bounds(series: pd.Series, multiplier: float) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def _analyze_column(series: pd.Series, multiplier: float) -> dict[str, Any]:
    clean = series.dropna()
    observations = int(clean.count())

    if observations == 0:
        return {"error": "Column contains no valid numeric values."}

    if clean.nunique() <= 1:
        return {
            "observations": observations,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "note": "Column is constant (no variance); no outliers to detect.",
        }

    lower_bound, upper_bound = _iqr_bounds(clean, multiplier)
    outlier_mask = (clean < lower_bound) | (clean > upper_bound)
    outliers = clean[outlier_mask]
    outlier_count = int(outliers.count())

    result = {
        "observations": observations,
        "outlier_count": outlier_count,
        "outlier_percentage": round(100 * outlier_count / observations, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
    }

    if outlier_count > 0:
        # A few example values only -- never the full list of outlier rows.
        examples = outliers.sort_values(key=lambda s: (s - clean.median()).abs(), ascending=False)
        result["example_outlier_values"] = [
            round(float(v), 2) for v in examples.head(MAX_EXAMPLE_VALUES).tolist()
        ]

    return result


def outlier_analysis(
    df: pd.DataFrame,
    column: str | None = None,
    multiplier: float | None = None,
) -> dict[str, Any]:
    """
    Detect outliers in numeric columns using the IQR method:
    values below Q1 - k*IQR or above Q3 + k*IQR are flagged.

    - If `column` is given, analyzes just that column.
    - Otherwise, analyzes all numeric columns and returns a summary per
      column (never the raw outlier rows), so large datasets stay safe.
    """
    if df is None:
        return {"error": "No dataset is loaded."}

    k = multiplier if multiplier is not None else DEFAULT_IQR_MULTIPLIER
    if k <= 0:
        return {"error": "multiplier must be a positive number."}

    if column is not None:
        if column not in df.columns:
            return {"error": f"Column '{column}' not found in the dataset.", "available_columns": list(df.columns)}
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column '{column}' is not numeric, so outlier detection cannot be applied."}

        analysis = _analyze_column(df[column], k)
        if "error" in analysis:
            return {"error": f"Column '{column}': {analysis['error']}"}

        return {"method": "iqr", "multiplier": k, "column": column, **analysis}

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {"message": "No numeric columns found in the dataset.", "non_numeric_columns": list(df.columns)}

    per_column: dict[str, dict[str, Any]] = {}
    for col in numeric_df.columns:
        analysis = _analyze_column(numeric_df[col], k)
        if "error" not in analysis:
            per_column[col] = analysis

    if not per_column:
        return {"error": "Outlier detection could not be applied to any numeric column."}

    # Rank by outlier_percentage so the most affected columns surface first,
    # and cap how many columns are returned for very wide datasets.
    ranked = sorted(per_column.items(), key=lambda kv: kv[1]["outlier_percentage"], reverse=True)
    total_columns = len(ranked)
    truncated = total_columns > MAX_COLUMNS_RETURNED
    ranked = ranked[:MAX_COLUMNS_RETURNED]

    output = {
        "method": "iqr",
        "multiplier": k,
        "columns_analyzed": [c for c, _ in ranked],
        "results": {c: v for c, v in ranked},
    }
    if truncated:
        output["note"] = (
            f"Showing the {MAX_COLUMNS_RETURNED} of {total_columns} numeric columns with the "
            "highest outlier percentage. Ask about a specific column to see more detail."
        )
    return output


OUTLIER_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "outlier_analysis",
        "description": (
            "Detect outliers in numeric columns using the IQR (interquartile range) method. "
            "Use this for questions like 'are there outliers in sales?', 'find unusual values "
            "in quantity', or 'which numerical columns contain many outliers?'. Set 'column' "
            "to analyze one specific column, or leave it unset to analyze all numeric columns "
            "and rank them by how affected they are. Returns counts and percentages, not the "
            "full list of outlier rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": ["string", "null"],
                    "description": (
                        "A single numeric column to check for outliers, e.g. 'sales'. "
                        "Set to null (or omit) to analyze all suitable numeric columns."
                    ),
                },
                "multiplier": {
                    "type": "number",
                    "description": f"IQR multiplier controlling sensitivity. Defaults to {DEFAULT_IQR_MULTIPLIER} (standard Tukey's fences).",
                },
            },
            "required": [],
        },
    },
}